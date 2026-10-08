"""Experiment orchestrator.

  python -m hddp.run <stage> [--dry] [--quiet-host]     stage in: smoke, pilot, verify, final, throttle

Launches run strictly one at a time, on the GPU sets derived by the env stage (results/env/topology.json),
only after those GPUs are verified idle, with a hard timeout after which torchrun and all of its workers are
terminated. Launch order (and condition order inside each launch) is randomized with a fixed seed. Results:
results/raw/<stage>.jsonl (+ <stage>.log). Completed launches are skipped on re-run (resumable); failed
launches are recorded, never dropped. Logs can contain paths of the local Python installation in third-party
warnings; run scripts/sanitize_data.py before publishing new data.
"""
import argparse
import gzip
import json
import os
import platform
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

from hddp import privacy

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
MODELS = ["resnet50", "vit_b_16"]
BUCKETS = [1, 4, 16, 25, 64, 128]
DELAYS = [0, 5, 10, 20, 40]
ALGOS = ["Auto", "Ring", "Tree"]
SEED = 20261007  # fixed shuffle seed of the recorded experiment; changing it changes every launch/condition order
CURRENT = None  # (Popen, launch token) of the active torchrun; all its processes are killed on SIGTERM/SIGHUP


def topo():
    return json.loads((ROOT / "results" / "env" / "topology.json").read_text())


def settings():
    return {k: v for k, v in topo()["settings"].items() if not k.endswith("_links")}


def L(stage, setting, gpus, model, algo, trial, conds, warmup=8, steps=30, prof=0, timeline=(), instrument=1,
      marker=0, tag="", straggler_rank=-1, profile_only=()):
    lid = f"{stage}|{setting}|{model}|{algo}|t{trial}" + (f"|{tag}" if tag else "")
    return dict(launch_id=lid, stage=stage, setting=setting, gpus=gpus, model=model, algo=algo, trial=trial,
                conditions=[list(c) for c in conds], warmup=warmup, steps=steps, profile_steps=prof,
                timeline=[list(t) for t in timeline], instrument=instrument, marker=marker, tag=tag,
                straggler_rank=straggler_rank, profile_only=[list(c) for c in profile_only], attempt="")


def grid(buckets, delays):
    return [(b, d) for b in buckets for d in delays]


def plan(stage, S):
    out = []
    if stage == "smoke":
        for m in MODELS:
            out.append(L(stage, "2gpu_nvlink", S["2gpu_nvlink"], m, "Auto", 0, [(25, 0), (25, 10)],
                         warmup=3, steps=5, prof=3, timeline=[(25, 10)]))
        out.append(L(stage, "2gpu_nvlink", S["2gpu_nvlink"], "resnet50", "Ring", 0, [(1, 0), (25, 0)],
                     warmup=2, steps=2, marker=1, tag="verify"))
    elif stage == "pilot":
        g = S["4gpu"]
        for m in MODELS:
            for t in range(2):
                out.append(L(stage, "4gpu", g, m, "Auto", t, grid([1, 25, 128], [0, 10, 40]),
                             prof=4 if t == 0 else 0, timeline=[(25, 0), (25, 40)] if t == 0 else ()))
                out.append(L(stage, "4gpu", g, m, "Auto", t, grid([1, 25, 128], [0]), instrument=0, tag="noinstr"))
            for algo in ("Ring", "Tree"):
                out.append(L(stage, "4gpu", g, m, algo, 0, grid([25], [0, 20])))
    elif stage == "verify":
        for s in ("2gpu_nvlink", "2gpu_pcie", "4gpu"):
            for m in MODELS:
                for algo in ALGOS:
                    out.append(L(stage, s, S[s], m, algo, 0, grid(BUCKETS, [0]), warmup=2, steps=2, marker=1))
    elif stage == "final":
        out += final_plan(S)
    return out


def final_plan(S):
    """Defined from the pilot (README, 'Final matrix'). 5 trials everywhere.
    4 GPU: Auto = full bucket x delay grid; Ring/Tree = all buckets x delays {0,10,40} (H3).
    2 GPU (NVLink, PCIe): delay sweep at the default 25 MB bucket, Auto.
    Straggler-placement control: 4 GPU, Auto, 25 MB, delay sweep with the straggler on rank 0 instead of rank 3.
    Profiler: every condition of trial 0; the 25 MB delay sweep in every trial (overlap CIs)."""
    out, T = [], 5
    sweep25 = grid([25], DELAYS)
    for m in MODELS:
        for t in range(T):
            out.append(L("final", "4gpu", S["4gpu"], m, "Auto", t, grid(BUCKETS, DELAYS), prof=4,
                         profile_only=() if t == 0 else sweep25,
                         timeline=[(25, 0), (25, 40)] if t == 0 else ()))
            for algo in ("Ring", "Tree"):
                out.append(L("final", "4gpu", S["4gpu"], m, algo, t, grid(BUCKETS, [0, 10, 40]),
                             prof=4 if t == 0 else 0))
            for s in ("2gpu_nvlink", "2gpu_pcie"):
                out.append(L("final", s, S[s], m, "Auto", t, sweep25, prof=4))
            out.append(L("final", "4gpu", S["4gpu"], m, "Auto", t, sweep25, prof=4, tag="strag0", straggler_rank=0))
    return out


# ---------------------------------------------------------------------------------------------- GPU state
PID_SALT = privacy.pid_salt()  # secret for this orchestrator run; foreign PIDs are stored only as pseudonyms


def gpu_state():
    """Per-GPU state and compute processes. A failing nvidia-smi raises instead of returning an empty process
    list, which would make busy GPUs look idle."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,temperature.gpu,power.draw,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60,
                       check=True).stdout
    gpus = {}
    for line in q.strip().splitlines():
        i, bus, temp, pw, util = [c.strip() for c in line.split(",")]
        gpus[bus.lower()[-12:]] = dict(index=int(i), temp=temp, power=pw, util=util, procs=[])
    a = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=60, check=True).stdout
    for line in a.strip().splitlines():
        bus, pid, name, mem = [c.strip() for c in line.split(",")]
        gpus[bus.lower()[-12:]]["procs"].append(privacy.proc_label(pid, name, mem, PID_SALT))
    return {g["index"]: g for g in gpus.values()}


# ---------------------------------------------------------------------------------------------- launching
class Watch(threading.Thread):
    """Polls the NVML process list of the launch's GPUs every second for the whole launch. Any PID outside the
    launch's own processes is a foreign process (the worker's own check samples only at window ends).
    Sightings are stored as [seconds since launch start, GPU, PID pseudonym]."""

    def __init__(self, p, gpus, token):
        super().__init__(daemon=True)
        from hddp import nvml
        nvml.init()
        self.nvml, self.p, self.token, self.h = nvml, p, token, {g: nvml.handle(g) for g in gpus}
        self.events, self.halt, self.t0 = [], threading.Event(), time.time()
        self.errors = 0  # failed NVML polls; stored with the launch so a blind watch is visible in the data
        self.start()

    def run(self):
        ours, foreign = set(), set()
        while not self.halt.wait(1.0):
            for g, h in self.h.items():
                try:
                    new = set(self.nvml.compute_pids(h)) - ours - foreign
                except self.nvml.NVMLError:
                    self.errors += 1
                    continue
                if new:
                    mine = _launch_pids(self.p, self.token)
                    ours |= new & mine
                    for pid in new - mine:
                        foreign.add(pid)
                        self.events.append([round(time.time() - self.t0), g, privacy.pid_pseudonym(pid, PID_SALT)])


def launch(Lx, out, log, extra_env=None):
    n = len(Lx["gpus"])
    token = privacy.new_attempt_id() + privacy.new_attempt_id()
    env = os.environ.copy()
    env.pop("NCCL_ALGO", None)
    env.pop("NCCL_P2P_LEVEL", None)
    env.update(topo()["nccl_env"])  # e.g. NCCL_P2P_LEVEL=NVL when PCIe P2P failed the env-stage check
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(map(str, Lx["gpus"])),
               OMP_NUM_THREADS="4", NCCL_DEBUG="WARN", PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1",
               HDDP_PID_SALT=PID_SALT, HDDP_LAUNCH_TOKEN=token)
    if Lx["algo"] != "Auto":
        env["NCCL_ALGO"] = Lx["algo"]
    env.update(extra_env or {})
    meta = {k: Lx[k] for k in ("launch_id", "stage", "setting", "gpus", "algo", "trial", "tag", "attempt")}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n}",
           "-m", "hddp.worker", "--model", Lx["model"], "--conditions", json.dumps(Lx["conditions"]),
           "--warmup", str(Lx["warmup"]), "--steps", str(Lx["steps"]), "--profile-steps", str(Lx["profile_steps"]),
           "--timeline", json.dumps(Lx["timeline"]), "--instrument", str(Lx["instrument"]),
           "--straggler-rank", str(Lx["straggler_rank"]), "--profile-only", json.dumps(Lx["profile_only"]),
           "--marker", str(Lx["marker"]), "--out", str(out), "--meta", json.dumps(meta)]
    per_cond = (Lx["warmup"] + Lx["steps"] + Lx["profile_steps"]) * 0.5 + 15
    timeout = 240 + len(Lx["conditions"]) * per_cond
    log.write(f"\n===== {Lx['launch_id']}  timeout={timeout:.0f}s\n$ "
              f"{privacy.redact_text(' '.join(cmd), [platform.node()])}\n")
    log.flush()
    global CURRENT
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    CURRENT = (p, token)
    watch = None
    try:  # everything after Popen is inside the try, so no failure can leave the workers running
        watch = Watch(p, Lx["gpus"], token)
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        rc = "timeout"
    except BaseException:
        _kill(p, token)
        CURRENT = None
        raise
    finally:
        if watch is not None:
            watch.halt.set()
    Lx["foreign_watch"] = watch.events
    Lx["foreign_watch_errors"] = watch.errors
    _kill(p, token)  # also on success: reaps any worker that outlived torchrun
    CURRENT = None
    return rc


def _tree(pid):
    """pid + all descendants (torchrun puts workers in their own sessions, so killpg is not enough)."""
    ps = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True).stdout.split()
    kids = {}
    for c, par in zip(ps[::2], ps[1::2]):
        kids.setdefault(int(par), []).append(int(c))
    out, stack = [], [pid]
    while stack:
        x = stack.pop()
        out.append(x)
        stack += kids.get(x, [])
    return out


def _marked(token):
    """PIDs of this user's processes that inherited the launch token. Unlike the process tree, this still finds
    workers that were re-parented to init after torchrun died."""
    needle, out = f"HDDP_LAUNCH_TOKEN={token}".encode(), set()
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if needle in (d / "environ").read_bytes().split(b"\0"):
                    out.add(int(d.name))
            except OSError:  # other users' processes, or already gone
                pass
    return out


def _launch_pids(p, token):
    """torchrun's process tree (only while torchrun is unreaped, so its PID cannot have been reused) plus every
    process carrying the launch token."""
    return (set(_tree(p.pid)) if p.poll() is None else set()) | _marked(token)


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"  # zombies are already dead
    except OSError:
        return False


def _kill(p, token):
    """Terminate torchrun and every worker of this launch: SIGTERM, wait up to 20 s, then SIGKILL."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        procs = {x for x in _launch_pids(p, token) if x != os.getpid()}
        for pid in procs:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        t0 = time.time()
        while time.time() - t0 < 20 and any(_alive(x) for x in procs):
            p.poll()
            time.sleep(0.5)
    p.poll()


# ---------------------------------------------------------------------------------------------- NCCL logs
COLL_RE = re.compile(r"NCCL INFO (\w+): opCount \w+ .*?count (\d+) datatype (\d+)")
TUNE_RE = re.compile(r"NCCL INFO (\d+) Bytes -> Algo (\d+) proto (\d+)")
ALGO_NAMES = {0: "Tree", 1: "Ring", 2: "CollNetDirect", 3: "CollNetChain", 4: "NVLS", 5: "NVLSTree"}
PROTO_NAMES = {0: "LL", 1: "LL128", 2: "Simple"}


def parse_nccl_logs(files, conditions):
    """Pair each COLL line with the following TUNING line (same process) -> algorithm actually selected."""
    per_cond = {i: Counter() for i in range(len(conditions))}
    env_lines, transports = set(), Counter()
    for fpath in files:
        cur, pending = None, None
        with open(fpath, errors="replace") as fh:
            for line in fh:
                if "set by environment" in line:
                    env_lines.add(line.split("NCCL INFO ")[-1].strip())
                m = re.search(r"via (\S+)", line)
                if "Channel" in line and m:
                    transports[m.group(1)] += 1
                m = COLL_RE.search(line)
                if m:
                    op, count = m.group(1), int(m.group(2))
                    if op == "Broadcast" and 101 <= count < 101 + len(conditions):
                        cur, pending = count - 101, None  # condition marker (uint8 broadcast, logged as datatype 0)
                    else:
                        pending = op if count > 1 else None  # count==1 AllReduce is dist.barrier()
                    continue
                m = TUNE_RE.search(line)
                if m and pending and cur is not None:
                    key = f"{pending}:{ALGO_NAMES.get(int(m.group(2)))}/{PROTO_NAMES.get(int(m.group(3)))}"
                    per_cond[cur][key] += 1
                    pending = None
    return per_cond, sorted(env_lines), dict(transports)


def run_verify_launch(Lx, out, log):
    with tempfile.TemporaryDirectory() as td:
        rc = launch(Lx, out, log, extra_env={"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,ENV,TUNING,COLL",
                                              "NCCL_DEBUG_FILE": f"{td}/nccl.%p.log"})
        files = sorted(Path(td).glob("nccl.*.log"))
        per_cond, env_lines, transports = parse_nccl_logs(files, Lx["conditions"])
        with gzip.open(RAW / "nccl_logs.txt.gz", "at") as gz:
            for fpath in files:
                gz.write(f"\n##### {Lx['launch_id']} {fpath.name}\n")
                gz.write(fpath.read_text(errors="replace"))
    with open(out, "a") as f:
        for i, (b, _delay) in enumerate(Lx["conditions"]):
            f.write(json.dumps({"type": "nccl_verify", "launch_id": Lx["launch_id"], "setting": Lx["setting"],
                                "model": Lx["model"], "algo_requested": Lx["algo"], "bucket": b,
                                "selected": dict(per_cond[i]), "env_lines": env_lines,
                                "transports": transports}) + "\n")
    return rc


# ---------------------------------------------------------------------------------------------- throttle
THROTTLE_WATTS = (200, 150)


def throttle_plan(S):
    g = S["4gpu"]
    return g[-1], [(watts, L("throttle", "4gpu", g, m, "Auto", 0, grid([25], [0]), tag=f"pl{watts}"))
                   for watts in THROTTLE_WATTS for m in MODELS]


def run_throttle(S, out, logp, dry=False):
    """Optional hardware heterogeneity: power-cap ONE GPU, with best-effort restoration of the original limit
    after every run (normal exit, exceptions, SIGINT/SIGTERM/SIGHUP; nothing can restore it after SIGKILL, a
    host crash or power loss). With dry=True nothing is probed, applied or written: the plan is only printed."""
    target, runs = throttle_plan(S)
    if dry:
        print(f"throttle (dry run): would probe power-limit permission on GPU {target}, then run:")
        for watts, Lx in runs:
            print(f"  {Lx['launch_id']} gpus={Lx['gpus']} power_limit={watts} W on GPU {target}")
        return
    from hddp import nvml
    ok, why = nvml.can_set_power_limit(target)
    rec = {"type": "throttle_probe", "gpu": target, "permitted": ok, "detail": why}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    if not ok:
        return
    with open(logp, "a") as log:
        for watts, Lx in runs:
            Lx["attempt"] = privacy.new_attempt_id()
            with nvml.ThrottleGuard(target, power_w=watts) as tg:
                rc = launch(Lx, out, log)
                applied = tg.applied
            with open(out, "a") as f:
                f.write(json.dumps({"type": "launch", **Lx, "rc": rc, "status": "ok" if rc == 0 else "failed",
                                    "throttle_applied": applied,
                                    "restored_limit_w": nvml.power_limit_w(nvml.handle(target))}) + "\n")


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "pilot", "verify", "final", "throttle"])
    ap.add_argument("--dry", action="store_true",
                    help="print the plan only: no GPU access, no hardware changes, no files written")
    ap.add_argument("--quiet-host", action="store_true",
                    help="start a launch only when NO GPU of the machine has a foreign process (other tenants' jobs on "
                         "other GPUs were measured to degrade the host-memory transport)")
    args = ap.parse_args()

    def on_term(signum, _frame):
        # SIGINT arrives as KeyboardInterrupt, which launch() handles; SIGTERM/SIGHUP end up here
        if CURRENT is not None:
            _kill(*CURRENT)
        sys.exit(128 + signum)

    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):  # SIGHUP does not exist on Windows (dry runs and analysis work there)
            signal.signal(getattr(signal, name), on_term)
    S = settings()
    out, logp = RAW / f"{args.stage}.jsonl", RAW / f"{args.stage}.log"
    if args.stage == "throttle":
        if not args.dry:
            RAW.mkdir(parents=True, exist_ok=True)
        run_throttle(S, out, logp, dry=args.dry)
        return

    launches = plan(args.stage, S)
    rng = random.Random(f"{SEED}-{args.stage}")
    rng.shuffle(launches)  # randomized trial ordering
    for Lx in launches:
        random.Random(f"{SEED}-{Lx['launch_id']}").shuffle(Lx["conditions"])
    done = clean_launch_ids(out)
    todo = [x for x in launches if x["launch_id"] not in done]
    print(f"{args.stage}: {len(launches)} launches, {len(todo)} to run, "
          f"{sum(len(x['conditions']) for x in todo)} conditions")
    if args.dry:
        for x in todo:
            print(x["launch_id"], x["gpus"], len(x["conditions"]))
        return
    RAW.mkdir(parents=True, exist_ok=True)

    # Queue semantics: a launch whose GPUs are busy is deferred (others proceed); a launch during which a foreign
    # process appeared on its GPUs is marked "contaminated" (excluded by the analysis) and re-queued.
    queue, retries, t_block = list(todo), {}, None
    with open(logp, "a") as log:
        while queue:
            st = gpu_state()
            busy_any = any(v["procs"] for v in st.values())
            ready = [x for x in queue if not any(st[g]["procs"] for g in x["gpus"])
                     and not (args.quiet_host and busy_any)]
            if not ready:
                t_block = t_block or time.time()
                if time.time() - t_block > MAX_BLOCK_S:
                    for x in queue:
                        record(out, x, "skipped_busy", "skipped_busy", st, st, 0)
                    print(f"giving up on {len(queue)} launches: GPUs busy > {MAX_BLOCK_S}s", flush=True)
                    break
                log.write(f"[wait] all {len(queue)} remaining launches blocked by foreign GPU processes\n")
                log.flush()
                time.sleep(60)
                continue
            t_block = None
            Lx = ready[0]
            queue.remove(Lx)
            Lx["attempt"] = privacy.new_attempt_id()
            t0 = time.time()
            rc = run_verify_launch(Lx, out, log) if Lx["marker"] else launch(Lx, out, log)
            post = gpu_state()
            foreign = foreign_in_attempt(out, Lx) or any(post[g]["procs"] for g in Lx["gpus"]) or \
                bool(Lx.get("foreign_watch"))
            status = "failed" if rc != 0 else ("contaminated" if foreign else "ok")
            record(out, Lx, rc, status, st, post, time.time() - t0)
            if status == "ok" and not Lx["marker"] and Lx["launch_id"] not in clean_launch_ids(out):
                status = "contaminated"  # timing anomaly or baseline outlier (host-side interference)
                record(out, Lx, rc, "anomalous_timing", st, post, 0)
            print(f"[{len(todo) - len(queue)}/{len(todo)}] {Lx['launch_id']} -> {status} ({time.time() - t0:.0f}s)",
                  flush=True)
            if status == "contaminated" and retries.get(Lx["launch_id"], 0) < MAX_RETRIES:
                retries[Lx["launch_id"]] = retries.get(Lx["launch_id"], 0) + 1
                queue.append(Lx)


MAX_BLOCK_S = 4 * 3600
MAX_RETRIES = 8


def record(out, Lx, rc, status, pre, post, dur):
    with open(out, "a") as f:
        f.write(json.dumps({"type": "launch", **Lx, "rc": rc, "status": status, "duration_s": round(dur, 1),
                            "gpu_pre": pre, "gpu_post": post, "loadavg": os.getloadavg()}) + "\n")


def clean_launch_ids(out):
    """Launch ids with at least one clean, completed attempt (same rules as the analysis)."""
    if not out.exists():
        return set()
    from hddp.analyze import attempt_flags
    with open(out) as f:
        rows = [json.loads(l) for l in f]
    return {lid for lid, _ in attempt_flags(rows)["good"]} if any(r["type"] == "cond" for r in rows) else \
        {r["launch_id"] for r in rows if r["type"] == "launch" and r["status"] == "ok"}


def foreign_in_attempt(out, Lx):
    with open(out) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "cond" and r.get("attempt") == Lx["attempt"] and \
                    any(k["foreign_pids"] for k in r["ranks"]):
                return True
    return False


if __name__ == "__main__":
    main()
