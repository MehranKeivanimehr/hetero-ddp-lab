"""Machine characterization + `nvidia-smi topo -m` parser that derives topology-aware GPU groupings.

python -m hddp.env --gpus 0,1,2,3   ->  results/env/machine.txt, results/env/topology.json
Nothing here assumes NVLink exists; groupings are derived only from the detected matrix.
"""
import argparse
import itertools
import json
import platform
import re
import os
import socket
import subprocess
import sys
from pathlib import Path

from hddp import privacy

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "results" / "env"

# The output is redacted before it is written (host name, user paths, GPU UUIDs, clock time, CPU vulnerability
# status). Other tenants' processes are deliberately not listed.
COMMANDS = [
    ["nvidia-smi"],
    ["nvidia-smi", "topo", "-m"],
    ["nvidia-smi", "nvlink", "-s"],
    ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,driver_version,pcie.link.gen.max,pcie.link.width.max,"
     "power.limit,power.default_limit,clocks.max.sm,clocks.max.mem", "--format=csv"],
    ["nvcc", "--version"],
    ["lscpu"],
    ["numactl", "-H"],
    ["lspci", "-tv"],
]

# nvidia-smi topo legend, ordered from closest to farthest
LINK_CLASS = {"PIX": "pcie_switch", "PXB": "pcie_switch", "PHB": "pcie_host_bridge",
              "NODE": "pcie_same_numa", "SYS": "pcie_cross_numa"}


def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return p.stdout + (p.stderr if p.returncode else "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"<unavailable: {e}>"


def parse_topo(text):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [l for l in text.splitlines() if l.strip()]
    header = next(l for l in lines if l.lstrip().startswith("GPU0"))
    cols = header.split("\t")
    cols = [c.strip() for c in cols if c.strip()]
    gpus, link, cpu_aff, numa = [], {}, {}, {}
    for l in lines:
        m = re.match(r"^GPU(\d+)\t", l)
        if not m:
            continue
        cells = [c.strip() for c in l.split("\t")[1:]]
        i = int(m.group(1))
        gpus.append(i)
        for name, cell in zip(cols, cells):
            if re.fullmatch(r"GPU\d+", name):
                link[(i, int(name[3:]))] = cell
        row = dict(zip(cols, cells))
        cpu_aff[i] = row.get("CPU Affinity")
        numa[i] = row.get("NUMA Affinity")
    return gpus, link, cpu_aff, numa


def classify(cell):
    if cell == "X":
        return "self"
    if cell.startswith("NV"):
        return "nvlink"
    return LINK_CLASS.get(cell, "unknown")


def derive(gpus, link, numa, allowed):
    pairs = {}
    for a, b in itertools.combinations(gpus, 2):
        pairs.setdefault(classify(link[(a, b)]), []).append([a, b])
    al = [g for g in gpus if g in allowed]
    settings = {}
    pick = lambda cls: next((p for p in pairs.get(cls, []) if set(p) <= set(al)), None)
    if pick("nvlink"):
        settings["2gpu_nvlink"] = pick("nvlink")
    for cls in ("pcie_switch", "pcie_host_bridge", "pcie_same_numa"):
        if pick(cls):
            settings["2gpu_pcie"] = pick(cls)
            break
    if pick("pcie_cross_numa"):
        settings["2gpu_cross_numa"] = pick("pcie_cross_numa")
    for n in (4, 8):
        same_numa = [list(c) for c in itertools.combinations(al, n) if len({numa[g] for g in c}) == 1]
        cand = same_numa[0] if same_numa else (al[:n] if len(al) >= n else None)
        if cand:
            classes = sorted({classify(link[(a, b)]) for a, b in itertools.combinations(cand, 2)})
            settings[f"{n}gpu"] = cand
            settings[f"{n}gpu_links"] = classes
    return pairs, settings


P2P_ELEMENTS = 10_000_000  # 40 MB of float32 per rank
P2P_PERIOD = 1021  # pattern period; keeps every partial sum an exactly representable small integer


def p2p_pattern(rank, n=P2P_ELEMENTS, xp=None):
    """Rank-dependent input of the correctness probe: (i mod 1021) + rank + 1.

    Unlike a tensor of ones, a position-dependent pattern also exposes stale, shifted or partially copied
    buffers. All values and all sums for up to 8 ranks stay below 2**24, so float32 arithmetic is exact and
    the result can be compared for equality."""
    if xp is None:
        import numpy as xp
    return xp.remainder(xp.arange(n, dtype=xp.float32), P2P_PERIOD) + (rank + 1)


def p2p_expected(world, n=P2P_ELEMENTS, xp=None):
    if xp is None:
        import numpy as xp
    return xp.remainder(xp.arange(n, dtype=xp.float32), P2P_PERIOD) * world + world * (world + 1) / 2


def p2p_worker():
    """torchrun target: one all-reduce of the probe pattern; EVERY rank checks its FULL result tensor."""
    import datetime
    import time
    import torch
    import torch.distributed as dist
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = dist.get_rank(), dist.get_world_size()
    x = p2p_pattern(rank, xp=torch).cuda()
    expected = p2p_expected(world, xp=torch).cuda()
    torch.cuda.synchronize()
    t = time.time()
    dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = time.time() - t
    bad = (x != expected).nonzero()
    checksum, ref = float(x.double().sum()), float(expected.double().sum())
    print(f"P2PCHECK rank={rank} world={world} mismatches={bad.numel()} "
          f"first_bad={int(bad[0]) if bad.numel() else -1} checksum={checksum:.1f} expected_checksum={ref:.1f} "
          f"first_allreduce_s={dt:.2f}", flush=True)
    dist.destroy_process_group()


P2P_LINE_RE = re.compile(r"P2PCHECK rank=(\d+) world=(\d+) mismatches=(\d+) first_bad=(-?\d+) "
                         r"checksum=([\d.]+) expected_checksum=([\d.]+) first_allreduce_s=([\d.]+)")


def parse_p2p_output(out, world):
    """Summarize the per-rank P2PCHECK lines. The probe passes only if every rank reported, no element of any
    rank's result differs from the expected value, and every checksum matches."""
    ranks = {}
    for m in P2P_LINE_RE.finditer(out):
        ranks[int(m.group(1))] = dict(mismatches=int(m.group(3)), first_bad=int(m.group(4)),
                                      checksum_ok=float(m.group(5)) == float(m.group(6)),
                                      first_allreduce_s=float(m.group(7)))
    complete = sorted(ranks) == list(range(world))
    correct = complete and all(r["mismatches"] == 0 and r["checksum_ok"] for r in ranks.values())
    if correct:
        failure = None
    elif not ranks:
        failure = "no rank reported a result (hang, timeout or crash)"
    elif not complete:
        failure = f"only ranks {sorted(ranks)} of {world} reported"
    else:
        failure = "wrong all-reduce result: " + ", ".join(
            f"rank {k}: {r['mismatches']} mismatched elements (first at index {r['first_bad']})"
            + ("" if r["checksum_ok"] else ", checksum differs") for k, r in sorted(ranks.items())
            if r["mismatches"] or not r["checksum_ok"])
    return dict(correct=correct, failure=failure, ranks_reported=len(ranks),
                mismatched_elements=sum(r["mismatches"] for r in ranks.values()),
                first_allreduce_s=max((r["first_allreduce_s"] for r in ranks.values()), default=None))


def p2p_check(gpus, level, kind="pair", link_class=None):
    """Correctness/latency probe of one NCCL all-reduce over a GPU set with a given NCCL_P2P_LEVEL
    (None = NCCL default). The record contains logical GPU indices only, no host identifiers."""
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
           "OMP_NUM_THREADS": "2", "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT", "PYTHONPATH": str(ROOT)}
    env.pop("NCCL_P2P_LEVEL", None)
    if level:
        env["NCCL_P2P_LEVEL"] = level
    rc = None
    try:
        p = subprocess.run(["timeout", "90", sys.executable, "-m", "torch.distributed.run", "--standalone",
                            f"--nproc_per_node={len(gpus)}", "-m", "hddp.env", "--p2p-worker"], env=env, cwd=ROOT,
                           capture_output=True, text=True, timeout=120)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = f"timeout {e}"
    res = {"gpus": list(gpus), "kind": kind, "link_class": link_class, "nccl_p2p_level": level or "default",
           "probe": "full-tensor pattern all-reduce, checked on every rank",
           "transport": sorted({m.group(1) for m in re.finditer(r"via (\S+)", out)}), "returncode": rc}
    res.update(parse_p2p_output(out, len(gpus)))
    return res


def p2p_targets(gpus, link, allowed, settings):
    """Every non-NVLink pair of allowed GPUs, plus each multi-GPU setting as a whole (mixed topologies)."""
    al = [g for g in gpus if g in allowed]
    out = [(list(p), "pair", classify(link[p])) for p in itertools.combinations(al, 2) if classify(link[p]) != "nvlink"]
    out += [(g, "group", "+".join(settings.get(f"{name}_links", []))) for name, g in settings.items()
            if not name.endswith("_links") and not name.startswith("2gpu") and isinstance(g, list)]
    return out


def versions():
    import torch
    import torchvision
    return {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "nccl": ".".join(map(str, torch.cuda.nccl.version())), "kernel": platform.release()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3", help="GPUs this study may use (shared machine)")
    ap.add_argument("--p2p-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--test-default-p2p", action="store_true",
                    help="also probe NCCL's DEFAULT P2P path over PCIe. Off by default: on the study host that path "
                         "returned wrong all-reduce results after a 30-40 s stall (results/env/topology.json), and "
                         "exercising a faulty P2P path on a shared machine is not harmless")
    args = ap.parse_args()
    if args.p2p_worker:
        return p2p_worker()
    allowed = [int(x) for x in args.gpus.split(",")]
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    hosts = [platform.node(), socket.getfqdn()]
    text = "".join(f"\n===== $ {' '.join(cmd)}\n{_run(cmd)}" for cmd in COMMANDS)
    (ENV_DIR / "machine.txt").write_text(privacy.redact_machine_text(text, hosts), newline="\n")
    gpus, link, cpu_aff, numa = parse_topo(_run(["nvidia-smi", "topo", "-m"]))
    pairs, settings = derive(gpus, link, numa, allowed)
    # NCCL correctness check on every non-NVLink pair of allowed GPUs and on every multi-GPU setting
    # (do not assume PCIe P2P works). By default only the safe configuration (P2P over NVLink only, PCIe via
    # host memory) is checked and used; NCCL's default PCIe P2P path is exercised only with --test-default-p2p.
    checks = []
    for g, kind, cls in p2p_targets(gpus, link, allowed, settings):
        levels = ([None] if args.test_default_p2p else []) + ["NVL"]
        checks += [p2p_check(g, lv, kind, cls) for lv in levels]
    for c in checks:
        if not c["correct"]:
            print(f"P2P CHECK FAILED: GPUs {c['gpus']} ({c['kind']}, {c['link_class']}), "
                  f"NCCL_P2P_LEVEL={c['nccl_p2p_level']}: {c['failure']}", file=sys.stderr)
    defaults = [c for c in checks if c["nccl_p2p_level"] == "default"]
    default_ok = bool(defaults) and all(c["correct"] and c["first_allreduce_s"] < 5 for c in defaults)
    nccl_env = {} if default_ok else {"NCCL_P2P_LEVEL": "NVL"}
    out = {"versions": versions(), "allowed_gpus": allowed, "gpus": gpus, "p2p_check": checks, "nccl_env": nccl_env,
           "matrix": {f"{a}-{b}": v for (a, b), v in link.items() if a != b},
           "cpu_affinity": cpu_aff, "numa": numa, "pairs_by_class": pairs, "settings": settings}
    (ENV_DIR / "topology.json").write_text(json.dumps(out, indent=1), newline="\n")
    print(json.dumps({k: out[k] for k in ("versions", "settings", "p2p_check", "nccl_env")}, indent=1))
    if any(not c["correct"] for c in checks if c["nccl_p2p_level"] != "default"):
        sys.exit("the configuration used for the experiments failed the correctness probe; do not run experiments")


if __name__ == "__main__":
    main()
