"""torchrun worker. One launch = one (GPU set, model, NCCL algo, trial); it runs a (pre-shuffled) list of
(bucket_cap_mb, delay_ms) conditions. Every condition gets a freshly built model/DDP/optimizer, a warmup
(during which the straggler's CUDA spin delay is closed-loop calibrated), a timed window, and optionally
a short torch.profiler window used only for the overlap estimate. Rank 0 appends one JSON line per
condition (all ranks' data gathered) to --out.
"""
import argparse
import datetime
import gc
import json
import os
import threading
import time
from collections import Counter

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP

from hddp import nvml, privacy

BATCH = {"resnet50": 64, "vit_b_16": 32}  # fixed per-GPU batch


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(BATCH))
    ap.add_argument("--conditions", required=True, help="JSON list of [bucket_cap_mb, delay_ms]")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--profile-steps", type=int, default=0)
    ap.add_argument("--profile-only", default="[]", help="JSON list of [bucket, delay] to profile ([] = all)")
    ap.add_argument("--timeline", default="[]", help="JSON list of [bucket, delay] whose kernel timeline is kept")
    ap.add_argument("--straggler-rank", type=int, default=-1, help="-1 -> last rank")
    ap.add_argument("--instrument", type=int, default=1, help="comm-hook bucket events + per-step NVML reads")
    ap.add_argument("--marker", type=int, default=0, help="uint8 broadcast marker per condition (NCCL log parsing)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="{}", help="JSON merged into every record")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class HookState:
    def __init__(self):
        self.events = []


def timed_allreduce(state, bucket):
    """Default allreduce hook + a CUDA event marking when this rank's bucket became ready."""
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    state.events.append(ev)
    return default_hooks.allreduce_hook(None, bucket)


class Sampler(threading.Thread):
    """Background NVML poller (ctypes releases the GIL during NVML calls). Samples are wall-clock stamped and
    later mapped onto GPU step intervals, because the host loop runs ahead of the GPU.
    It deliberately never reads the cumulative energy counter: in a diagnostic on the study's driver,
    querying that counter every 5-100 ms coincided with severe under-reporting of energy (cause unknown;
    see docs/limitations.md). Energy is read only at window edges by the main thread."""

    FIELDS = ("power_w", "temp_c", "sm_mhz", "util_pct")

    def __init__(self, h, period_s=0.02):
        super().__init__(daemon=True)
        self.h, self.period, self.buf, self.active = h, period_s, [], False
        self.start()

    def run(self):
        h = self.h
        while True:
            if self.active:
                self.buf.append((time.perf_counter(), nvml.power_instant_w(h), nvml.temp_c(h), nvml.sm_clock_mhz(h),
                                 nvml.util_pct(h)))
            time.sleep(self.period)


def build(model, bucket, dev, hook_state):
    torch.manual_seed(0)
    with torch.device(dev):
        m = torchvision.models.resnet50() if model == "resnet50" else torchvision.models.vit_b_16()
    if model == "resnet50":
        m = m.to(memory_format=torch.channels_last)
    ddp = DDP(m, device_ids=[dev.index], bucket_cap_mb=bucket)
    if hook_state is not None:
        ddp.register_comm_hook(hook_state, timed_allreduce)
    if model == "resnet50":
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    return ddp, opt


def new_events():
    return [torch.cuda.Event(enable_timing=True) for _ in range(5)]


def train_step(ddp, opt, x, y, cycles, ev):
    ev[0].record()
    if cycles:
        torch.cuda._sleep(cycles)  # GPU-side spin on this rank's compute stream (no host sleep)
    ev[1].record()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(ddp(x), y)
    ev[2].record()
    loss.backward()  # returns after DDP has waited for all bucket allreduces
    ev[3].record()
    opt.step()
    opt.zero_grad(set_to_none=True)
    ev[4].record()


def _union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _intersect(a, b):
    i = j = tot = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        tot += max(0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def kernel_class(name):
    if "nccl" in name.lower():
        return "n"
    if "spin_kernel" in name:
        return "d"
    return "c"


def overlap_metrics(kernels, steps):
    """kernels: list of (start_ns, end_ns, name, stream). See docs/methodology.md (overlap definition)."""
    by = {"n": [], "c": [], "d": []}
    for s, e, name, _ in kernels:
        by[kernel_class(name)].append((s, e))
    un, uc = _union(by["n"]), _union(by["c"])
    comm = sum(e - s for s, e in un)
    comp = sum(e - s for s, e in uc)
    ov = _intersect(un, uc)
    span = (max(k[1] for k in kernels) - min(k[0] for k in kernels)) if kernels else 0
    ms = lambda ns: round(ns / steps / 1e6, 4)
    names = Counter(k[2].split("(")[0] for k in kernels if kernel_class(k[2]) == "n")
    return {"comm_ms": ms(comm), "compute_ms": ms(comp), "overlap_ms": ms(ov), "exposed_comm_ms": ms(comm - ov),
            "overlap_ratio": round(ov / comm, 4) if comm else None, "span_ms": ms(span),
            "delay_kernel_ms": ms(sum(e - s for s, e in _union(by["d"]))), "nccl_kernels": dict(names)}


def main():
    a = parse()
    meta = json.loads(a.meta)
    conditions = json.loads(a.conditions)
    timeline_set = {tuple(c) for c in json.loads(a.timeline)}
    profile_set = {tuple(c) for c in json.loads(a.profile_only)}
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    straggler = world - 1 if a.straggler_rank < 0 else a.straggler_rank
    torch.backends.cudnn.benchmark = True

    # NVML handle for the *physical* GPU behind this rank (verified by UUID) + NUMA-local CPU affinity
    phys = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[local])
    nvml.init()
    h = nvml.handle(phys)
    assert nvml.uuid(h).endswith(str(torch.cuda.get_device_properties(local).uuid)), "CUDA/NVML index mismatch"
    cpus = []
    for part in open(f"/sys/bus/pci/devices/{nvml.pci_bus_id(h)}/local_cpulist").read().strip().split(","):
        lo, _, hi = part.partition("-")
        cpus += range(int(lo), int(hi or lo) + 1)
    os.sched_setaffinity(0, cpus)

    # fixed synthetic inputs (never change across conditions)
    B = BATCH[a.model]
    g = torch.Generator(device=dev).manual_seed(a.seed * 1000 + rank)
    x = torch.randn(B, 3, 224, 224, device=dev, generator=g)
    y = torch.randint(0, 1000, (B,), device=dev, generator=g)
    if a.model == "resnet50":
        x = x.contiguous(memory_format=torch.channels_last)

    job_pids = [None] * world
    dist.all_gather_object(job_pids, os.getpid())
    salt = privacy.pid_salt()  # shared by all ranks through HDDP_PID_SALT; foreign PIDs are stored as pseudonyms

    # idle power of this GPU with the CUDA context resident (used for idle-subtracted energy)
    torch.cuda.synchronize()
    dist.barrier()
    time.sleep(1.0)
    e0, t0 = nvml.energy_j(h), time.perf_counter()
    time.sleep(3.0)
    idle_w = (nvml.energy_j(h) - e0) / (time.perf_counter() - t0)

    # initial spin calibration (cycles per ms) on the straggler; refined per condition in warmup
    cpm = None
    if rank == straggler:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            torch.cuda._sleep(5_000_000)
        s.record()
        torch.cuda._sleep(20_000_000)
        e.record()
        torch.cuda.synchronize()
        cpm = 20_000_000 / s.elapsed_time(e)

    hs = HookState() if a.instrument else None
    sampler = Sampler(h) if a.instrument else None
    for ci, (bucket, delay) in enumerate(conditions):
        if a.marker:
            mk = torch.zeros(101 + ci, dtype=torch.uint8, device=dev)
            dist.broadcast(mk, 0)
            torch.cuda.synchronize()
        ddp, opt = build(a.model, bucket, dev, hs)
        target = delay if rank == straggler else 0
        cycles = int(target * cpm) if target else 0

        # warmup (synchronized per step) with closed-loop delay calibration
        for i in range(a.warmup):
            if hs:
                hs.events = []
            ev = new_events()
            train_step(ddp, opt, x, y, cycles, ev)
            torch.cuda.synchronize()
            if target and i >= 1:
                cycles = max(1, int(cycles * target / ev[0].elapsed_time(ev[1])))

        # timed window (no per-step host sync)
        dist.barrier()
        torch.cuda.synchronize()
        evs, bevs = [], []
        if sampler:
            sampler.buf, sampler.active = [], True
        E0, T0 = nvml.energy_j(h), time.perf_counter()
        for i in range(a.steps):
            if hs:
                hs.events = []
            ev = new_events()
            train_step(ddp, opt, x, y, cycles, ev)
            evs.append(ev)
            if hs:
                bevs.append(hs.events)
        torch.cuda.synchronize()
        T1, E1 = time.perf_counter(), nvml.energy_j(h)
        samples, e_sampled = [], None
        if sampler:
            sampler.active = False
            samples = [smp for smp in sampler.buf if T0 <= smp[0] <= T1]
            if len(samples) > 2:  # trapezoid integral of instant power, scaled to the full window (cross-check)
                integ = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(samples, samples[1:]))
                e_sampled = integ / (samples[-1][0] - samples[0][0]) * (T1 - T0)
        throttle = nvml.throttle_reasons(h)

        r4 = lambda v: round(v, 4)
        st = {k: [] for k in ("delay", "fwd", "bwd", "opt", "gpu_step", "period",
                              "first_ready", "last_ready", "exposed_comm", "n_buckets")}
        for i, ev in enumerate(evs):
            st["delay"].append(r4(ev[0].elapsed_time(ev[1])))
            st["fwd"].append(r4(ev[1].elapsed_time(ev[2])))
            st["bwd"].append(r4(ev[2].elapsed_time(ev[3])))
            st["opt"].append(r4(ev[3].elapsed_time(ev[4])))
            st["gpu_step"].append(r4(ev[0].elapsed_time(ev[4])))
            if i + 1 < len(evs):
                st["period"].append(r4(ev[0].elapsed_time(evs[i + 1][0])))
            if bevs and bevs[i]:
                b = bevs[i]
                st["first_ready"].append(r4(ev[2].elapsed_time(b[0])))
                st["last_ready"].append(r4(ev[2].elapsed_time(b[-1])))
                st["exposed_comm"].append(r4(b[-1].elapsed_time(ev[3])))
                st["n_buckets"].append(len(b))
        # per-step NVML: mean of samples whose timestamp falls inside the GPU step interval
        # (step i starts at T0 + elapsed(ev0 of step 0 -> ev0 of step i); valid because the window starts idle)
        nvs = {k: [] for k in Sampler.FIELDS}
        starts = [T0 + evs[0][0].elapsed_time(ev[0]) / 1000 for ev in evs] + [T1]
        for i in range(len(evs)):
            inside = [smp for smp in samples if starts[i] <= smp[0] < starts[i + 1]]
            for j, k in enumerate(Sampler.FIELDS):
                nvs[k].append(round(sum(smp[j + 1] for smp in inside) / len(inside), 2) if inside else None)
        rec = {"rank": rank, "phys_gpu": phys, "straggler": rank == straggler, "wall_s": T1 - T0,
               "energy_j": E1 - E0, "energy_sampled_j": e_sampled, "idle_w": round(idle_w, 3), "cycles": cycles, "cpm": cpm, "steps": st,
               "nvml": nvs, "nvml_window_mean": {k: round(sum(smp[j + 1] for smp in samples) / len(samples), 2)
                                                 for j, k in enumerate(Sampler.FIELDS)} if samples else {},
               "n_samples": len(samples),
               "throttle": nvml.decode_throttle(throttle),
               "foreign_pids": [privacy.pid_pseudonym(p, salt) for p in nvml.compute_pids(h) if p not in job_pids]}

        if a.profile_steps and (not profile_set or (bucket, delay) in profile_set):
            from torch.profiler import ProfilerActivity, profile
            dist.barrier()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(a.profile_steps):
                    if hs:
                        hs.events = []
                    train_step(ddp, opt, x, y, cycles, new_events())
                torch.cuda.synchronize()
            kernels = [(k.start_ns(), k.end_ns(), k.name(), k.device_resource_id())
                       for k in prof.profiler.kineto_results.events()
                       if k.device_type() == torch.autograd.DeviceType.CUDA]
            rec["prof"] = overlap_metrics(kernels, a.profile_steps)
            if (bucket, delay) in timeline_set:
                t0k = min(k[0] for k in kernels)
                rec["timeline_t0_us"] = t0k // 1000  # absolute (host clock), aligns ranks on one machine
                rec["timeline"] = [[kernel_class(n), (s - t0k) // 1000, (e - t0k) // 1000, sid]
                                   for s, e, n, sid in kernels]

        gathered = [None] * world if rank == 0 else None
        dist.gather_object(rec, gathered, dst=0)
        if rank == 0:
            row = {**meta, "type": "cond", "model": a.model, "world": world, "batch": B, "bucket": bucket,
                   "delay_req": delay, "cond_index": ci, "warmup": a.warmup, "steps": a.steps,
                   "profile_steps": a.profile_steps, "instrument": a.instrument, "straggler_rank": straggler,
                   "ranks": gathered}
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[cond {ci + 1}/{len(conditions)}] bucket={bucket} delay={delay} "
                  f"step={1000 * max(r['wall_s'] for r in gathered) / a.steps:.1f}ms", flush=True)
        del ddp, opt
        gc.collect()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
