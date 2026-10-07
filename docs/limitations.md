# Limitations

## Scope of the study
- **One machine, one GPU model.** 8× RTX A6000 (GA102, 300 W default power limit) in a 2-socket AMD
  EPYC 7713 host, with NVLink bridges between GPU pairs (0-1, 2-3, 4-5, 6-7). Only GPUs 0-3 (NUMA node 0) were used
  because the machine is shared. There are no 8-GPU or cross-NUMA results; the topology parser finds such
  groupings, but they were not run. Conclusions may not transfer to NVSwitch systems, multi-node
  training, or other GPU generations.
- **Two workloads, one batch size each, bf16 autocast, synthetic data.** Real input pipelines add host-
  and I/O-side variance that this study removes on purpose.
- **Heterogeneity is emulated.** A fixed per-step GPU spin on one rank, inserted before the forward pass,
  is a simple model of a slow GPU. A genuinely slower GPU stretches *every* kernel, including the backward
  kernels whose timing interacts with bucket readiness. A spin before forward instead shifts the
  straggler's whole step. The position of the delay is a modelling choice that affects overlap results.
  Only one injection position was studied. All conclusions are conditional on this synthetic straggler
  model.
- **Hardware throttling (condition C) was not run.** The account has no root, and the non-destructive
  probe (`nvml.can_set_power_limit`) returned `NVML_ERROR_NO_PERMISSION`. The code path is implemented
  (`ThrottleGuard`: record, apply, verify, best-effort restore in `finally`/`atexit`/signal handlers) but
  untested on real hardware changes. Restoration cannot be guaranteed: no user-space program can restore a
  power limit after SIGKILL, a host crash or a power loss. This stage produced no measurements, so it does
  not affect any reported result.
- **Mechanisms are inferred, not isolated.** Several explanations in the README (compute/communication
  contention, BatchNorm buffer broadcast, lower power while waiting, relief of interference under a small
  delay) are consistent with the profiler and timing data, but no control experiment isolated them
  (for example a `broadcast_buffers=False` run, an overlap microbenchmark with fixed work and bytes, or a
  power-state decomposition). They should be read as measured associations with plausible mechanisms.

## Platform findings that shaped the setup
- **PCIe peer-to-peer between GPUs on different root complexes produced an incorrect result on this
  host.** With NCCL defaults, a 2-GPU allreduce over the `NODE` path (GPUs 0 and 2) stalled for about
  32-42 s (32 s in the recorded probe) and returned a **wrong result** (1.0 instead of 2.0). Bare-metal IOMMU
  settings and PCI Access Control Services (ACS) are plausible causes documented by NVIDIA, but the cause
  was not diagnosed: no kernel command line, `dmesg` IOMMU messages, ACS state or `nvidia-smi topo -p2p`
  output was collected. The recorded probe checked only the first result element and only the pair 0-2
  (methodology §1). All experiments therefore use `NCCL_P2P_LEVEL=NVL`: NVLink P2P stays enabled, and
  non-NVLink paths go through host shared memory (`SHM`). The "2 GPU, PCIe" setting is therefore really
  "PCIe via host memory", and the 4-GPU ring/tree mixes NVLink and SHM hops. A host where PCIe P2P works
  would likely show faster PCIe communication and could show different straggler behaviour.
- **Frequent NVML energy-counter queries under-reported energy on this driver (580.173.02).** In a short
  diagnostic, querying `nvmlDeviceGetTotalEnergyConsumption` from a background thread every 5 / 50 /
  100 ms made a steady ~297 W load read as 66 / 243 / 263 W. Polling power, temperature, clocks and
  utilization did not have this effect. NVIDIA does not document this behaviour and the cause is unknown;
  the observation does not show that the counter itself was corrupted. The counter is therefore read only
  at window edges, and the integrated instantaneous power is kept as a cross-check.
- **NVML sensors are coarse.** Measured on these GPUs:
  - The energy counter advances in steps about every 100-180 ms.
  - Instantaneous power and utilization refresh at a similar or slower rate.
  - Window energy has an edge quantization of roughly ±1 counter update per GPU, which is ~1-2% of a
    2-4 s window after summing 4 GPUs. The power-sample integral cross-check agrees with the counter to
    within about 2%.
  - Per-step power, utilization and temperature values are sample-and-hold and cannot resolve 70-160 ms
    steps.
- **The software power-limit policy frequently constrained clocks.** NVML reported the `sw_power_cap`
  clock-event reason on at least one GPU at the end of 783 of the 792 analysed windows. This means the
  driver was holding clocks down to stay under the 300 W software limit at those moments; it does not mean
  that each GPU drew 300 W continuously. Clock frequency, and therefore compute speed, depends on the power
  headroom. A straggler that spins draws less power, so it, or the fast ranks while they wait, can boost
  differently. This couples energy and timing and makes energy results specific to power-limited
  operation.

## Measurement caveats
- **Communication timing is indirect.** PyTorch 2.6 does not expose per-collective NCCL durations. The
  per-step "exposed communication / wait" metric is the time from a rank's last bucket being ready to
  backward completion. It cannot separate pure transfer time from waiting for peers. The profiler-based
  overlap metric (methodology §8) counts spinning NCCL kernels as communication, and treats
  "overlapped" as concurrent on the same GPU even though concurrent kernels compete for SMs.
- **The comm hook used for bucket-ready events replaces DDP's built-in C++ allreduce path** with
  `default_hooks.allreduce_hook` (same math). Its overhead was measured in the pilot rather than assumed
  to be zero.
- **Overlap is profiled in a subset of trials.** Profiling runs after the timed window, so it does not
  perturb timing. Where only one trial per condition was profiled, overlap values have no across-trial
  CI.
- **NCCL protocol is not controlled.** Forcing the algorithm leaves NCCL free to pick LL, LL128 or Simple,
  so Ring-vs-Tree comparisons also include NCCL's protocol choice. The selection is logged (see
  `nccl_verification.csv`).
- **Straggler placement.** A depends strongly on *which* rank is slow, because only a rank with slack can
  absorb a delay. Only rank 3 (main sweep) and rank 0 (control, 25 MB Auto only) were tested.
- **H2 confound under Auto.** For ViT-B/16 with NCCL Auto, the 128 MB bucket also changes NCCL's algorithm
  choice (some buckets switch Ring → Tree). The Auto bucket effect at 128 MB is therefore partly an
  algorithm effect. The forced-Tree sweep isolates the bucket effect.
- **Energy is noisy per window.** The counter vs. sampled-power cross-check agrees on average (+0.08%),
  but differs by up to 14% in single 2-4 s windows. Use the across-trial CIs (typically ±1-3% of J/step),
  not single windows.
- **Statistics.** Five trials per important condition give wide CIs for small effects.
  - H1 and H3 p-values are not corrected for multiple comparisons and are descriptive. H1 alone has 112
    cells, so about 5-6 nominal p < 0.05 results are expected by chance; one cell (A = 1.010) reached it.
  - H2 uses a repeated-measures ANOVA with Holm adjustment across 16 cells (methodology §10). With 4-5
    launches and 6 bucket sizes, sphericity cannot be tested; the Greenhouse-Geisser correction is used as
    the primary, conservative choice. The number of significant cells depends on the analysis (10, 13 or
    15 of 16 for A), so individual ResNet-50 cells should not be over-interpreted.
  - The H3 comparison picks the arg-max configuration and then tests it (winner's curse), so the reported
    regret is an optimistic upper bound on what a tuner would recover.
- **Shared machine, with measured cross-tenant interference.** Another user ran GPU jobs during the final
  sweep. Their processes appeared on our GPUs during 3 launches. Their jobs on *other* GPUs also
  degraded our 4-GPU runs by 10-67% through shared host resources (8 attempts of one launch), even though
  a bare host-memory allreduce benchmark stayed at its normal latency. All affected attempts were
  excluded whole and re-run.
  - The detectors are complementary: process lists, forward-time drift, and cross-trial consistency.
    Kept data lie within 2.93% of the cross-trial median, and degraded data deviated by 10-67%.
  - Interference smaller than the detection threshold, or one that shifts *all* trials of a condition
    equally, cannot be excluded.
  - Host CPU load is not controlled, but it is recorded per launch.
  - One trial of 4-GPU ResNet-50 Ring could not be collected cleanly: n = 4 for that block (final; see
    `final_summary.json["completeness"]`).
- **No Nsight Systems traces.** The node later crashed while this study's jobs were idle, and was under
  admin investigation afterwards. The Nsight traces were therefore not recorded. Kernel-level timelines
  come only from torch.profiler (fig10).
- **No independent replication.** The measurements come from one campaign on one host. Re-running the
  analysis from the raw data is fully reproducible; re-running the hardware experiment on another system
  has not been done.
- **Thermal state.** Launches run back-to-back without cool-down. GPU temperatures are logged and
  randomization spreads drift, but absolute energy/clock values depend on the thermal history.

## Related work and novelty
Straggler effects in synchronous data-parallel training, and straggler-aware or topology-aware
collectives, are well-studied. Examples include backup workers and asynchronous/stale-synchronous SGD,
straggler-tolerant and partial-reduce allreduce designs, and work on communication-computation overlap
and bucket sizing in PyTorch DDP. This repository is an **experimental characterization** on one machine.
It does not propose a new algorithm and does not claim novelty for the phenomena it measures.
