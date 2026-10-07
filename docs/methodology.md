# Methodology

Every number in this repository comes from code in `hddp/`. This file defines each metric exactly.

## 1. Machine characterization and topology (`hddp/env.py`)

`python -m hddp.env --gpus <allowed>` records `nvidia-smi`, `nvidia-smi topo -m`, `nvidia-smi nvlink -s`,
GPU/driver/PCIe/power/clock queries, `nvcc --version`, `lscpu`, `numactl -H`, `lspci -tv` and the
Python/PyTorch/CUDA/cuDNN/NCCL versions into `results/env/machine.txt`. It writes the parsed topology to
`results/env/topology.json`. Before writing, the output is redacted (host name, user paths, GPU UUIDs, clock
time, per-process rows of other tenants, CPU vulnerability status; see `hddp/privacy.py`).

The parser reads the GPU×GPU matrix of `nvidia-smi topo -m` and classifies every pair as one of:
- `nvlink` (`NV#`)
- `pcie_switch` (`PIX`/`PXB`)
- `pcie_host_bridge` (`PHB`)
- `pcie_same_numa` (`NODE`)
- `pcie_cross_numa` (`SYS`)

It also records each GPU's CPU and NUMA affinity. From the GPUs the study is *allowed* to use, it derives
the experiment settings: a 2-GPU NVLink pair, a 2-GPU non-NVLink pair (closest PCIe class available), a
cross-NUMA pair (only if one exists among the allowed GPUs), and a same-NUMA 4-GPU set if 4 are allowed.
NVLink is never assumed.

**P2P correctness check (current code).** The probe runs one 40 MB NCCL allreduce with
`NCCL_P2P_LEVEL=NVL` on
- **every non-NVLink pair** of allowed GPUs (for GPUs 0-3: 0-2, 0-3, 1-2, 1-3), and
- **every multi-GPU setting as a whole** (the 4-GPU group mixes NVLink and non-NVLink hops).

Each rank contributes the pattern `(i mod 1021) + rank + 1`. All values and sums stay below 2²⁴, so the
float32 result is exact, and **every rank compares its full result tensor** with the expected values and
also compares a float64 checksum. A position-dependent pattern also exposes stale, shifted or partially
copied buffers, which a tensor of ones would not. A probe passes only if every rank reports, no element
differs, and every checksum matches. The record (`topology.json["p2p_check"]`) holds the logical GPU
indices, the link class, the transport NCCL chose, the latency, the number of mismatched elements and a
plain-language failure reason; it contains no host identifiers. If the configuration used for the
experiments fails, the env stage exits with an error.
- With `--test-default-p2p` the probe also exercises NCCL's default PCIe P2P path on the same GPU sets.
  This is **opt-in** because exercising a faulty P2P path on a shared node is not harmless.
- Unless the default path is tested and found correct and fast (< 5 s), every launch uses
  `NCCL_P2P_LEVEL=NVL`, stored in `topology.json["nccl_env"]`.

**What was actually recorded for this study.** The `p2p_check` entries in the published `topology.json`
come from the earlier version of the probe, which all-reduced a tensor of ones, compared only the first
element on rank 0, and tested only the representative pair 0-2. With NCCL's default path it returned 1.0
instead of 2.0 after a 32 s stall: a wrong first element is conclusive evidence of a failure. With
`NCCL_P2P_LEVEL=NVL` (SHM transport) it returned 2.0 in 0.48 s, which is consistent with, but does not
prove, correct results over the full tensor and all pairs. The strengthened probe has not been re-run on
the study host. Bare-metal IOMMU settings and PCI ACS are plausible causes documented by NVIDIA; no
diagnostic evidence (kernel command line, `dmesg`, `lspci -vvv` ACS state, `nvidia-smi topo -p2p`) was
collected, so the cause is undiagnosed.

## 2. Workloads and training step (`hddp/worker.py`)

- `torchvision.models.resnet50()` (channels_last) and `torchvision.models.vit_b_16()`, randomly
  initialized with the same seed.
- PyTorch DDP with default settings except `bucket_cap_mb`: `broadcast_buffers=True`,
  `gradient_as_bucket_view=False`, no `find_unused_parameters`, no static graph.
- bf16 autocast for forward and loss; fp32 parameters, gradients and allreduce.
- Optimizers: SGD (momentum 0.9) for ResNet-50, AdamW for ViT-B/16. `zero_grad(set_to_none=True)`.
- Fixed per-GPU batch: 64 for ResNet-50, 32 for ViT-B/16.
- **Inputs are synthetic.** One fixed `randn` image tensor and one fixed label tensor are generated per
  rank on the GPU and reused for every step and condition. There is no dataloader or storage I/O.
- Each rank pins itself to the CPU cores of its GPU's NUMA node, read from sysfs `local_cpulist`.
  `OMP_NUM_THREADS=4`.
- `cudnn.benchmark=True`.

## 3. Conditions, warmup, randomization, trials (`hddp/run.py`)

**Launch structure.**
- One *launch* is one `torchrun` process group for a fixed (GPU set, model, NCCL algorithm, trial).
- Inside a launch, the (bucket, delay) conditions run in an order shuffled with a seed derived from the
  launch id.
- The launches of a stage run in an order shuffled with a fixed seed (`SEED` in `run.py`).

**Per condition.**
- The model, DDP wrapper and optimizer are rebuilt from scratch, so there is no carry-over of DDP bucket
  state or optimizer state between conditions.
- Then come `W` warmup steps (default 8), each followed by a device sync, and a timed window of `M` steps
  (default 30) with no host synchronization inside the window.

**Trials.**
- A *trial* is one launch, i.e. a separate set of processes and a new NCCL communicator.
- Confidence intervals are computed across trials, never across steps of the same window.
- Delay-0 baselines for paired metrics come from the *same launch*.

**GPU hygiene and contamination handling.** The machine is shared, and during the final sweep another user
started jobs on the study's GPUs, so this part was hardened mid-sweep.

Detection uses four independent signals:
1. **Pre-launch check.** A launch starts only when `nvidia-smi` shows no compute process on its GPUs.
   Otherwise it is deferred and other launches run first. If every launch is blocked, the orchestrator
   waits, up to 4 h.
2. **Continuous watch.** For the whole launch, the orchestrator polls the NVML process list of the
   launch's GPUs every 1 s. Any PID outside the launch's own processes is recorded in the launch's
   `foreign_watch` (as a pseudonym, with the seconds since the launch started). This watch was added after contamination was observed; earlier launches rely on
   signals 3 and 4.
3. **In-window check.** At the end of every timed window, each rank lists the processes on its GPU
   (`foreign_pids`).
4. **Timing check.** Forward compute of ranks that never wait during forward is fixed work and should
   be constant. Those ranks are every ViT rank, and rank 0 (the BN-broadcast root) for ResNet. If a
   rank's mean forward time exceeds 1.08× the model's median, the attempt is flagged.

Exclusion rules:
- Any attempt with a signal from 2, 3 or 4 is excluded **as a whole**, not just the flagged windows.
  Contamination can start before the first sampled sighting.
- An excluded launch is marked `contaminated` and re-queued (up to 8 retries). Each attempt carries a
  unique `attempt` id, and only attempts that finished `ok` and clean enter the analysis.
- `final_summary.json["completeness"]` lists every planned (launch, bucket, delay) cell that has no
  clean data. Missing cells are reported, never filled.

Pre/post-launch GPU temperature, power, utilization and the host load average are stored with every
launch record.

**Robustness.**
- Launches have a hard timeout and run in their own session. Every launch gets a random token in its
  environment, which all of its processes inherit. On timeout, error, SIGINT, SIGTERM or SIGHUP, and also
  after a normal exit, the orchestrator terminates torchrun's process tree **and** every process that
  carries the token (SIGTERM, then SIGKILL after 20 s). The token also finds workers that were re-parented
  after torchrun died. The foreign-process watch is started inside the same cleanup block, so a failure to
  start it cannot leave workers running. Nothing can clean up after a SIGKILL of the orchestrator itself;
  check `nvidia-smi` after such an event.
- `--dry` prints the plan without touching the GPUs, the hardware settings or any file, including for the
  throttle stage.
- Failed launches are recorded with their return code. Conditions completed before a failure stay in the
  raw data. Nothing is re-filled.

## 4. Controlled CUDA straggler

**Mechanism.**
- Exactly one rank (the last rank, so it never coincides with rank 0's gather/IO work) executes
  `torch.cuda._sleep(cycles)` on its compute stream at the start of every step, before the forward pass.
- This is a GPU-side spin kernel (`spin_kernel`, busy-waits on `clock64()`). The host is never blocked,
  so later kernels of that rank are simply queued behind it.
- `time.sleep` is not used anywhere in the training path.

**Calibration.**
1. At launch start, the straggler measures one 20 M-cycle spin with CUDA events to get an initial
   cycles-per-ms value.
2. In every condition's warmup (from warmup step 2 onward) it re-scales the cycle count by
   `requested / measured`. This closed loop absorbs SM-clock changes under load.
3. The cycle count is then frozen for the timed window.
4. Each step's actual delay is measured with CUDA events around the spin (`delay` column).

All straggler metrics use the **measured** mean delay δ of the timed window, not the requested value.
`<stage>_summary.json["delay_calibration"]` reports requested vs. measured.

## 5. Per-step timing

CUDA events on each rank's compute stream, per step:
`e0 | spin | e1 | forward+loss | e2 | backward (DDP allreduce) | e3 | optimizer.step + zero_grad | e4`.

| column | definition |
|---|---|
| `delay` | e0→e1 (≈0 on non-straggler ranks) |
| `fwd`, `bwd`, `opt` | e1→e2, e2→e3, e3→e4. `bwd` ends only after DDP has waited for all bucket allreduces, so it includes any wait for peers |
| `gpu_step` | e0→e4 |
| `period` | e0(step i) → e0(step i+1). Steady-state step time including inter-step gaps |
| `first_ready`, `last_ready` | e2 → CUDA event recorded when DDP's comm hook fires for the first/last bucket on this rank (bucket's gradients enqueued) |
| `exposed_comm` | last bucket ready → e3. The time the compute stream sits after this rank's last gradient bucket is ready until all allreduces complete. On a fast rank this is residual communication **plus waiting for the straggler**; on the straggler it is the communication tail. Used as the per-step "communication wait" metric |

**Step time (primary)** is the mean `period` over the window, averaged across ranks. Ranks agree to
within `period_rank_spread_ms`, which is reported. Throughput = world × batch / step time. The wall-clock
window time divided by M is kept as `step_ms_wall`.

The bucket-ready events require a Python DDP comm hook that wraps the stock
`default_hooks.allreduce_hook`. The pilot measures the hook's overhead against DDP's built-in C++
allreduce (`--instrument 0` launches, tag `noinstr`); the result is in
`pilot_summary.json["instrumentation_overhead"]`.

## 6. NCCL algorithm control and verification

- **Auto** = `NCCL_ALGO` unset. **Ring** / **Tree** = `NCCL_ALGO=Ring|Tree`. NCCL 2.21 still uses Ring for
  non-allreduce collectives (broadcasts); only allreduce is forced. The protocol (LL/LL128/Simple) is not
  forced.
- **Verification is never assumed.** The `verify` stage repeats each (setting, model, algorithm) with all
  six bucket sizes under `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING,COLL`.
  - Each `COLL` line (op, count, dtype) is paired with the following `TUNING` line
    (`<bytes> Bytes -> Algo <a> proto <p>`) from the same process.
  - Conditions are delimited by a uint8 broadcast marker of size 101+i.
  - The result is the algorithm/protocol NCCL actually selected for every gradient-bucket allreduce
    (`results/processed/nccl_verification.csv`). The raw logs are kept in `results/raw/nccl_logs.txt.gz`.
- Independently, every profiled production window records the NCCL kernel names that ran, e.g.
  `ncclDevKernel_AllReduce_Sum_f32_RING_LL`. These are summarized in `nccl_kernels_in_profiled_runs`. The
  algorithm part of the name agrees with the TUNING logs. The protocol suffix does **not**: Ring/Simple work
  ran in a kernel named `RING_LL`, because NCCL 2.21 launches a specialized kernel that also executes other
  protocols. Only the logs are authoritative.

## 7. GPU telemetry and energy (NVML via ctypes, `hddp/nvml.py`)

**Per-step series.**
- A background thread on each rank polls its GPU every 20 ms: instantaneous power
  (`NVML_FI_DEV_POWER_INSTANT`), temperature, SM clock and utilization.
- Samples are wall-clock stamped and assigned to GPU steps by mapping CUDA-event offsets onto the wall
  clock. The window starts from an idle, synchronized GPU, so step *i* starts at
  `T0 + elapsed(e0[0], e0[i])`.
- The NVML sensors on this GPU refresh only about every 100 ms or slower, and utilization is a
  sampling-period percentage. Per-step values are therefore sample-and-hold readings, not step-resolved
  measurements. Window means are also stored.

**Energy.**
- The cumulative NVML energy counter (`nvmlDeviceGetTotalEnergyConsumption`) is read by the main thread
  only at the timed window edges (after a barrier and device sync, and after the final device sync).
- In a diagnostic on this driver, frequent queries of that counter coincided with severe under-reporting
  of cumulative energy (cause unknown; see limitations), so the sampler never touches it.
- As a cross-check, the trapezoid integral of the 20 ms instantaneous-power samples over the window is
  stored as `energy_sampled_j`. The trial table reports the relative disagreement
  (`energy_sampled_vs_counter_pct`).

**Derived energy metrics.**
- `energy_J_step` = Σ_GPUs ΔE / M.
- `energy_idle_sub_J_step` = (Σ ΔE − Σ P_idle × window) / M. `P_idle` is measured per GPU at launch start
  over 3 s with the CUDA context resident and no kernels running (after a 1 s settle).
- `images_per_J` = world × batch × M / Σ ΔE.

## 8. Profiler and the overlap metric

When `--profile-steps P` is set (P = 4), each condition runs P extra steps after the timed window under
`torch.profiler` with CUDA activity only. These steps are excluded from all timing and energy metrics.

The kernel intervals (CUPTI start/end, host clock) are split into three classes:
- **NCCL kernels**: name contains `nccl`. This includes the per-step BN-buffer broadcasts of ResNet-50.
- **delay kernels**: `spin_kernel`.
- **compute kernels**: everything else, including memcpy/memset.

With `U_N` and `U_C` the unions of NCCL and compute intervals on one rank:

- `comm_ms` = |U_N| / P
- `overlap_ms` = |U_N ∩ U_C| / P
- **`overlap_ratio` = |U_N ∩ U_C| / |U_N|**: the fraction of NCCL-kernel time during which at least one
  compute kernel is also running on that GPU
- `exposed_comm_ms` = |U_N \ U_C| / P

Ratios are averaged over the fast (non-straggler) ranks and reported separately for the straggler.

How to read it:
- A high overlap ratio does **not** mean communication is free. Overlapped compute kernels were
  measured to run slower: in the 4-GPU ViT setting, the same 1369 kernels per step took about 135 ms
  on the ranks with the heaviest concurrent NCCL traffic, against about 104-118 ms elsewhere. This is an
  association in profiler windows; contention is the leading explanation, but no microbenchmark held the
  compute work and the communicated bytes fixed while toggling overlap.
- Under delay, the fast ranks' NCCL kernels mostly spin while waiting for the straggler. "Overlap
  degradation" therefore measures waiting, not a loss of transfer hiding.

Caveats:
- NCCL kernels spin while waiting for peers, so `comm_ms` on a fast rank includes synchronization wait.
- Concurrent kernels share SMs, so "overlapped" time is not free.
- The profiler adds overhead, which is why it never runs in the timed window.

One window per condition (4 ranks × 4 steps) is profiled per profiled trial. Where only one trial was
profiled there is no across-trial CI for overlap. The per-step `exposed_comm` metric from §5, available
in every trial, is the CI-backed counterpart.

## 9. Straggler metrics (paired within a launch, `hddp/analyze.py`)

For a condition with delay δ > 0 and its δ = 0 baseline in the same launch, with the same bucket and
algorithm:

- **A(δ)** = (step(δ) − step(0)) / δ_measured. A = 1: the slowdown passes through additively. A > 1:
  amplification. 0 < A < 1: partially absorbed.
- throughput degradation % = 100 · (thr(0) − thr(δ)) / thr(0)
- energy increase % = 100 · (E/step(δ) − E/step(0)) / E/step(0), also idle-subtracted
- overlap degradation % = 100 · (ovl(0) − ovl(δ)) / ovl(0), fast ranks, profiled trials only
- **communication wait increase** (fast ranks, ms/step) = forward-side wait increase + backward-side wait
  increase:
  - Forward-side = fwd_fast(δ) − fwd_fast(0). DDP's default `broadcast_buffers=True` broadcasts
    BatchNorm buffers from rank 0 before every forward pass, so for ResNet-50 ranks downstream of the
    straggler can wait *inside forward*. ViT-B/16 has no buffers. Forward work of non-delayed ranks is
    unchanged, so an increase is attributed to waiting; a change in compute speed from altered interference
    cannot be fully excluded, and no `broadcast_buffers=False` control was run.
  - Backward-side = exposed_comm_fast(δ) − exposed_comm_fast(0).

  This wait is **not** the same as lost time. A fast rank's own backward compute can get faster under
  delay (less interference, §8), so its wait can grow by more than δ.

**Exploratory slack model (formulated after the pilot, i.e. post hoc).**
- At δ = 0, let `slack` be the straggler-to-be's `exposed_comm` minus the smallest per-rank
  `exposed_comm` in the same window. This is the time that rank already idles, beyond the critical rank's
  communication tail.
- The model predicts `A_pred = max(0, δ − slack) / δ`: a delay smaller than the slack is hidden.
- It is evaluated (RMSE, Pearson r) on the final data in `final_summary.json["H1_slack_model"]`. It is an
  explanation, not a tested hypothesis.

## 10. Statistics

- Condition summaries: mean and 95% Student-t CI across trials (n = number of launches).
- **H1**: one-sided one-sample t-tests of trial-level A(δ) against 1 (A > 1 and A < 1), per setting,
  model, algorithm, bucket and δ. These 112 p-values are not corrected for multiple comparisons and are
  descriptive. The H1 conclusion is a negative one (no amplification), which a correction would only
  strengthen.
- **H2** (does bucket size change the paired straggler metrics?). One test per cell = (setting, model,
  algorithm, δ) with at least 3 bucket sizes, i.e. 16 cells on 4 GPUs, for each of three metrics: A(δ),
  throughput degradation and energy increase.
  - *Design.* Every bucket condition of a launch is measured inside that launch and paired with that
    launch's own δ = 0 baseline of the same bucket size. The launch is therefore the **subject**, and
    bucket size is a **within-subject** factor with 6 levels. Treating the 6 bucket groups as independent
    samples (one-way ANOVA) ignores this pairing.
  - *Primary test.* One-way **repeated-measures ANOVA** on the complete launches (those with a value for
    every bucket size; launches with missing buckets are excluded from the test and counted in
    `n_launches_incomplete_excluded`). With n = 4-5 launches and 6 levels the covariance matrix is rank
    deficient and sphericity cannot be tested, so the **Greenhouse-Geisser**-corrected p-value is primary;
    the sphericity-assumed p-value is also reported.
  - *Multiple comparisons.* p-values are **Holm-adjusted** (step-down, family-wise error rate 0.05)
    across the 16 cells of the same metric, separately for each test type.
  - *Effect sizes.* Generalized η² (comparable across designs), partial η², the range of the bucket means
    (in the metric's own unit), and a paired contrast between the buckets with the highest and lowest mean
    (mean paired difference, 95% CI, Cohen's d<sub>z</sub>). The contrast buckets are chosen by their means,
    so the contrast is descriptive, not a test.
  - *Sensitivity analysis.* **Friedman test** (rank-based, tie-corrected) with Kendall's W, Holm-adjusted
    in the same way.
  - *Secondary, for comparison only.* The earlier one-way ANOVA that ignores the pairing, unadjusted and
    Holm-adjusted.
  - The implementation (`hddp/analyze.py`: `rm_anova`, `holm`, `h2_cell`) was cross-checked against
    pingouin 0.7.0 and statsmodels 0.15.0 on the study data: F, p, Greenhouse-Geisser ε and corrected p,
    generalized η², Friedman p, Kendall's W and the Holm adjustment agree to floating-point precision.
  - Results: `<stage>_summary.json["H2"]` (per cell), `["H2_overview"]` (counts of significant cells per
    analysis) and `results/processed/<stage>_h2.csv` (flat table).
- **H3**: per setting and model, the (algorithm, bucket) with the highest mean throughput at δ = 0 (the
  "homogeneous-optimal" configuration) is compared with the best configuration at each δ:
  - regret % = 100 · (best − homo-opt) / best
  - rank of the homo-opt configuration at δ
  - Welch t-test between the two configurations' trials

  The same analysis is repeated for images/J. These p-values are uncorrected and descriptive. Choosing the
  arg-max and then testing it is optimistic (winner's curse), so a non-significant regret means "no
  detectable loss in this matrix", not "re-tuning never matters".

**Straggler placement control.** The 4-GPU, Auto, 25 MB delay sweep is repeated with the straggler on rank 0
instead of rank 3 (tag `strag0`), because absorption depends on the delayed rank's slack.

## 11. Data handling and privacy

The raw records are the source of truth; nothing in `results/processed` is edited by hand. The published
raw data were redacted with `scripts/sanitize_data.py`, which replaces identifiers (host name, user and
installation paths, network addresses, GPU UUIDs, launch-attempt ids, other tenants' process ids) with
placeholders and removes wall-clock timestamps without touching any measurement. The redacted data
reproduce the processed tables (`scripts/check_reproducibility.py`). Field-level details are in
[`data_dictionary.md`](data_dictionary.md).
