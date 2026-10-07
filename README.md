# hetero-ddp-lab

An experimental characterization of how a **controlled GPU straggler** propagates through synchronous
PyTorch DistributedDataParallel (DDP) training on one 8× RTX A6000 server. The study measures NCCL
collectives, communication/computation overlap, throughput and energy, and asks whether two cross-layer
knobs, the DDP gradient **bucket size** and the **NCCL allreduce algorithm**, change the sensitivity to a
slow rank.

This is an exploratory measurement study of **one host, one topology, two workloads and one synthetic
straggler model**. It does not propose a new distributed-training algorithm (see
[Related work](#related-work)), and its results are not general laws.

## Motivation
Synchronous data parallelism advances at the pace of the slowest rank: every step ends with gradient
allreduces that cannot complete until all ranks contribute. In practice GPUs in one job are rarely
identical. Causes include thermal or power throttling, a noisy neighbour, a different SKU, or a slower
PCIe path. DDP hides communication behind the backward pass by bucketing gradients, and NCCL picks a
collective algorithm per message. Both decisions are normally tuned on healthy, homogeneous hardware. It
is not obvious whether those settings stay good when one rank is slow, or how a local slowdown turns into
lost throughput and wasted energy on the ranks that wait.

## Research questions
- **RQ1.** How does a controlled slowdown of one GPU propagate through synchronous DDP: step time,
  NCCL/communication wait, communication-computation overlap, throughput and energy?
- **RQ2.** Can cross-layer configuration (DDP `bucket_cap_mb`, NCCL algorithm Auto/Ring/Tree) reduce the
  sensitivity to a heterogeneous or straggling GPU?

## Hypotheses (tested, not assumed)
- **H1.** Synchronous DDP may *amplify* local GPU slowdowns through synchronization: A(δ) > 1.
- **H2.** Gradient bucketization may change sensitivity to heterogeneity because it changes
  communication-computation overlap.
- **H3.** The NCCL/DDP configuration that maximizes throughput on homogeneous GPUs may stop being optimal
  once one rank is slower.

## Setup
| | |
|---|---|
| GPUs | 8× NVIDIA RTX A6000 48 GB (300 W default power limit), driver 580.173.02. **Only GPUs 0-3 used for experiments** (shared machine); GPU 4 once for a 20 s NVML diagnostic |
| Interconnect | NVLink bridge pairs (NV4) 0-1, 2-3, 4-5, 6-7; other pairs PCIe within a NUMA node (`NODE`) or across sockets (`SYS`) |
| CPU / memory | 2× AMD EPYC 7713 (2×64 cores, SMT2), 2 NUMA nodes, 1 TB RAM; GPUs 0-3 on NUMA 0 |
| Software | Python 3.13.12, PyTorch 2.6.0+cu124, torchvision 0.21.0, CUDA runtime 12.4, cuDNN 9.1, NCCL 2.21.5 |
| Raw characterization | [`results/env/machine.txt`](results/env/machine.txt), parsed topology [`results/env/topology.json`](results/env/topology.json) (both redacted, see [Privacy](#privacy-and-redaction)) |

Settings derived by the topology parser from GPUs 0-3:

| setting | GPUs | links | NCCL transport (verified) |
|---|---|---|---|
| `2gpu_nvlink` | 0,1 | NVLink | P2P over NVLink |
| `2gpu_pcie` | 0,2 | PCIe, same NUMA, different root complex | **SHM via host memory** (see below) |
| `4gpu` | 0,1,2,3 | 2 NVLink pairs joined over PCIe | P2P (NVLink) + SHM |

> **Platform finding: PCIe P2P produced an incorrect result on this host.** With NCCL defaults, an
> allreduce between GPUs 0 and 2 ran over P2P, took about 32 s, and returned a **wrong value** (1.0
> instead of 2.0). The evidence is in [`topology.json`](results/env/topology.json) → `p2p_check`. Bare-metal
> IOMMU settings and PCI Access Control Services (ACS) are plausible causes documented by NVIDIA, but the
> cause was **not diagnosed** (no kernel command line, ACS state or `dmesg` evidence was collected). Every
> experiment therefore runs with `NCCL_P2P_LEVEL=NVL`: NVLink P2P stays on, and PCIe paths go through host
> shared memory.
>
> The recorded probe compared only the first element of the result and tested one representative pair
> (0-2); a wrong first element is conclusive evidence of a failure, but a correct one would not have proven
> correctness. The probe in [`hddp/env.py`](hddp/env.py) has since been strengthened (full-tensor check on
> every rank, every non-NVLink pair plus the 4-GPU group); it has not been re-run on the study host.
> Re-running the env stage never exercises NCCL's default PCIe P2P path unless `--test-default-p2p` is
> passed.

Workloads: torchvision **ResNet-50** (batch 64/GPU, channels_last) and **ViT-B/16** (batch 32/GPU), bf16
autocast, fixed synthetic inputs, SGD-momentum / AdamW, DDP defaults except `bucket_cap_mb`.

## Methodology (summary)
Full definitions are in [`docs/methodology.md`](docs/methodology.md).

- **Straggler.** Exactly one rank (rank 3, or rank 0 in a control) runs a GPU-side spin kernel
  (`torch.cuda._sleep`) before every forward pass. It is closed-loop calibrated during warmup, and the
  actual delay is measured with CUDA events every step. All derived metrics use the **measured** δ.
  The mean measured delay is within 0.8% of the requested 5/10/20/40 ms. This is a synthetic model of a
  slow GPU: it shifts the whole step of one rank instead of stretching every kernel.
- **Timing.** CUDA events give forward/backward/optimizer times and the steady-state step period. A DDP
  comm hook adds bucket-ready events, giving the per-step **exposed communication/wait**: last bucket
  ready → backward done.
- **Profiler overlap.** torch.profiler kernel intervals give the overlap ratio
  |NCCL ∩ compute| / |NCCL| per rank.
- **NVML energy.** Cumulative energy counter read at window edges only, cross-checked against integrated
  instantaneous power. Gives J/step, idle-subtracted J/step and images/J. Power, temperature, clocks and
  utilization are sampled at 20 ms.
- **NCCL verification.** `NCCL_DEBUG=INFO` COLL+TUNING logs give the algorithm/protocol NCCL actually
  selected for every bucket allreduce. Profiled NCCL kernel names corroborate the algorithm. The protocol
  suffix in kernel names is not reliable: Ring/Simple work ran in a kernel named `RING_LL`.
- **Rigor.**
  - Warmup before every condition, with a fresh model/DDP/optimizer per condition.
  - 5 independent launches (trials) per condition; randomized launch and condition order.
  - 95% t-CIs across trials; paired δ=0 baselines within each launch.
  - GPU-idle check before every launch, plus a continuous (1 s) foreign-process watch.
  - Forward-time and cross-trial consistency checks.
  - Any contaminated launch is excluded whole and re-run.
  - Failed launches recorded, never imputed.
  - Bucket-size effects (H2) are tested with a repeated-measures ANOVA (launch = subject) with
    Greenhouse-Geisser correction and Holm adjustment across cells.

**Process.** smoke test → pilot (instrumentation checks, overhead, variance) → NCCL verification →
final matrix. The optional hardware-throttling stage (power cap on one GPU, with best-effort restoration of
the original limit on normal exit, exceptions and SIGINT/SIGTERM/SIGHUP) is implemented but **could not
run**: no permission (`NVML_ERROR_NO_PERMISSION`, probe recorded in `results/raw/throttle.jsonl`). It
therefore produced no measurements.

### Final matrix (chosen from the pilot)
The pilot showed:
- Trial-to-trial step-time variation of about 0.2%.
- Instrumentation overhead of 0.3-1.2%.
- Correct delay calibration.
- Clear Auto/Ring/Tree differences on 4 GPUs.

The final matrix (5 trials planned everywhere, 60 launches, 810 conditions; one block ended with 4,
see Data integrity):

| block | setting | model | NCCL | buckets (MB) | delays (ms) | purpose |
|---|---|---|---|---|---|---|
| A | 4gpu | both | Auto | 1, 4, 16, 25, 64, 128 | 0, 5, 10, 20, 40 | H1, H2 |
| B | 4gpu | both | Ring, Tree | 1, 4, 16, 25, 64, 128 | 0, 10, 40 | H2, H3 |
| C | 2gpu_nvlink, 2gpu_pcie | both | Auto | 25 | 0, 5, 10, 20, 40 | H1 across topologies |
| D | 4gpu, straggler on rank 0 | both | Auto | 25 | 0, 5, 10, 20, 40 | placement control |

Profiler windows: every condition in trial 0, and the 25 MB delay sweeps in every trial.

## Results
All numbers below are measured and taken from
[`results/processed/final_conditions.csv`](results/processed/final_conditions.csv),
[`final_h2.csv`](results/processed/final_h2.csv) and
[`final_summary.json`](results/processed/final_summary.json). They are mean ± 95% CI over **n = 5
independent launches** unless stated otherwise. A is the slowdown amplification factor:
A(δ) = (step(δ) − step(0)) / δ, where δ is the injected delay. Unless a mechanism is explicitly said to
be isolated by a control, explanations below are interpretations of measured associations.

### Data integrity (read this first)
- **Completeness.** 792 of 810 planned conditions have clean data. The missing 18 are one of the five
  trials of *4-GPU ResNet-50 Ring* (`final|4gpu|resnet50|Ring|t1`): every attempt was degraded by another
  tenant's jobs and excluded. That block therefore has **n = 4 (final)**. Across its 18 conditions, the
  four clean trials vary by 0.29% (median CV; max 0.52%), and fastest-to-slowest differs by at most 1.0%.
  Nothing is imputed.
- **Contamination from other users was detected and removed.** The machine is shared. During the sweep,
  another user's jobs:
  - appeared on our GPUs mid-launch (3 launches);
  - degraded our 4-GPU runs from *other* GPUs through shared host resources (8 attempts, 10-67% slower).

  All affected attempts were excluded **whole** and re-run (methodology §3). In the kept data, every
  condition lies within **2.93%** of the cross-trial median of the same condition, and no forward-time
  anomalies remain.
- **Node crash (not attributed to this study).** The server later crashed while this study's jobs were
  idle; at the time the orchestrator was only polling `nvidia-smi` once a minute. The cause is unknown from
  a non-root account. All raw data survived (0 corrupt records), and the analysis was re-run from it with
  identical results. The planned **Nsight Systems traces were not recorded**, to avoid loading the node
  while it was under admin investigation. `scripts/nsys.sh` is ready and refuses to run on a busy machine.
- **Validity checks:**
  - **Delay calibration:** the measured δ averages within 0.8% of the requested value in every cell
    (within 0.25% on 4 GPUs). Individual 30-step windows fall between −2.0% and +1.7%.
  - **Auto vs Ring:** in the 18 cells where NCCL verifiably selected Ring for Auto too, the two
    independently launched runs differ by at most **0.28%** (0/18 significant).
  - **NCCL forcing:** all **72 of 72** forced Ring/Tree conditions were confirmed in NCCL's logs.
  - **Instrumentation overhead:** 0.3-1.2% step time (pilot).
  - **Robustness:** A computed from mean vs median step time differs by 0.011 on average.

### Homogeneous baselines (δ = 0, NCCL Auto, 25 MB buckets)
| setting | ResNet-50 step / throughput | ViT-B/16 step / throughput |
|---|---|---|
| 2 GPU, NVLink | 64.3 ± 0.1 ms / 1992 img/s | 87.9 ± 0.2 ms / 728 img/s |
| 2 GPU, PCIe via host memory (SHM) | 79.1 ± 0.4 ms / 1618 img/s | 131.7 ± 1.5 ms / 486 img/s |
| 4 GPU, NVLink + SHM | 80.6 ± 0.3 ms / 3175 img/s | 138.9 ± 0.9 ms / 922 img/s |

**Even the "homogeneous" baseline behaves heterogeneously when communication overlaps compute.** In the
4-GPU ViT runs, the GPU compute kernels per step (profiler, compute-busy time) take **126.9 ms** on the fast
ranks but 116.1 ms on rank 3; the ranks carrying more concurrent NCCL/SHM traffic compute more slowly. When
the ranks desynchronize (δ = 40), the fast ranks' compute drops to **105.3 ms** (−17%), and the delayed
rank's rises to 123.6 ms. Compute kernels were thus 17-20% longer when heavily overlapped in these
profiler windows; contention for SMs and memory bandwidth is the leading explanation, but it was not
isolated by a controlled microbenchmark.

### H1: does synchronous DDP amplify a local slowdown? Not supported on this host
A(δ) at NCCL Auto and 25 MB buckets (straggler = last rank unless noted):

| setting | ResNet-50 δ=10 | ResNet-50 δ=40 | ViT-B/16 δ=10 | ViT-B/16 δ=40 |
|---|---|---|---|---|
| 2 GPU, NVLink | 0.940 ± 0.007 | 0.977 ± 0.002 | 0.955 ± 0.006 | 0.976 ± 0.004 |
| 2 GPU, PCIe (SHM) | 0.967 ± 0.044 | 1.010 ± 0.012 | 0.992 ± 0.316 | 0.981 ± 0.035 |
| 4 GPU, straggler = rank 3 | 0.952 ± 0.036 | 0.989 ± 0.011 | **0.004 ± 0.069** | **0.748 ± 0.016** |
| 4 GPU, straggler = rank 0 (control) | 1.003 ± 0.122 | 0.997 ± 0.007 | 0.989 ± 0.047 | 1.005 ± 0.023 |

- **No meaningful amplification.** Across all 112 tested cells, A ranges from −0.25 to 1.03. Only one cell
  is nominally above 1 (2-GPU SHM ResNet, δ = 40: A = 1.010 ± 0.012). With 112 uncorrected tests,
  about 5-6 such cells are expected by chance; a multiple-comparison correction would only make an
  amplification claim harder.
- **The delay mostly passes through one-for-one** (A ≈ 0.94-1.0). The exception is a rank with
  **slack**: a rank that already idles while waiting on communication can hide a delay.
  - 4-GPU ViT, rank 3 delayed: δ = 10 ms is completely absorbed (A ≈ 0), and only 75% of a 40 ms delay
    reaches step time.
  - Delaying rank 0 instead, which has no measured slack, gives A ≈ 1.0. In this setup, **where** the slow
    GPU sits mattered as much as how slow it was.
  - A post-hoc slack model, A ≈ max(0, δ − slack)/δ, tracks the measured A with Pearson r = 0.89 (ViT)
    and 0.59 (ResNet) ([fig11](results/figures/fig11_slack_model.png)). It under-predicts A, plausibly
    because it ignores the change in compute/communication interference described above.
- **A small delay coincided with a small speedup.** 4-GPU ViT with rank 3 delayed by 5 ms:
  A = −0.25 ± 0.19 (n = 5), i.e. about 1.2 ms (0.9%) faster per step. This is consistent with reduced
  compute/communication interference on the critical ranks; the mechanism was not isolated.
- **ResNet-50 waits inside the forward pass.** At δ = 40 the fast ranks' forward-side wait rises by
  26.3 ms and the backward-side by 17.1 ms. ViT, which has no buffers, waits only in backward. This is
  consistent with DDP's default `broadcast_buffers=True`, which synchronizes BatchNorm buffers at the start
  of every forward pass; a `broadcast_buffers=False` control that would attribute the whole effect to the
  broadcast was not run.

### Throughput, energy, overlap (Auto, 25 MB)
- **Throughput loss at δ = 40** ranges from 17.7% (4-GPU ViT, rank 3) to 37.8% (2-GPU NVLink ResNet)
  ([fig02](results/figures/fig02_throughput_vs_delay.png)).
- **Energy per step grew less than step time**
  ([fig04a](results/figures/fig04a_energy_vs_delay.png), [fig04b](results/figures/fig04b_energy_idle_sub_vs_delay.png)):
  - 2-GPU NVLink ResNet, δ = 40: step time +61%, J/step **+30.5 ± 2.9%**, idle-subtracted
    **+18.3 ± 3.8%**.
  - 4-GPU ViT, δ = 40: step time +22%, J/step +6.5 ± 2.5%, idle-subtracted −0.1 ± 3.2% (no detectable
    change).

  This is consistent with GPUs drawing less power while they wait, but no power-state decomposition was
  done, so it is an interpretation rather than a measured mechanism.
- **Overlap ratio on fast ranks** falls from 0.84-0.98 at δ = 0 to 0.37-0.55 at δ = 40
  ([fig05](results/figures/fig05_overlap_vs_delay.png)). The metric counts NCCL kernel residency,
  including spinning, so this mostly reflects NCCL kernels *waiting* for the straggler rather than a loss of
  hidden transfer time (methodology §8).

### H2: does bucket size change sensitivity? Yes for ViT-B/16; small and mostly not significant for ResNet-50
Every bucket condition is measured inside the same launch and paired with that launch's own δ = 0
baseline, so the six bucket sizes are **repeated measurements of one launch**. The primary test is
therefore a **repeated-measures ANOVA** of trial-level A across the 6 bucket sizes (launch = subject,
n = 5 launches, n = 4 for ResNet-50 Ring) with the **Greenhouse-Geisser** correction, and p-values are
**Holm-adjusted** across the 16 tested (model, algorithm, δ) cells on 4 GPUs. Effect sizes are generalized
η² (η²<sub>G</sub>) and the range of the bucket means. Full results: [`final_h2.csv`](results/processed/final_h2.csv).

| analysis of A (16 cells) | cells with p < 0.05 |
|---|---|
| **Repeated measures, Greenhouse-Geisser, Holm (primary)** | **10** |
| Repeated measures, sphericity assumed, Holm | 15 |
| Friedman test (non-parametric), Holm | 13 |
| One-way ANOVA ignoring the pairing (earlier analysis; secondary), unadjusted / Holm | 15 / 15 |

- **ViT-B/16: significant in all 8 cells** (Holm-adjusted p ≤ 0.015, η²<sub>G</sub> = 0.77-0.96). The
  bucket means of A differ by 0.06-0.56. Examples:
  - **Forced Tree, δ = 10:** A = 0.36-0.58 across bucket sizes (adjusted p = 4×10⁻⁵); throughput loss
    3.1-4.9%.
  - **Auto, δ = 10:** A ≈ 0 for 1-64 MB but 0.50 at 128 MB (adjusted p = 5×10⁻⁶). Confounded: at 128 MB NCCL
    Auto switches some buckets to Tree (`nccl_verification.csv`).
- **ResNet-50: significant in only 2 of 8 cells** (forced Tree, δ = 10 and 40; adjusted p = 0.015 and
  2×10⁻⁴). In the other 6 cells (all Auto and Ring cells) the adjusted p is 0.07-0.24. Across all ResNet
  cells A varies by only 0.02-0.09 between bucket sizes, and throughput loss by ≤ 2.9 percentage points.
- **Other metrics (primary analysis):** the paired throughput degradation depends on bucket size in 15 of
  16 cells (exception: ResNet-50 Auto δ = 5), the paired energy increase in 2 of 16 (ViT Auto and Ring at
  δ = 40).
- With 4-5 launches and 6 bucket sizes, sphericity cannot be tested, which is why the more conservative
  Greenhouse-Geisser p-values are primary. The earlier "15 of 16" headline came from an ANOVA that treated
  the bucket conditions as independent groups and is kept only as a secondary comparison.
- **Sensitivity is not the same as speed.** A configuration can absorb a delay better while still being
  slower overall ([fig06](results/figures/fig06_bucket_vs_throughput.png),
  [fig07b](results/figures/fig07b_bucket_vs_amplification.png)).

### H3: does the homogeneous optimum stop being optimal? Not supported in the tested matrix
Among 18 configurations on 4 GPUs (3 NCCL algorithms × 6 bucket sizes)
([fig08](results/figures/fig08_algo_comparison.png), [fig09](results/figures/fig09_homo_vs_het_optimal.png)):
- **Throughput:** the δ = 0 optimum (**Tree/1 MB** for ResNet-50, **Tree/16 MB** for ViT-B/16) is still
  the best, or statistically tied, at δ = 10 and 40. The largest regret is 0.08% (ViT, δ = 10; Tree/25 MB
  nominally best, Welch p = 0.70).
- **images/J:** the δ = 0 optimum is Tree/128 MB for both models. Its largest regret is 0.77% (ViT, δ = 40,
  rank 5/18, p = 0.36), which is not significant.
- So no statistically detectable loss from the homogeneous optimum was found in this host's tested matrix.
  Selecting the best configuration and testing it on the same small sample is optimistic (winner's curse),
  and this local negative result does not show that re-tuning never matters elsewhere.
- **Tree beat NCCL's own Auto choice at every delay** on this NVLink+SHM topology:
  - ResNet: 3557 vs 3378 img/s at δ = 0 (+5%).
  - ViT: 1165 vs 1057 img/s (+10%), and +19% vs Ring.

  Auto almost always picked Ring here. The configuration lesson for this host is "verify NCCL's choice",
  not "re-tune for stragglers".

### Bottom line
On this host, with this topology (NVLink pairs joined through host memory), these two workloads and a
synthetic pre-forward delay on one rank:
- A single slow rank **passed its delay into synchronous DDP step time almost one-for-one, without
  amplifying it**. Part of the delay was absorbed only when the delayed rank had measured slack.
- Bucket size changed *how much* of a delay was absorbed for ViT-B/16, much less so for ResNet-50, and
  neither bucket size nor NCCL algorithm **changed which configuration was best**.
- Overlapped communication was not free: compute kernels were 17-20% longer when heavily overlapped, an
  effect comparable in size to the straggler effects studied here.

These measurements do not establish general laws (see [Limitations](#limitations) and
[`docs/scientific_validity.md`](docs/scientific_validity.md)).

## Limitations
See [`docs/limitations.md`](docs/limitations.md). The most important ones:
- One machine, 2-4 GPUs, two workloads with synthetic inputs.
- An emulated straggler with one injection position (before the forward pass).
- PCIe paths go through host memory because PCIe P2P returned a wrong result on this host.
- Coarse NVML sensors; the software power-limit policy constrained clocks in almost every window.
- No hardware-throttling results (no permission).
- A shared host: other users' jobs contaminated some runs. These were detected and excluded, with one
  block at n = 4.
- The A results depend on which rank is slow; only two placements were tested.
- Mechanisms (contention, buffer broadcast, waiting power) are inferred, not isolated by controls.

## Related work
Stragglers in synchronous data-parallel training, and collectives or training schemes that tolerate them,
have extensive prior work. This repository makes **no novelty claim**. Examples:

- **Backup workers and bounded staleness:**
  - J. Chen et al., *Revisiting Distributed Synchronous SGD*, arXiv:1604.00981, 2016.
  - Q. Ho et al., *More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server*,
    NeurIPS 2013.
- **Straggler-tolerant / partial collectives and heterogeneity-aware decentralized training:**
  - S. Li et al., *Taming Unbalanced Training Workloads in Deep Learning with Partial Collective
    Operations* (eager-SGD), PPoPP 2020.
  - Q. Luo et al., *Prague: High-Performance Heterogeneity-Aware Asynchronous Decentralized Training*,
    ASPLOS 2020.
- **Tail-tolerant allreduce:** E. Warraich et al., *OptiReduce: Resilient and Tail-Optimal AllReduce for
  Distributed Deep Learning in the Cloud*, NSDI 2025.
- **Straggler diagnosis at scale:** T. Wu et al., *FALCON: Pinpointing and Mitigating Stragglers for
  Large-Scale Hybrid-Parallel Training*, arXiv:2410.12588, 2024.
- **DDP bucketing, overlap and communication scheduling:**
  - S. Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel Training*, VLDB 2020.
  - Y. Peng et al., *A Generic Communication Scheduler for Distributed DNN Training Acceleration*
    (ByteScheduler), SOSP 2019.
- **Compute slowdown under communication overlap:** Lee et al., *Characterizing Compute-Communication
  Overlap in GPU-Accelerated Distributed Deep Learning*, ISPASS 2025, arXiv:2507.03114 (reports a similar
  average slowdown on
  different hardware and workloads; supportive, not a replication).

## Reproducibility

### Install
Analysis, tests and figures need only CPU packages and work on Linux, macOS and Windows:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt
```

Re-running the experiments needs Linux, NVIDIA GPUs with the driver's NVML library, and the recorded
PyTorch build (CUDA 12.4 wheels, which bundle NCCL 2.21.5):

```bash
pip install -r requirements-experiment.txt
```

The recorded experiment used Python 3.13.12 with the PyTorch/torchvision versions pinned in
`requirements-experiment.txt` and Matplotlib 3.10.9. The NumPy and SciPy versions of the experiment host
were not recorded; the processed tables in this repository were reproduced with the versions pinned in
`requirements-analysis.txt` (all CSV tables byte-identical, summary values equal to within 1e-9 relative).

### Reproduce the analysis and figures from the raw data (no GPU needed)

```bash
python -m hddp.analyze --stage final            # -> results/processed/final_* and nccl_verification.csv
python -m hddp.analyze --stage pilot            # -> results/processed/pilot_*
python -m hddp.figures --stage final            # -> results/figures/fig01..fig11
python scripts/check_reproducibility.py         # re-runs the analysis in a temp dir and compares
python -m pytest                                # parsing, statistics, data invariants, privacy, safety
```

`python -m hddp.figures --stage final --outdir /tmp/figs` writes the figures elsewhere for comparison.

### Re-run the experiment (GPU host)
A privacy-safe command sequence. Each stage resumes from where it stopped. Choose GPUs with `--gpus`
(env stage); every later stage uses the GPU sets recorded in `results/env/topology.json`.

```bash
python -m hddp.env --gpus 0,1,2,3       # characterization, topology parse, P2P correctness probe
python -m hddp.run smoke --dry          # print the plan only (no GPU access, nothing written)
python -m hddp.run smoke                # smoke test (2 GPUs, both models, NCCL verify path)
python -m hddp.run pilot                # pilot sweep + instrumentation overhead check
python -m hddp.analyze --stage pilot
python -m hddp.run verify               # NCCL algorithm verification (NCCL_DEBUG logs)
python -m hddp.run final                # final matrix (about 95 min on the study host)
python -m hddp.run throttle --dry       # optional; prints what would be power-capped, touches nothing
python -m hddp.run throttle             # optional; needs permission to set GPU power limits
python -m hddp.analyze --stage final
python -m hddp.figures --stage final
python scripts/sanitize_data.py         # redact host/user identifiers before sharing new data
scripts/nsys.sh                         # optional: Nsight traces of 3 conditions (not run in this study)
```

Do not pass `--test-default-p2p` to the env stage on a shared machine unless you accept that NCCL's default
PCIe P2P path may hang or return wrong results there.

| figure | produced by (`hddp/figures.py`) | data |
|---|---|---|
| fig01 step time vs measured δ | `by_delay(..., "step_ms")` | `final_conditions.csv` |
| fig02 throughput vs δ | `by_delay(..., "throughput")` | same |
| fig03 A(δ) vs δ (+ rank-0 control) | `by_delay(..., "A")` | same |
| fig04a/b J/step and idle-subtracted J/step vs δ | `by_delay(..., "energy_*")` | same |
| fig05 overlap ratio vs δ | `by_delay(..., "overlap_ratio_fast")` | same |
| fig06/07/07b bucket vs throughput / energy / A per δ | `by_bucket` | same |
| fig08 Auto vs Ring vs Tree | `algo_cmp` | same |
| fig09 homogeneous- vs heterogeneous-optimal config | `homo_vs_het` | `final_summary.json` (H3) |
| fig10 profiler timeline | `timeline` | `results/raw/final.jsonl` (stored kernel intervals) |
| fig11 slack model vs measured A | `slack_scatter` | `final_trials.csv` |

### Layout
- `hddp/env.py` — characterization, topology parser, P2P correctness probe
- `hddp/worker.py` — DDP worker: straggler, timing, NVML, profiler
- `hddp/run.py` — orchestrator: plans, randomization, GPU-idle checks, timeouts, process cleanup, NCCL log
  parsing, optional throttling
- `hddp/nvml.py` — ctypes NVML, throttle guard with best-effort restoration
- `hddp/analyze.py` — tables and statistical tests
- `hddp/figures.py` — plots
- `hddp/privacy.py` — redaction helpers used by the experiment code and the sanitizer
- `scripts/` — `check_reproducibility.py`, `sanitize_data.py`, `nsys.sh`
- `tests/` — automated tests (`python -m pytest`)
- `results/raw` — JSONL per stage, logs, gzipped NCCL logs
- `results/processed` — CSV/JSON tables
- `results/figures` — PNG figures
- `docs/` — [methodology](docs/methodology.md), [limitations](docs/limitations.md),
  [data dictionary](docs/data_dictionary.md), [scientific-validity notes](docs/scientific_validity.md)

### Privacy and redaction
The published raw data and logs were passed through [`scripts/sanitize_data.py`](scripts/sanitize_data.py).
It replaces the host name, user and installation paths, network addresses, GPU UUIDs, launch-attempt ids
and other tenants' process ids with neutral placeholders, removes wall-clock timestamps, and leaves every
measurement untouched. The processed tables regenerate identically from the redacted data (the only
differences are the placeholder attempt ids listed in `final_summary.json`). New runs write privacy-safe
identifiers directly, but third-party warnings in the logs can still contain local paths, so run the
sanitizer before sharing new data. Details are in the [data dictionary](docs/data_dictionary.md).
