# Detailed results

All numbers are measured and taken from
[`final_conditions.csv`](../results/processed/final_conditions.csv),
[`final_h2.csv`](../results/processed/final_h2.csv) and
[`final_summary.json`](../results/processed/final_summary.json). They are mean ± 95% t-CI over **n = 5
independent launches** (n = 4 for 4-GPU ResNet-50 Ring) unless stated otherwise. The 30 steps of a timed
window are averaged into one value per launch; steps are never treated as independent replicates.

A is the slowdown amplification factor, A(δ) = (step(δ) − step(0)) / δ, with δ the measured injected delay
and step(0) the δ = 0 condition of the **same launch and bucket size**. A = 1 means the delay passes into the
step time one-for-one, A > 1 amplification, A < 1 partial absorption.

The delay is a calibrated GPU busy-wait (`torch.cuda._sleep`) on one rank before each forward pass. It
shifts that rank's whole step; it does not reproduce a genuinely slower or power-capped GPU, which would
stretch every kernel. Unless a control is named, the explanations below are interpretations of measured
associations, not demonstrated mechanisms.

## Data integrity
- **Completeness.** 59 of 60 planned launches and 792 of 810 planned conditions have clean data
  (`final_summary.json`: `launches`, `completeness`). The missing 18 conditions are the whole launch
  `final|4gpu|resnet50|Ring|t1`: all seven complete attempts were rejected by the forward-time check (six of
  them were also flagged at run time), and an eighth attempt was interrupted after 5 conditions. The
  degradation was attributed to other tenants' jobs on the shared host. That block therefore has n = 4.
  Across its 18 conditions the four clean trials vary by 0.29% (median CV; max 0.52%). Nothing is imputed.
- **Excluded attempts** (`final_summary.json["attempts"]`): 73 attempts in total, 59 accepted, 14 excluded.
  Reasons (an attempt can have several): foreign GPU process (4 attempts), forward-time anomaly (8, one of
  which also had a foreign process), and 3 interrupted partial attempts without a launch record. Excluded
  attempts stay in the raw data and never enter any statistic.
- **Contamination.** Another user's jobs appeared on the study's GPUs during 3 launches and degraded 4-GPU
  runs from other GPUs through shared host resources (8 attempts, 10-67% slower). In the kept data every
  condition lies within 2.93% of the cross-trial median of the same condition.
- **Node crash (not attributed to this study).** The server later crashed while the study's jobs were idle.
  All raw data survived, and the planned Nsight Systems traces were not recorded.
- **Validity checks.**
  - Delay calibration: the measured δ averages within 0.8% of the requested value in every cell (within
    0.25% on 4 GPUs); individual windows fall between −2.0% and +1.7%.
  - Auto vs Ring where NCCL verifiably chose Ring for Auto too: the two independently launched runs differ
    by at most 0.28% (0/18 cells with Welch p < 0.05).
  - NCCL forcing: 72 of 72 forced Ring/Tree conditions were confirmed in NCCL's logs.
  - Instrumentation overhead: 0.3-1.2% of step time (pilot).
  - A from mean vs median step time differs by 0.011 on average.

## Homogeneous baselines (δ = 0, NCCL Auto, 25 MB buckets)
| setting | ResNet-50 step / throughput | ViT-B/16 step / throughput |
|---|---|---|
| 2 GPU, NVLink | 64.3 ± 0.1 ms / 1992 img/s | 87.9 ± 0.2 ms / 728 img/s |
| 2 GPU, no NVLink (NCCL SHM via host memory) | 79.1 ± 0.4 ms / 1618 img/s | 131.7 ± 1.5 ms / 486 img/s |
| 4 GPU, NVLink + SHM | 80.6 ± 0.3 ms / 3175 img/s | 138.9 ± 0.9 ms / 922 img/s |

The two 2-GPU settings differ in the NCCL transport (P2P over NVLink vs. shared memory through the host,
because PCIe P2P returned a wrong result on this host), not only in the physical link. The difference is
therefore not a measurement of NVLink vs. functioning PCIe P2P.

**Compute kernels took longer while NCCL kernels ran concurrently.** In the 4-GPU ViT runs, the compute
kernels of one step (profiler, compute-busy time) take 126.9 ms on the non-delayed ranks but 116.1 ms on
rank 3, which carries less concurrent NCCL/SHM traffic. When the ranks desynchronize (δ = 40), the
non-delayed ranks' compute time drops to 105.3 ms (−17%) and rank 3's rises to 123.6 ms. Compute kernels
were thus 17-20% longer under heavy concurrent NCCL activity in these profiler windows. Contention for SMs
and memory bandwidth is a plausible explanation; no experiment held the work and the communicated bytes
fixed while varying the concurrency, so the causal direction is not established.

## H1: does synchronous DDP amplify a local slowdown? Not supported on this host
A(δ) at NCCL Auto and 25 MB buckets (delay on the last rank unless noted):

| setting | ResNet-50 δ=10 | ResNet-50 δ=40 | ViT-B/16 δ=10 | ViT-B/16 δ=40 |
|---|---|---|---|---|
| 2 GPU, NVLink | 0.940 ± 0.007 | 0.977 ± 0.002 | 0.955 ± 0.006 | 0.976 ± 0.004 |
| 2 GPU, no NVLink (SHM) | 0.967 ± 0.044 | 1.010 ± 0.012 | 0.992 ± 0.316 | 0.981 ± 0.035 |
| 4 GPU, delay on rank 3 | 0.952 ± 0.036 | 0.989 ± 0.011 | **0.004 ± 0.069** | **0.748 ± 0.016** |
| 4 GPU, delay on rank 0 (control) | 1.003 ± 0.122 | 0.997 ± 0.007 | 0.989 ± 0.047 | 1.005 ± 0.023 |

- **No systematic amplification.** Across all 112 tested cells A ranges from −0.25 to 1.03. One cell is
  nominally above 1 (2-GPU SHM ResNet-50, δ = 40: A = 1.010 ± 0.012, one-sided p < 0.05, uncorrected). The
  112 comparisons are correlated (they share launches, configurations and baselines) and are not corrected
  for multiplicity, so some nominal false positives are expected. This single small value does not
  establish straggler amplification.
- **The delay mostly passes through one-for-one** (A ≈ 0.94-1.0). The exception is a rank with measured
  **slack**, i.e. one that already idles while waiting on communication:
  - 4-GPU ViT, rank 3 delayed: δ = 10 ms is completely absorbed (A ≈ 0), and about 75% of a 40 ms delay
    reaches the step time.
  - Delaying rank 0 instead, which has no measured slack, gives A ≈ 1.0. In this setup the position of the
    delayed rank mattered as much as the size of the delay.
  - An exploratory slack model formulated after the pilot, A ≈ max(0, δ − slack)/δ, correlates with the
    measured A (Pearson r = 0.89 for ViT, 0.59 for ResNet;
    [fig11](../results/figures/fig11_slack_model.png)). It is descriptive, not a calibrated or causal model,
    and it under-predicts A.
- **A 5 ms delay coincided with a slightly shorter step.** 4-GPU ViT, rank 3 delayed by 5 ms:
  A = −0.25 ± 0.19 (n = 5), about 1.2 ms (0.9%) per step. This is a small observation in one cell; reduced
  interference on the critical ranks is one possible explanation, but no experiment tested it.
- **ResNet-50 waits inside the forward pass.** At δ = 40 the non-delayed ranks' forward-side wait rises by
  26.3 ms and the backward-side wait by 17.1 ms; ViT, which has no buffers, waits only in backward. This is
  consistent with DDP's default `broadcast_buffers=True`, which synchronizes BatchNorm buffers at the start
  of every forward pass. A `broadcast_buffers=False` control was not run.

## Throughput, GPU-board energy, NCCL concurrency (Auto, 25 MB)
- **Throughput loss at δ = 40** ranges from 17.7% (4-GPU ViT, rank 3 delayed) to 37.8% (2-GPU NVLink
  ResNet) ([fig02](../results/figures/fig02_throughput_vs_delay.png)).
- **GPU-board energy per step grew less than step time**
  ([fig04a](../results/figures/fig04a_energy_vs_delay.png),
  [fig04b](../results/figures/fig04b_energy_idle_sub_vs_delay.png)). Energy is the NVML cumulative counter of
  the GPU boards, summed over the job's GPUs; it excludes CPU, DRAM, network and storage.
  - 2-GPU NVLink ResNet, δ = 40: step time +61%, J/step +30.5 ± 2.9%, energy above idle +18.3 ± 3.8%.
  - 4-GPU ViT, δ = 40: step time +22%, J/step +6.5 ± 2.5%, energy above idle −0.1 ± 3.2% (no detectable
    change).

  "Energy above idle" subtracts each GPU's idle power over the window; it still contains compute,
  communication and waiting, and is not a measure of communication energy. Lower power while waiting would
  explain the pattern, but no power-state decomposition was done.
- **NCCL-kernel time concurrent with compute** (non-delayed ranks; `overlap_ratio_fast`) falls from
  0.84-0.98 at δ = 0 to 0.37-0.55 at δ = 40 ([fig05](../results/figures/fig05_overlap_vs_delay.png)). The
  metric is the fraction of NCCL-kernel residency during which a non-NCCL, non-delay kernel also runs on the
  same GPU. NCCL kernels are resident while they wait for peers, so the decline mostly reflects waiting for
  the delayed rank; it does not measure hidden data transfer or achieved bandwidth (methodology §8).

## H2: does bucket size change sensitivity? Yes for ViT-B/16; small and mostly not significant for ResNet-50
The six bucket sizes of a launch are repeated measurements of that launch, each paired with its own δ = 0
baseline. The primary test is a repeated-measures ANOVA of trial-level A across bucket sizes (launch =
subject) with the Greenhouse-Geisser correction, Holm-adjusted across the 16 tested (model, algorithm, δ)
cells on 4 GPUs. Full results, effect sizes and the secondary analyses are in
[`final_h2.csv`](../results/processed/final_h2.csv); the method is in methodology §10.

| analysis of A (16 cells) | cells with adjusted p < 0.05 |
|---|---|
| Repeated measures, Greenhouse-Geisser, Holm (primary) | 10 |
| Repeated measures, sphericity assumed, Holm | 15 |
| Friedman test (rank-based), Holm | 13 |
| One-way ANOVA ignoring the pairing (secondary), unadjusted / Holm | 15 / 15 |

- **ViT-B/16: significant in all 8 cells** (adjusted p ≤ 0.015, generalized η² = 0.77-0.96); the bucket
  means of A differ by 0.06-0.56.
  - Forced Tree, δ = 10: A = 0.36-0.58 across bucket sizes (adjusted p = 4×10⁻⁵); throughput loss 3.1-4.9%.
    With the algorithm forced, this is a bucket-size effect.
  - NCCL Auto, δ = 10: A ≈ 0 for 1-64 MB but 0.50 at 128 MB (adjusted p = 5×10⁻⁶). This is **not** a pure
    bucket-size effect: in the verification logs NCCL Auto chose Tree for 7 of 10 gradient all-reduces at
    128 MB, but for at most 1 per bucket size at 1-64 MB (≤ 6%). The 128 MB point mixes bucket size and
    collective algorithm ([fig07b](../results/figures/fig07b_bucket_vs_amplification.png), shaded).
- **ResNet-50: significant in 2 of 8 cells** (forced Tree, δ = 10 and 40; adjusted p = 0.015 and 2×10⁻⁴).
  In the other six cells (all Auto and Ring cells) the adjusted p is 0.07-0.24. Across all ResNet cells A
  varies by only 0.02-0.09 between bucket sizes, and throughput loss by ≤ 2.9 percentage points.
- **Other metrics (primary analysis):** the paired throughput degradation depends on bucket size in 15 of
  16 cells (exception: ResNet-50 Auto δ = 5), the paired energy increase in 2 of 16 (ViT Auto and Ring at
  δ = 40).
- With 4-5 launches and 6 bucket sizes sphericity cannot be tested, so the conservative Greenhouse-Geisser
  p-values are primary.
- **Sensitivity is not speed.** A configuration can absorb a delay better while being slower overall
  ([fig06](../results/figures/fig06_bucket_vs_throughput.png)).

## H3: does the homogeneous optimum stop being optimal? No detectable loss within the tested matrix
Among 18 configurations on 4 GPUs (3 NCCL algorithms × 6 bucket sizes)
([fig08](../results/figures/fig08_algo_comparison.png), [fig09](../results/figures/fig09_homo_vs_het_optimal.png)):
- **Throughput:** the δ = 0 optimum (Tree/1 MB for ResNet-50, Tree/16 MB for ViT-B/16) is the best, or not
  significantly different from the best, at δ = 10 and 40. The largest regret is 0.08% (ViT, δ = 10;
  Tree/25 MB nominally best, Welch p = 0.70).
- **images/J:** the δ = 0 optimum is Tree/128 MB for both models. Its largest regret is 0.77% (ViT, δ = 40,
  rank 5 of 18, p = 0.36).
- This is a local negative result: no detectable performance loss within this host's tested matrix and
  delay model. The best configuration at each δ is selected and tested on the same small sample (winner's
  curse), and the result does not show that re-tuning is unnecessary on other systems or under real
  hardware slowdowns.
- **Forced Tree outperformed NCCL Auto at every delay** on this NVLink + SHM topology: ResNet-50 3557 vs
  3378 img/s at δ = 0 (+5%); ViT-B/16 1165 vs 1057 img/s (+10%), +19% vs forced Ring. Auto chose Ring for
  almost all bucket all-reduces here.

## Figures
| figure | content | produced by (`hddp/figures.py`) | data |
|---|---|---|---|
| fig01 | step time vs measured δ | `by_delay(..., "step_ms")` | `final_conditions.csv` |
| fig02 | throughput vs δ | `by_delay(..., "throughput")` | same |
| fig03 | A(δ) vs δ, with the rank-0 control | `by_delay(..., "A")` | same |
| fig04a/b | GPU-board energy per step, and above idle | `by_delay(..., "energy_*")` | same |
| fig05 | NCCL-kernel time concurrent with compute | `by_delay(..., "overlap_ratio_fast")` | same |
| fig06/07/07b | throughput / energy / A vs bucket size | `by_bucket` | same, plus `nccl_verification.csv` |
| fig08 | NCCL Auto vs Ring vs Tree | `algo_cmp` | same |
| fig09 | homogeneous-optimal vs best configuration per δ | `homo_vs_het` | `final_summary.json` (H3) |
| fig10 | profiler timeline (one window of one launch) | `timeline` | `results/raw/final.jsonl` |
| fig11 | exploratory slack model vs measured A | `slack_scatter` | `final_trials.csv` |
