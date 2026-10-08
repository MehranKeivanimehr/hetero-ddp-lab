# Scientific-validity notes

This note summarizes an internal review of the study against the PyTorch and NVIDIA documentation and the
literature, and records what was changed in response. It does not re-run any GPU experiment; it checks
internal consistency, the analysis choices and the agreement with outside evidence.

## Overall assessment
The study is best read as a **credible exploratory characterization of one unusual machine** (NVLink pairs
joined through host memory because PCIe P2P failed), not as evidence for a general law.

| dimension | assessment | reason |
|---|---|---|
| Agreement with PyTorch DDP mechanics | strong | gradient synchronization, bucket overlap (`bucket_cap_mb`), forward buffer broadcast (`broadcast_buffers`), CUDA-event timing and profiler use match the documented behaviour |
| Agreement with NVIDIA documentation | strong, after wording fixes | NCCL transport controls (`NCCL_P2P_LEVEL`, SHM fallback) and the NVML energy method are used as documented; the P2P root cause and the energy-polling effect were stated too strongly and have been reworded |
| Internal data consistency | strong | processed tables regenerate from the raw records; completeness and exclusions match the data |
| Statistical support | moderate to strong | H1 (no amplification) is robust; the H2 bucket effect is robust for ViT-B/16 but not for most ResNet-50 cells once the repeated-measures design and multiplicity are accounted for |
| Causal interpretation | moderate | contention, buffer broadcast and lower waiting power are plausible and profiler-consistent, but not isolated by controls |
| External validity | low | one host, one GPU generation, two models, synthetic inputs, 2-4 GPUs, SHM fallback, one synthetic pre-forward delay |

## Changes made in response

| issue | change |
|---|---|
| H2 used a one-way ANOVA that treats the six bucket conditions of a launch as independent | Repeated-measures ANOVA with launch as subject, Greenhouse-Geisser correction, Holm adjustment across the 16 cells, generalized and partial η², bucket-mean ranges, paired contrasts, a Friedman sensitivity analysis; the old ANOVA is kept and labelled secondary (methodology §10). Result for A: 10/16 significant cells (primary) instead of 15/16 |
| "PCIe P2P is broken ... probably the IOMMU/ACS problem" | "PCIe P2P produced an incorrect result on this host; IOMMU/ACS is a plausible NVIDIA-documented cause but was not diagnosed" |
| P2P probe checked only `x[0]` and one representative pair | Probe now uses a position-dependent pattern, checks the full tensor and a checksum on every rank, and covers every non-NVLink pair plus each multi-GPU group; default-path testing stays opt-in. The published record still comes from the old probe and is labelled as such |
| "Polling the NVML energy counter corrupts it" | "Frequent queries under-reported cumulative energy in a diagnostic on driver 580.173.02; the cause is unknown" |
| "Guaranteed restore" of GPU power limits | "Best-effort restoration on normal exit, exceptions, SIGINT, SIGTERM and SIGHUP"; impossible after SIGKILL, host crash or power loss. The throttle stage never ran, so no result is affected |
| "GPUs run at their 300 W power cap" | "The software power-limit policy (`sw_power_cap`) was reported at the end of 783 of 792 windows"; this constrains clocks and does not imply a continuous 300 W draw |
| Mechanisms stated as facts (overlap slowdown, speed-up at δ = 5 ms, forward-side waiting, lower waiting power) | Reworded as measured associations with plausible, non-isolated mechanisms |
| "About 5-6 of 112 cells significant by chance" | The 112 H1 comparisons are correlated, so that count assumed independence; now: some nominal false positives are expected, and the single A > 1 does not establish amplification |
| "Overlap ratio" read as hidden communication | Defined as NCCL-kernel residency concurrent with compute kernels, which includes waiting; not a transfer or bandwidth measure |
| Energy described without scope | GPU-board energy (NVML) only; idle-subtracted energy is not communication energy |
| Summary reported all 60 final launches as `ok` | Launch status is derived from the clean data: 59 clean, 1 incomplete; attempt outcomes are reported separately |
| Conclusions phrased generally | Limited to this host, topology, workload set and synthetic straggler model |
| `--dry` ignored for the throttle stage; process-watch start-up outside the cleanup block | Fixed; worker cleanup now also finds re-parented workers through a per-launch environment token |
| No environment specification, tests or data dictionary | Added `requirements-*.txt`, `tests/`, `scripts/check_reproducibility.py`, `docs/data_dictionary.md` |

## Agreement with outside evidence

- **Slow-rank propagation.** PyTorch's performance guide states that in synchronous data parallelism all
  devices wait for the worker that finishes last. This supports the premise but does not predict A > 1; a
  result near A = 1, or below 1 when existing slack absorbs part of the delay, is plausible. The study's
  defensible statement is: on this host, a fixed pre-forward delay usually passed into step time about
  one-for-one, and part of it was absorbed when the delayed rank had measured slack.
- **Bucket size.** PyTorch documents `bucket_cap_mb` as controlling which reductions can overlap the
  backward pass. This supports a mechanism for H2 but not that one bucket size is universally best.
- **Forward-side waiting in ResNet-50.** `broadcast_buffers=True` synchronizes module buffers at the start
  of forward, consistent with the observed forward-side wait; a `broadcast_buffers=False` control would be
  needed to attribute the whole effect to it.
- **Longer compute kernels under concurrent communication.** An independent study reports an average compute
  slowdown of about 19% under communication overlap on different hardware and workloads, close to the
  17-20% longer compute-kernel time observed here while NCCL kernels ran concurrently. This is supportive,
  not a replication, and the observation here is an association, not a controlled causal result.
- **Topology-dependent collectives.** Work on topology-aware collective synthesis supports comparing Ring,
  Tree and Auto; it does not imply that Tree beats Auto on other systems.
- **NVML energy.** `nvmlDeviceGetTotalEnergyConsumption` is a cumulative millijoule counter, so differences
  between window edges are a valid window-energy measurement.

## Recommended further controls (not done)
1. Diagnose the P2P failure: kernel command line, IOMMU messages in `dmesg`, ACS state from
   `lspci -vvv`, `nvidia-smi topo -p2p`, NVIDIA `simpleP2P` and `nvbandwidth`; re-run the strengthened probe.
2. Record model-parameter and gradient checksums after DDP synchronization.
3. Add a `broadcast_buffers=False` control for ResNet-50.
4. Separate overlap from contention with a microbenchmark that keeps compute work and communicated bytes
   fixed while toggling overlap.
5. Replace or complement the pre-forward spin with a genuine slowdown (power or clock cap) or delays
   distributed across layers.
6. Repeat on a host with working PCIe P2P, an NVSwitch system, another GPU generation, more workloads and
   batch sizes, and ideally multiple nodes.

## Sources
- PyTorch `DistributedDataParallel` API (`bucket_cap_mb`, `broadcast_buffers`):
  https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html
- PyTorch DDP communication hooks: https://docs.pytorch.org/docs/2.14/ddp_comm_hooks.html
- PyTorch performance tuning guide: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide
- PyTorch CUDA semantics (asynchronous execution):
  https://docs.pytorch.org/docs/main/notes/cuda.html#asynchronous-execution
- NCCL 2.21.5 environment variables:
  https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2215/user-guide/docs/env.html
- NCCL GPU troubleshooting (IOMMU, ACS, P2P validation):
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html
- CUDA programming guide, multi-GPU systems (IOMMU and ACS):
  https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#host-iommu-hardware-pci-access-control-services-and-vms
- NVML device queries, field values and clock-event reasons:
  https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html,
  https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html,
  https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html
- Chen et al., *Revisiting Distributed Synchronous SGD*, ICLR Workshop 2016, https://arxiv.org/abs/1604.00981
- Li et al., *Taming Unbalanced Training Workloads in Deep Learning with Partial Collective Operations*,
  PPoPP 2020, https://arxiv.org/abs/1908.04207
- Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel Training*, VLDB 2020,
  https://www.vldb.org/pvldb/vol13/p3005-li.pdf
- Peng et al., *A Generic Communication Scheduler for Distributed DNN Training Acceleration*, SOSP 2019,
  https://i.cs.hku.hk/~cwu/papers/yhpeng-sosp19.pdf
- Wang et al., *Blink: Fast and Generic Collectives for Distributed ML*, MLSys 2020,
  https://proceedings.mlsys.org/paper_files/paper/2020/file/cd3a9a55f7f3723133fa4a13628cdf03-Paper.pdf
- Shah et al., *TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches*, NSDI 2023,
  https://www.usenix.org/conference/nsdi23/presentation/shah
- Lee et al., *Characterizing Compute-Communication Overlap in GPU-Accelerated Distributed Deep Learning*,
  ISPASS 2025, https://arxiv.org/abs/2507.03114
- You et al., *Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training*, NSDI 2023,
  https://www.usenix.org/conference/nsdi23/presentation/you
- Georges et al., *Statistically Rigorous Java Performance Evaluation*, OOPSLA 2007,
  https://dri.es/files/oopsla07-georges.pdf
