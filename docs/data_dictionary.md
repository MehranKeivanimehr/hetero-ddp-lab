# Data dictionary

Units: ms = milliseconds, s = seconds, J = joules, W = watts, MHz, °C. "Window" = the timed window of one
condition (M = 30 steps in the pilot and final stages). Metric definitions are in
[`methodology.md`](methodology.md); this file describes where each value is stored.

## Files

| path | format | produced by | content |
|---|---|---|---|
| `results/env/machine.txt` | text | `hddp.env` | redacted output of `nvidia-smi`, `nvidia-smi topo -m`, `nvidia-smi nvlink -s`, GPU queries, `lscpu`, `numactl -H`, `lspci -tv` |
| `results/env/topology.json` | JSON | `hddp.env` | software versions, GPU link matrix, NUMA/CPU affinity, derived settings, P2P probe results, NCCL environment used by every launch |
| `results/raw/<stage>.jsonl` | JSON Lines | `hddp.run` + `hddp.worker` | one record per line, see [Raw records](#raw-records); stages `smoke`, `pilot`, `verify`, `final`, `throttle` |
| `results/raw/<stage>.log` | text | `hddp.run` | orchestrator and torchrun console output per launch (redacted) |
| `results/raw/nccl_logs.txt.gz` | gzip text | `hddp.run` (verify launches) | `NCCL_DEBUG=INFO` logs of the NCCL verification launches; one section per process, headed `##### <launch_id> <file>` |
| `results/processed/<stage>_trials.csv` | CSV | `hddp.analyze` | one row per clean (launch, condition) = one trial |
| `results/processed/<stage>_conditions.csv` | CSV | `hddp.analyze` | per condition: mean, 95% CI and n across trials of every metric |
| `results/processed/<stage>_h2.csv` | CSV | `hddp.analyze` | H2 bucket-size tests, one row per (cell, metric) |
| `results/processed/<stage>_summary.json` | JSON | `hddp.analyze` | data quality, completeness, calibration, NCCL verification, H1-H3 results |
| `results/processed/nccl_verification.csv` | CSV | `hddp.analyze` | requested vs. NCCL-selected allreduce algorithm per (setting, model, algorithm, bucket) |
| `results/figures/fig*.png` | PNG | `hddp.figures` | figures; see the table in the README |

CSV floats are rounded to 5 decimals (6 significant digits in `<stage>_h2.csv`); the JSON summary keeps
full precision. Empty CSV cells mean "not measured / not applicable", never zero. Only the `final` and
`pilot` stages are analysed; `smoke`, `verify` and `throttle` raw data are kept as evidence.

## Identifiers and redaction

| field / text | meaning |
|---|---|
| `launch_id` | `<stage>\|<setting>\|<model>\|<algo>\|t<trial>[\|<tag>]`; one launch = one torchrun process group |
| `attempt` | one execution of a launch. Historical attempts are `attempt-NNN` (numbered in order of first appearance; replaces an id that contained a clock time and a process id). New runs write a random 8-hex-digit id. Missing/`null` on records written before attempt tracking existed: then the launch id identifies the attempt |
| `P1`, `P2`, ... | pseudonyms of other tenants' process ids (the same process keeps the same pseudonym across the dataset). New runs write `P` + 8 hex digits from a salted hash that is not stored |
| `host`, `<pid>`, `<tid>` | host name and this study's own process/thread ids in NCCL log lines |
| `<python-env>`, `<repo>`, `/home/<user>` | Python installation prefix, checkout path and home directory in log lines |
| `GPU-<uuid>`, `<ip>` | GPU serial identifiers and network addresses |
| `tag` | `""` main sweep, `strag0` straggler on rank 0, `noinstr` no comm hook / no per-step NVML (pilot), `verify` NCCL verification inside the smoke test, `pl<W>` power-limit run (throttle stage, never executed) |

Wall-clock timestamps were removed from all records and logs. The order of records in each JSONL file is
the order in which they were written.

## Raw records

Every JSONL line has a `type`.

### `type = "cond"`: one condition of one launch (written by rank 0)
| field | meaning |
|---|---|
| `launch_id`, `stage`, `setting`, `gpus`, `algo`, `trial`, `tag`, `attempt` | launch metadata (see above); `gpus` are physical GPU indices |
| `model` | `resnet50` or `vit_b_16` |
| `world`, `batch` | number of ranks, per-GPU batch size |
| `bucket` | DDP `bucket_cap_mb` |
| `delay_req` | requested straggler delay (ms) |
| `cond_index` | position of the condition in the launch's shuffled order |
| `warmup`, `steps`, `profile_steps` | number of warmup, timed and profiled steps |
| `instrument` | 1 = comm hook and NVML sampler on, 0 = off (overhead check) |
| `straggler_rank` | rank that runs the spin delay |
| `ranks` | list with one object per rank (below) |

Per rank (`ranks[i]`):

| field | meaning |
|---|---|
| `rank`, `phys_gpu`, `straggler` | rank, physical GPU index, whether this rank is the straggler |
| `wall_s` | wall-clock duration of the timed window (s) |
| `energy_j` | NVML cumulative-energy difference over the window (J) |
| `energy_sampled_j` | trapezoid integral of the 20 ms instantaneous-power samples, scaled to the window (J); cross-check |
| `idle_w` | idle power of this GPU with the CUDA context resident, measured at launch start (W) |
| `cycles`, `cpm` | spin length (GPU clock cycles) used in the window; initial cycles-per-ms calibration (straggler only) |
| `steps` | per-step lists (ms unless noted): `delay`, `fwd`, `bwd`, `opt`, `gpu_step`, `period` (M−1 values), `first_ready`, `last_ready`, `exposed_comm`, `n_buckets` (count); see methodology §5 |
| `nvml` | per-step means of the 20 ms samples: `power_w`, `temp_c`, `sm_mhz`, `util_pct` (sample-and-hold; `null` if no sample fell inside a step) |
| `nvml_window_mean` | window means of the same four quantities |
| `n_samples` | number of NVML samples in the window |
| `throttle` | NVML clock-event reasons read once at the end of the window (e.g. `sw_power_cap`) |
| `foreign_pids` | pseudonyms of processes not belonging to this launch seen on this GPU at the end of the window |
| `prof` | profiler-window metrics (only for profiled conditions): `comm_ms`, `compute_ms`, `overlap_ms`, `exposed_comm_ms`, `overlap_ratio`, `span_ms`, `delay_kernel_ms` (per step), `nccl_kernels` (kernel name → count) |
| `timeline`, `timeline_t0_us` | kernel intervals `[class, start_us, end_us, stream]` (class `c` compute, `n` NCCL, `d` delay) for the conditions drawn in fig10; absolute start on the host clock (µs), used only to align ranks |
| `energy_raw_j`, `energy_interp`, `energy_ticks` | smoke stage only: fields of an earlier energy method, superseded before the pilot |

### `type = "launch"`: one attempt of a launch (written by the orchestrator)
| field | meaning |
|---|---|
| launch metadata | as in `cond`, plus `model`, `conditions` (shuffled `[bucket, delay]` list), `warmup`, `steps`, `profile_steps`, `timeline`, `profile_only`, `instrument`, `marker` (1 = NCCL verification launch), `straggler_rank` |
| `rc` | torchrun return code, or `"timeout"` / `"skipped_busy"` |
| `status` | `ok`, `failed`, `contaminated` (foreign process seen), `anomalous_timing` (second record of an attempt that finished `ok` but failed the timing/consistency checks), `skipped_busy` |
| `duration_s` | attempt duration (s) |
| `gpu_pre`, `gpu_post` | per GPU index: `temp` (°C), `power` (W), `util` (%), `procs` (`<pseudonym>:<process name>:<memory>` of compute processes) before and after the attempt |
| `foreign_watch` | continuous-watch sightings `[seconds since attempt start, GPU, pseudonym]` (only on attempts run after the watch was added) |
| `loadavg` | host 1/5/15-minute load average after the attempt |
| `busy` | unused (always `null`) |

### `type = "nccl_verify"`: NCCL's algorithm choice per condition of a verification launch
`setting`, `model`, `algo_requested`, `bucket`; `selected` maps `<collective>:<algorithm>/<protocol>` to the
number of calls; `env_lines` are NCCL's "set by environment" lines; `transports` counts channel transports.

### `type = "throttle_probe"`
`gpu`, `permitted` (bool) and `detail` (NVML error text) of the non-destructive power-limit permission probe.

## `topology.json`
| field | meaning |
|---|---|
| `versions` | Python, PyTorch, torchvision, CUDA runtime, cuDNN, NCCL, kernel release |
| `allowed_gpus`, `gpus` | GPUs the study was allowed to use; all GPUs found |
| `matrix` | `nvidia-smi topo -m` link type for each ordered GPU pair (`NV4`, `NODE`, `SYS`, ...) |
| `cpu_affinity`, `numa` | per GPU |
| `pairs_by_class` | all GPU pairs grouped by link class |
| `settings` | GPU sets used for each experiment setting, and the link classes inside multi-GPU sets |
| `nccl_env` | environment applied to every launch (`NCCL_P2P_LEVEL=NVL` here) |
| `p2p_check` | P2P probe results. The published entries come from the earlier probe and contain `pair`, `nccl_p2p_level`, `transport`, `correct`, `first_allreduce_s` and `value` (the first result element; expected = number of GPUs). The current probe writes `gpus`, `kind` (`pair`/`group`), `link_class`, `nccl_p2p_level`, `probe`, `transport`, `returncode`, `correct`, `failure`, `ranks_reported`, `mismatched_elements`, `first_allreduce_s` |

## Processed tables

### `<stage>_trials.csv` (one row per trial)
- **Keys:** `stage`, `setting`, `model`, `algo`, `bucket`, `delay_req`, `instrument`, `tag`, `trial`,
  `launch_id`, `cond_index`, `world`, `batch`, `steps`, `gpus`.
- **Timing (ms):** `step_ms` (mean step period, averaged over ranks; primary), `step_ms_wall`,
  `step_ms_median`, `period_rank_spread_ms`, `step_max_over_median` (ratio), `delay_meas_ms` and
  `delay_meas_sd` (measured straggler delay), `fwd_*`, `bwd_*`, `opt_*`, `exposed_comm_*`, `last_ready_*`
  (`_fast` = mean over non-straggler ranks, `_strag` = straggler), `exposed_comm_min`, per-rank `fwd_r<k>`
  and `last_ready_r<k>`; `n_buckets` (count); `cycles`.
- **Throughput and energy:** `throughput` (images/s), `energy_J_step`, `energy_idle_sub_J_step`,
  `images_per_J`, `power_W_gpu`, `power_W_fast`, `power_W_strag`, `idle_W_total` (W),
  `energy_sampled_vs_counter_pct`.
- **Telemetry:** `sm_mhz_fast`, `sm_mhz_strag`, `util_fast` (%), `temp_max` (°C), `throttle` (reasons seen
  on any rank), `foreign_procs` (count; 0 in all used data).
- **Profiler (profiled trials only):** `overlap_ratio_fast`, `overlap_ratio_strag`, `comm_ms_*`,
  `exposed_comm_prof_*`, `compute_ms_*`, `nccl_kernels`.
- **Paired with the δ = 0 condition of the same launch and bucket (δ > 0 only):** `A`, `A_median`,
  `A_slack_model`, `slack_strag_ms`, `thr_deg_pct`, `energy_inc_pct`, `energy_idle_sub_inc_pct`,
  `wait_fwd_inc_ms`, `wait_bwd_inc_ms`, `comm_wait_inc_ms`, `overlap_deg_pct`, `comm_wait_inc_prof_ms`.

### `<stage>_conditions.csv` (one row per condition)
The key columns of the trials table, `n_trials`, `throttle`, `foreign_procs`, and for every metric `<m>`:
`<m>_mean`, `<m>_ci95` (half-width of the 95% t-interval across trials; empty when n = 1) and `<m>_n`.

### `<stage>_h2.csv` (one row per H2 cell and metric)
`setting`, `model`, `algo`, `delay_req`, `metric` (`A`, `thr_deg_pct`, `energy_inc_pct`), `n_launches`,
`n_launches_incomplete_excluded`, `range_of_bucket_means`; repeated-measures ANOVA `rm_F`, `rm_df1`,
`rm_df2`, `rm_gg_epsilon`, `rm_p_gg` and `rm_p_gg_holm` (primary), `rm_p` and `rm_p_holm`
(sphericity assumed), `partial_eta2`, `generalized_eta2`; Friedman `friedman_chi2`, `friedman_p`,
`friedman_p_holm`, `kendalls_w`; post hoc contrast `contrast_high_bucket`, `contrast_low_bucket`,
`contrast_mean_diff`, `contrast_ci95`, `contrast_cohens_dz`; secondary independent-groups ANOVA
`secondary_independent_anova_F`, `_p`, `_p_holm`; bucket means `mean_b<bucket>`.

### `nccl_verification.csv`
`setting`, `model`, `algo_requested`, `bucket`, `allreduce_selected` (JSON: `<algo>/<protocol>` → calls),
`allreduce_algos` (algorithms seen), `honored` (forced algorithm used for every bucket; empty for Auto),
`env_lines`, `transports`.

### `<stage>_summary.json`
| key | content |
|---|---|
| `n_trials` | number of clean trials (condition windows) in the analysis |
| `launch_status`, `launches` | logical launches: planned, `clean` (the accepted attempt delivered every planned condition) and `incomplete` (fewer clean conditions than planned, including none). Derived from the clean data, not from the status the orchestrator recorded at run time |
| `incomplete_launches` | per incomplete logical launch: planned and clean condition counts and every attempt with its verdict |
| `attempts` | attempt level: total, accepted, excluded, counts of recorded run-time statuses, exclusion reasons, and the list of excluded attempts (`recorded_status`, `conditions_recorded`, `reasons`; an attempt with condition records but no launch record was interrupted) |
| `completeness` | planned vs. clean conditions and the list of missing (launch, bucket, delay) cells |
| `data_quality` | excluded attempts by reason (foreign process, forward-time anomaly, cross-trial deviation), forward-time references, throttle reasons, energy cross-check, maximum rank spread and temperature |
| `delay_calibration` | requested vs. measured delay per (setting, model, δ) |
| `robustness` | A from mean vs. median step time; Auto vs. Ring where NCCL chose Ring for Auto |
| `instrumentation_overhead` | instrumented vs. plain step time (pilot) |
| `nccl_verification_summary`, `nccl_kernels_in_profiled_runs` | NCCL algorithm checks |
| `H1`, `H1_slack_model`, `straggler_placement` | A vs. 1 per cell; slack model fit; rank-3 vs. rank-0 straggler |
| `H2_overview`, `H2` | number of significant cells per analysis; per-cell results (same content as `<stage>_h2.csv`, unrounded, plus bucket means) |
| `H3_throughput`, `H3_images_per_J` | homogeneous-optimal vs. best configuration per δ |

## Exclusions and incomplete data
- Analyses use only attempts that finished `ok` with no foreign process (in-window or continuous watch),
  no forward-time anomaly and no condition deviating more than 5% from the cross-trial median
  (`hddp.analyze.attempt_flags`). Excluded attempts stay in the raw files.
- Missing conditions are listed in `completeness["missing"]` and never imputed. In the final stage, all 18
  conditions of `final|4gpu|resnet50|Ring|t1` are missing, so that block has n = 4.
- In H2, launches without a value for every bucket size are excluded from the repeated-measures tests and
  counted in `n_launches_incomplete_excluded` (0 in the published data).
