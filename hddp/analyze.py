"""Raw JSONL -> processed tables.

  python -m hddp.analyze --stage final [--outdir DIR]     (also: pilot, smoke)

Writes results/processed/<stage>_trials.csv     one row per (launch, condition) = one trial
       results/processed/<stage>_conditions.csv mean / 95% CI (t) / n per condition across trials
       results/processed/<stage>_summary.json   H1-H3 tests, calibration, NCCL verification, data quality
       results/processed/<stage>_h2.csv         H2 bucket-size tests, one row per (cell, metric)
       results/processed/nccl_verification.csv  requested vs. NCCL-selected algorithm (from verify stage logs)
Nothing is imputed: a missing measurement stays empty and is counted in the data-quality section.
Floats in the CSV files are rounded to 5 decimals (6 significant digits in <stage>_h2.csv, which holds
p-values); the JSON summary keeps full precision.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "results" / "raw", ROOT / "results" / "processed"
KEY = ("stage", "setting", "model", "algo", "bucket", "delay_req", "instrument", "tag")


def _m(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _sd(v):
    v = [x for x in v if x is not None]
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


CONSISTENCY_TOL_PCT = 5.0  # kept attempts: every condition within 2.93% of the cross-trial median;
#                            degraded attempts: 10-67% (see the data-integrity notes). Any value between
#                            2.93 and 10 selects the same attempts on the published data.


def attempt_flags(rows):
    """Attempts that must not be used. Three independent rules (docs/methodology.md, contamination handling):
    foreign GPU process seen (in-window or continuous watch); forward-time anomaly; any condition's step time
    deviating > CONSISTENCY_TOL_PCT from the cross-trial median of the same condition (catches host-side
    interference from other tenants, including interference that starts mid-launch)."""
    att = lambda r: (r["launch_id"], r.get("attempt"))
    launches = [r for r in rows if r["type"] == "launch"]
    conds = [r for r in rows if r["type"] == "cond"]
    ok = {att(l) for l in launches if l["status"] == "ok" and not l.get("foreign_watch")}
    foreign = {att(r) for r in conds if any(k["foreign_pids"] for k in r["ranks"])}
    foreign |= {att(l) for l in launches if l.get("foreign_watch")}
    _, anomalies = timing_anomalies([r for r in conds if att(r) in ok])
    anomalous = {(a["launch_id"], a["attempt"]) for a in anomalies}
    keep = ok - foreign - anomalous
    step = lambda r: float(np.mean([np.mean(k["steps"]["period"]) for k in r["ranks"]]))
    groups = defaultdict(list)
    for r in conds:
        if att(r) in keep:
            groups[(r["setting"], r["model"], r["algo"], r["bucket"], r["delay_req"], r.get("tag") or "")].append(r)
    dev = defaultdict(list)
    for rs in groups.values():
        if len(rs) >= 3:
            med = np.median([step(r) for r in rs])
            for r in rs:
                dev[att(r)].append(100 * (step(r) - med) / med)
    outliers = {a for a, v in dev.items() if max(map(abs, v)) > CONSISTENCY_TOL_PCT}
    return dict(ok=ok, foreign=foreign, anomalous=anomalous, baseline_outliers=outliers,
                good=keep - outliers, baseline_dev_pct={str(a): float(max(v, key=abs)) for a, v in dev.items()})


def load(stage):
    """Only conditions from clean, completed launch attempts (see attempt_flags). Records without an 'attempt'
    field predate attempt tracking and form one attempt per launch id."""
    with open(RAW / f"{stage}.jsonl") as fh:
        rows = [json.loads(l) for l in fh]
    launches = [r for r in rows if r["type"] == "launch"]
    conds = [r for r in rows if r["type"] == "cond"]
    f = attempt_flags(rows)
    used = [r for r in conds if (r["launch_id"], r.get("attempt")) in f["good"]]
    load.flags = f
    load.excluded = dict(conditions=len(conds) - len(used),
                         attempts_with_foreign_procs=sorted(map(str, f["foreign"])),
                         attempts_with_forward_time_anomaly=sorted(map(str, f["anomalous"])),
                         attempts_with_cross_trial_deviation=sorted(map(str, f["baseline_outliers"])),
                         kept_attempts_max_condition_dev_pct=max(
                             (abs(v) for a, v in f["baseline_dev_pct"].items() if a in map(str, f["good"])),
                             default=None))
    return used, launches, rows


def trial_row(r):
    ks, M, W, B = r["ranks"], r["steps"], r["world"], r["batch"]
    fast = [k for k in ks if not k["straggler"]]
    slow = next(k for k in ks if k["straggler"])
    per = lambda k, f: _m(k["steps"][f])
    fm = lambda f: _m([per(k, f) for k in fast])
    periods = [per(k, "period") for k in ks]
    step = _m(periods)
    wall = max(k["wall_s"] for k in ks)
    E = sum(k["energy_j"] for k in ks)
    idle = sum(k["idle_w"] for k in ks)
    row = {k: r.get(k) for k in KEY}
    row["tag"] = row["tag"] or ""
    row.update(trial=r["trial"], launch_id=r["launch_id"], cond_index=r["cond_index"], world=W, batch=B, steps=M,
               gpus="-".join(map(str, r["gpus"])))
    row.update(
        step_ms=step, step_ms_wall=1000 * wall / M, period_rank_spread_ms=max(periods) - min(periods),
        step_ms_median=float(np.median([np.median(k["steps"]["period"]) for k in ks])),
        step_max_over_median=max(max(k["steps"]["period"]) / np.median(k["steps"]["period"]) for k in ks),
        throughput=W * B / step * 1000,
        delay_meas_ms=per(slow, "delay"), delay_meas_sd=_sd(slow["steps"]["delay"]),
        fwd_fast=fm("fwd"), bwd_fast=fm("bwd"), opt_fast=fm("opt"), exposed_comm_fast=fm("exposed_comm"),
        last_ready_fast=fm("last_ready"),
        fwd_strag=per(slow, "fwd"), bwd_strag=per(slow, "bwd"), opt_strag=per(slow, "opt"),
        exposed_comm_strag=per(slow, "exposed_comm"), last_ready_strag=per(slow, "last_ready"),
        exposed_comm_min=min((per(k, "exposed_comm") for k in ks if k["steps"]["exposed_comm"]), default=None),
        n_buckets=per(slow, "n_buckets"),
        energy_J_step=E / M, energy_idle_sub_J_step=(E - idle * wall) / M, images_per_J=W * B * M / E,
        power_W_gpu=E / wall / W, power_W_strag=slow["energy_j"] / slow["wall_s"],
        power_W_fast=_m([k["energy_j"] / k["wall_s"] for k in fast]), idle_W_total=idle,
        sm_mhz_fast=_m([k["nvml_window_mean"].get("sm_mhz") for k in fast]),
        sm_mhz_strag=slow["nvml_window_mean"].get("sm_mhz"),
        util_fast=_m([k["nvml_window_mean"].get("util_pct") for k in fast]),
        temp_max=max((max(t for t in k["nvml"]["temp_c"] if t is not None) for k in ks
                      if any(t is not None for t in k["nvml"].get("temp_c", []))), default=None),
        throttle="|".join(sorted({t for k in ks for t in k["throttle"]})),
        foreign_procs=sum(len(k["foreign_pids"]) for k in ks),
        energy_sampled_vs_counter_pct=(100 * (sum(k["energy_sampled_j"] for k in ks) - E) / E
                                       if all(k.get("energy_sampled_j") for k in ks) else None),
        cycles=slow["cycles"],
        **{f"fwd_r{k['rank']}": per(k, "fwd") for k in ks},
        **{f"last_ready_r{k['rank']}": per(k, "last_ready") for k in ks})
    if "prof" in slow:
        fp = [k["prof"] for k in fast]
        row.update(overlap_ratio_fast=_m([p["overlap_ratio"] for p in fp]),
                   overlap_ratio_strag=slow["prof"]["overlap_ratio"],
                   comm_ms_fast=_m([p["comm_ms"] for p in fp]), comm_ms_strag=slow["prof"]["comm_ms"],
                   exposed_comm_prof_fast=_m([p["exposed_comm_ms"] for p in fp]),
                   exposed_comm_prof_strag=slow["prof"]["exposed_comm_ms"],
                   compute_ms_fast=_m([p["compute_ms"] for p in fp]), compute_ms_strag=slow["prof"]["compute_ms"],
                   nccl_kernels="|".join(sorted({n for k in ks for n in k["prof"]["nccl_kernels"]})))
    return row


def add_paired(trials):
    """Straggler metrics paired against the delay=0 condition of the SAME launch (same process, bucket, algo)."""
    base = {(t["launch_id"], t["bucket"]): t for t in trials if t["delay_req"] == 0}
    for t in trials:
        b = base.get((t["launch_id"], t["bucket"]))
        if t["delay_req"] == 0 or b is None:
            continue
        d = t["delay_meas_ms"]
        t["A"] = (t["step_ms"] - b["step_ms"]) / d if d else None
        # slack model: at delay 0 the straggler's wait in excess of the least-waiting (critical) rank's wait is
        # idle time that can hide part of the injected delay: A_pred = max(0, delta - slack) / delta
        if d and b.get("exposed_comm_strag") is not None and b.get("exposed_comm_min") is not None:
            t["slack_strag_ms"] = b["exposed_comm_strag"] - b["exposed_comm_min"]
            t["A_slack_model"] = max(0.0, d - t["slack_strag_ms"]) / d
        t["thr_deg_pct"] = 100 * (b["throughput"] - t["throughput"]) / b["throughput"]
        t["energy_inc_pct"] = 100 * (t["energy_J_step"] - b["energy_J_step"]) / b["energy_J_step"]
        t["energy_idle_sub_inc_pct"] = 100 * (t["energy_idle_sub_J_step"] - b["energy_idle_sub_J_step"]) / \
            b["energy_idle_sub_J_step"]
        t["A_median"] = (t["step_ms_median"] - b["step_ms_median"]) / d if d else None
        if t.get("exposed_comm_fast") is not None and b.get("exposed_comm_fast") is not None:
            # fast ranks' forward work is unchanged by the delay, so any forward-time increase is waiting
            # (DDP broadcast_buffers=True broadcasts BN buffers from rank 0 along the ring before every forward)
            t["wait_fwd_inc_ms"] = t["fwd_fast"] - b["fwd_fast"]
            t["wait_bwd_inc_ms"] = t["exposed_comm_fast"] - b["exposed_comm_fast"]
            t["comm_wait_inc_ms"] = t["wait_fwd_inc_ms"] + t["wait_bwd_inc_ms"]
        if t.get("overlap_ratio_fast") is not None and b.get("overlap_ratio_fast"):
            t["overlap_deg_pct"] = 100 * (b["overlap_ratio_fast"] - t["overlap_ratio_fast"]) / b["overlap_ratio_fast"]
            t["comm_wait_inc_prof_ms"] = t["exposed_comm_prof_fast"] - b["exposed_comm_prof_fast"]


def timing_anomalies(conds, tol=1.08):
    """Residual-contamination detector independent of NVML process lists: forward compute of ranks that never
    wait in forward (all ViT ranks; ResNet rank 0 = BN-broadcast root) is fixed work and should be constant."""
    ref = {}
    for m in {r["model"] for r in conds}:
        v = [np.mean(k["steps"]["fwd"]) for r in conds if r["model"] == m for k in r["ranks"]
             if m == "vit_b_16" or k["rank"] == 0]
        ref[m] = float(np.median(v))
    bad = []
    for r in conds:
        for k in r["ranks"]:
            if (r["model"] == "vit_b_16" or k["rank"] == 0) and np.mean(k["steps"]["fwd"]) > tol * ref[r["model"]]:
                bad.append(dict(launch_id=r["launch_id"], attempt=r.get("attempt"), cond_index=r["cond_index"],
                                rank=k["rank"], fwd=float(np.mean(k["steps"]["fwd"])), ref=ref[r["model"]]))
    return ref, bad


def completeness(stage, conds):
    """Planned (launch, bucket, delay) cells vs. cells present in the clean data used for analysis."""
    from hddp import run
    if stage not in ("final", "pilot"):
        return None
    have = {(r["launch_id"], r["bucket"], r["delay_req"]) for r in conds}
    missing = [f"{L['launch_id']} b={b} d={d}" for L in run.plan(stage, run.settings())
               for b, d in L["conditions"] if (L["launch_id"], b, d) not in have]
    planned = sum(len(L["conditions"]) for L in run.plan(stage, run.settings()))
    return dict(planned_conditions=planned, clean_conditions=len(have), missing=missing)


def planned_launches(stage):
    """{launch_id: number of planned conditions} for stages with a fixed plan, else None."""
    from hddp import run
    if stage not in ("final", "pilot"):
        return None
    return {L["launch_id"]: len(L["conditions"]) for L in run.plan(stage, run.settings())}


def attempt_verdicts(rows, flags):
    """One entry per launch attempt: its recorded status(es) and whether the analysis accepted it.

    The status written by the orchestrator at run time is not the final word: an attempt that finished 'ok' can
    still be rejected by the analysis (foreign process, forward-time anomaly, cross-trial deviation)."""
    att = lambda r: (r["launch_id"], r.get("attempt"))
    recorded, n_cond = defaultdict(list), Counter()
    for r in rows:
        if r["type"] == "launch":
            recorded[att(r)].append(r["status"])
        elif r["type"] == "cond":
            n_cond[att(r)] += 1
    out = []
    for a in sorted(set(recorded) | set(n_cond), key=lambda a: (a[0], str(a[1]))):
        reasons = [f"recorded status {s}" for s in dict.fromkeys(recorded[a]) if s != "ok"]
        if not recorded[a]:
            reasons.append("no launch record (attempt interrupted)")
        reasons += [why for key, why in (("foreign", "foreign GPU process"), ("anomalous", "forward-time anomaly"),
                                         ("baseline_outliers", "cross-trial deviation")) if a in flags[key]]
        accepted = a in flags["good"]
        out.append(dict(launch_id=a[0], attempt=a[1], recorded_status=list(dict.fromkeys(recorded[a])),
                        conditions_recorded=n_cond[a], verdict="accepted" if accepted else "excluded",
                        reasons=[] if accepted else reasons))
    return out


def launch_outcomes(stage, rows, used, flags):
    """Logical-launch status derived from the clean data that enters the statistics.

    A logical launch is 'clean' only if its accepted attempt delivered every planned condition, and 'incomplete'
    otherwise (including when no attempt was accepted). Attempt-level outcomes are reported separately, so a
    rejected attempt with a recorded 'ok' status can no longer make its logical launch look clean."""
    planned = planned_launches(stage)
    attempts = attempt_verdicts(rows, flags)
    clean = Counter(r["launch_id"] for r in used)
    lids = list(planned) if planned else sorted({a["launch_id"] for a in attempts})
    status, incomplete = Counter(), []
    for lid in lids:
        n_plan = planned[lid] if planned else max((a["conditions_recorded"] for a in attempts
                                                    if a["launch_id"] == lid), default=0)
        if clean[lid] == n_plan and n_plan > 0:
            status["clean"] += 1
            continue
        status["incomplete"] += 1
        incomplete.append(dict(launch_id=lid, planned_conditions=n_plan, clean_conditions=clean[lid],
                               attempts=[a for a in attempts if a["launch_id"] == lid]))
    excluded = [a for a in attempts if a["verdict"] == "excluded"]
    return dict(
        launch_status=dict(status),
        launches=dict(planned=len(lids), clean=status["clean"], incomplete=status["incomplete"]),
        incomplete_launches=incomplete,
        attempts=dict(total=len(attempts), accepted=len(attempts) - len(excluded), excluded=len(excluded),
                      recorded_status=dict(Counter(s for a in attempts for s in a["recorded_status"])),
                      excluded_by_reason=dict(Counter(r for a in excluded for r in a["reasons"])),
                      excluded_attempts=excluded))


def ci95(v):
    v = [x for x in v if x is not None]
    n = len(v)
    if n == 0:
        return None, None, 0
    m = float(np.mean(v))
    if n < 2:
        return m, None, n
    return m, float(stats.t.ppf(0.975, n - 1) * np.std(v, ddof=1) / np.sqrt(n)), n


METRICS = ["step_ms", "step_ms_wall", "throughput", "delay_meas_ms", "fwd_fast", "bwd_fast", "opt_fast",
           "exposed_comm_fast", "fwd_strag", "bwd_strag", "exposed_comm_strag", "energy_J_step",
           "energy_idle_sub_J_step", "images_per_J", "power_W_fast", "power_W_strag", "sm_mhz_fast", "sm_mhz_strag",
           "util_fast", "temp_max", "overlap_ratio_fast", "overlap_ratio_strag", "comm_ms_fast",
           "exposed_comm_prof_fast", "A", "thr_deg_pct", "energy_inc_pct", "energy_idle_sub_inc_pct",
           "comm_wait_inc_ms", "wait_fwd_inc_ms", "wait_bwd_inc_ms", "overlap_deg_pct", "comm_wait_inc_prof_ms",
           "slack_strag_ms", "A_slack_model", "A_median", "step_ms_median", "compute_ms_strag", "compute_ms_fast",
           "fwd_r0", "fwd_r1", "fwd_r2", "fwd_r3", "last_ready_r0", "last_ready_r1", "last_ready_r2", "last_ready_r3"]


def aggregate(trials):
    groups = defaultdict(list)
    for t in trials:
        groups[tuple(t[k] for k in KEY)].append(t)
    out = []
    for key, ts in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = dict(zip(KEY, key))
        row["n_trials"] = len(ts)
        for mtr in METRICS:
            m, h, n = ci95([t.get(mtr) for t in ts])
            if n:
                row[f"{mtr}_mean"], row[f"{mtr}_ci95"], row[f"{mtr}_n"] = m, h, n
        row["throttle"] = "|".join(sorted({x for t in ts for x in t["throttle"].split("|") if x}))
        row["foreign_procs"] = sum(t["foreign_procs"] for t in ts)
        out.append(row)
    return out


def write_csv(path, rows, sig=None):
    """Floats are rounded to 5 decimals, or to `sig` significant digits (used for tables of p-values)."""
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    fmt = (lambda v: round(v, 5)) if sig is None else (lambda v: float(f"{v:.{sig}g}"))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (fmt(v) if isinstance(v, float) else v) for k, v in r.items()})


def sel(trials, **kw):
    return [t for t in trials if all(t.get(k) == v for k, v in kw.items())]


def h1(trials):
    """A(delta) vs 1: one-sided t-tests across trials (A>1 = amplification, A<1 = partial absorption)."""
    res = []
    for (s, m, a, b) in sorted({(t["setting"], t["model"], t["algo"], t["bucket"]) for t in trials}):
        for d in sorted({t["delay_req"] for t in trials if t["delay_req"] > 0}):
            A = [t["A"] for t in sel(trials, setting=s, model=m, algo=a, bucket=b, delay_req=d)
                 if t.get("A") is not None]
            if len(A) < 2:
                continue
            mean, h, n = ci95(A)
            p_gt = stats.ttest_1samp(A, 1.0, alternative="greater").pvalue
            p_lt = stats.ttest_1samp(A, 1.0, alternative="less").pvalue
            verdict = "A>1 (amplification)" if p_gt < 0.05 else ("A<1 (partial absorption)" if p_lt < 0.05
                                                                else "A not distinguishable from 1")
            res.append(dict(setting=s, model=m, algo=a, bucket=b, delay_req=d, A_mean=mean, A_ci95=h, n=n,
                            p_A_gt_1=p_gt, p_A_lt_1=p_lt, verdict=verdict))
    return res


def holm(pvals):
    """Holm (1979) step-down adjusted p-values. None entries are left out of the family and stay None."""
    idx = sorted((i for i, p in enumerate(pvals) if p is not None), key=lambda i: pvals[i])
    adj, running = [None] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (len(idx) - rank) * pvals[i]))
        adj[i] = running
    return adj


def rm_anova(Y):
    """One-way repeated-measures ANOVA. Y: subjects (launches) x levels (bucket sizes), complete cases.

    Returns F, degrees of freedom, the sphericity-assumed p, the Greenhouse-Geisser epsilon and corrected p,
    partial eta^2 = SS_b / (SS_b + SS_err) and generalized eta^2 = SS_b / (SS_b + SS_subj + SS_err)
    (Olejnik & Algina 2003; Bakeman 2005), which is comparable across within- and between-subject designs."""
    Y = np.asarray(Y, dtype=float)
    n, k = Y.shape
    gm = Y.mean()
    ss_b = n * float(((Y.mean(axis=0) - gm) ** 2).sum())
    ss_s = k * float(((Y.mean(axis=1) - gm) ** 2).sum())
    ss_err = float(((Y - gm) ** 2).sum()) - ss_b - ss_s
    df1, df2 = k - 1, (k - 1) * (n - 1)
    F = (ss_b / df1) / (ss_err / df2) if ss_err > 0 else float("inf")
    # Greenhouse-Geisser epsilon from the double-centred covariance matrix of the k levels. With n - 1 < k - 1
    # the covariance is rank deficient and sphericity cannot be tested; the GG correction is then a
    # conservative safeguard rather than an estimate with known accuracy.
    C = np.eye(k) - 1.0 / k
    D = C @ np.cov(Y, rowvar=False) @ C
    eps = float(np.trace(D) ** 2 / ((k - 1) * np.sum(D * D))) if np.sum(D * D) > 0 else 1.0
    eps = min(1.0, max(1.0 / (k - 1), eps))
    return dict(F=F, df1=df1, df2=df2, p=float(stats.f.sf(F, df1, df2)), gg_epsilon=eps,
                p_gg=float(stats.f.sf(F, eps * df1, eps * df2)),
                partial_eta2=ss_b / (ss_b + ss_err) if ss_b + ss_err > 0 else None,
                generalized_eta2=ss_b / (ss_b + ss_s + ss_err) if ss_b + ss_s + ss_err > 0 else None)


H2_METRICS = ("A", "thr_deg_pct", "energy_inc_pct")


def h2_cell(trials, buckets, mtr):
    """Bucket effect on one metric in one (setting, model, algorithm, delay) cell.

    Every bucket condition of a launch is measured inside that launch and paired with that launch's own
    delay-0 baseline, so the launch is the subject and bucket size is a within-subject factor. Only launches
    with a value for every bucket enter the repeated-measures tests (complete cases); the others are counted."""
    by_launch = defaultdict(dict)
    for t in trials:
        if t.get(mtr) is not None:
            assert t["bucket"] not in by_launch[t["launch_id"]], "two clean values for one launch/bucket"
            by_launch[t["launch_id"]][t["bucket"]] = t[mtr]
    complete = sorted(lid for lid, v in by_launch.items() if all(b in v for b in buckets))
    groups = [[v[b] for v in by_launch.values() if b in v] for b in buckets]
    out = dict(n_launches=len(complete), n_launches_incomplete_excluded=len(by_launch) - len(complete),
               by_bucket={b: float(np.mean(g)) if g else None for b, g in zip(buckets, groups)})
    if len(complete) < 2:
        return out
    Y = np.array([[by_launch[lid][b] for b in buckets] for lid in complete])
    means = Y.mean(axis=0)
    out["range"] = float(means.max() - means.min())
    out["rm_anova"] = rm_anova(Y)
    chi2, p_f = stats.friedmanchisquare(*Y.T)
    out["friedman"] = dict(chi2=float(chi2), p=float(p_f),
                           kendalls_w=float(chi2 / (len(complete) * (len(buckets) - 1))))
    hi, lo = int(np.argmax(means)), int(np.argmin(means))
    diff = Y[:, hi] - Y[:, lo]
    m_, h_, n_ = ci95(list(diff))
    out["contrast_max_vs_min_bucket"] = dict(
        high_bucket=buckets[hi], low_bucket=buckets[lo], mean_paired_diff=m_, ci95=h_, n=n_,
        cohens_dz=float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else None,
        note="post hoc (buckets chosen by their means); descriptive, not a confirmatory test")
    if all(len(g) >= 2 for g in groups):
        F, p = stats.f_oneway(*groups)
        out["secondary_independent_anova"] = dict(
            F=float(F), p=float(p), note="ignores the pairing of buckets within a launch; kept only for "
                                         "comparison with the earlier analysis")
    return out


def h2(trials):
    """Does bucket size change sensitivity to the straggler? See docs/methodology.md section 10.

    Primary test: repeated-measures ANOVA (launch = subject, bucket = within factor) with the Greenhouse-Geisser
    correction, Holm-adjusted across all tested (setting, model, algorithm, delay) cells of the same metric.
    Sensitivity analysis: Friedman test (Holm-adjusted in the same way). Secondary, for comparison only:
    independent-groups one-way ANOVA (Holm-adjusted in the same way)."""
    res = []
    for (s, m, a) in sorted({(t["setting"], t["model"], t["algo"]) for t in trials}):
        buckets = sorted({t["bucket"] for t in sel(trials, setting=s, model=m, algo=a)})
        if len(buckets) < 3:
            continue
        for d in sorted({t["delay_req"] for t in sel(trials, setting=s, model=m, algo=a) if t["delay_req"] > 0}):
            ts = sel(trials, setting=s, model=m, algo=a, delay_req=d)
            row = dict(setting=s, model=m, algo=a, delay_req=d, buckets=buckets)
            row.update({mtr: h2_cell(ts, buckets, mtr) for mtr in H2_METRICS})
            res.append(row)
    for mtr in H2_METRICS:
        cells = [r[mtr] for r in res]
        for test, key in (("rm_anova", "p_gg"), ("rm_anova", "p"), ("friedman", "p"),
                          ("secondary_independent_anova", "p")):
            adj = holm([c[test][key] if test in c else None for c in cells])
            for c, pa in zip(cells, adj):
                if test in c:
                    c[test][f"{key}_holm"] = pa
    return res


def h2_overview(rows, alpha=0.05):
    """Number of cells with a significant bucket effect under each analysis (Holm-adjusted within metric)."""
    out = {"alpha": alpha, "n_cells": len(rows),
           "correction": "Holm step-down across all cells of the same metric",
           "primary": "rm_anova p_gg_holm (repeated measures, launch = subject, Greenhouse-Geisser corrected)"}
    for mtr in H2_METRICS:
        def cnt(test, key, mtr=mtr):
            return sum(1 for r in rows if r[mtr].get(test, {}).get(key) is not None and r[mtr][test][key] < alpha)
        out[mtr] = dict(
            rm_anova_gg_holm=cnt("rm_anova", "p_gg_holm"), rm_anova_sphericity_assumed_holm=cnt("rm_anova", "p_holm"),
            rm_anova_gg_unadjusted=cnt("rm_anova", "p_gg"), friedman_holm=cnt("friedman", "p_holm"),
            secondary_independent_anova_unadjusted=cnt("secondary_independent_anova", "p"),
            secondary_independent_anova_holm=cnt("secondary_independent_anova", "p_holm"),
            n_tested=sum(1 for r in rows if "rm_anova" in r[mtr]))
    return out


def h2_table(rows):
    """Flat table of the H2 results (one row per cell and metric) for results/processed/<stage>_h2.csv."""
    out = []
    for r in rows:
        for mtr in H2_METRICS:
            c = r[mtr]
            if "rm_anova" not in c:
                continue
            rm, fr, ct = c["rm_anova"], c["friedman"], c["contrast_max_vs_min_bucket"]
            ind = c.get("secondary_independent_anova", {})
            out.append(dict(
                setting=r["setting"], model=r["model"], algo=r["algo"], delay_req=r["delay_req"], metric=mtr,
                n_launches=c["n_launches"], n_launches_incomplete_excluded=c["n_launches_incomplete_excluded"],
                range_of_bucket_means=c["range"], rm_F=rm["F"], rm_df1=rm["df1"], rm_df2=rm["df2"],
                rm_gg_epsilon=rm["gg_epsilon"], rm_p_gg=rm["p_gg"], rm_p_gg_holm=rm["p_gg_holm"], rm_p=rm["p"],
                rm_p_holm=rm["p_holm"], partial_eta2=rm["partial_eta2"], generalized_eta2=rm["generalized_eta2"],
                friedman_chi2=fr["chi2"], friedman_p=fr["p"], friedman_p_holm=fr["p_holm"],
                kendalls_w=fr["kendalls_w"], contrast_high_bucket=ct["high_bucket"],
                contrast_low_bucket=ct["low_bucket"], contrast_mean_diff=ct["mean_paired_diff"],
                contrast_ci95=ct["ci95"], contrast_cohens_dz=ct["cohens_dz"],
                secondary_independent_anova_F=ind.get("F"), secondary_independent_anova_p=ind.get("p"),
                secondary_independent_anova_p_holm=ind.get("p_holm"),
                **{f"mean_b{b:g}": v for b, v in c["by_bucket"].items()}))
    return out


def h3(trials, metric="throughput"):
    """Homogeneous-optimal (algo, bucket) vs. the optimum at each delay; Welch t-test between the two."""
    res = []
    for (s, m) in sorted({(t["setting"], t["model"]) for t in trials}):
        ts = sel(trials, setting=s, model=m)
        cfgs = sorted({(t["algo"], t["bucket"]) for t in ts})
        if len(cfgs) < 2:
            continue
        def vals(c, d, ts=ts):
            return [t[metric] for t in ts if (t["algo"], t["bucket"]) == c and t["delay_req"] == d]

        def mean_at(d, cfgs=cfgs, vals=vals):  # configurations with >= 2 trials only (a Welch test needs 2)
            return {c: float(np.mean(vals(c, d))) for c in cfgs if len(vals(c, d)) >= 2}
        m0 = mean_at(0)
        if not m0:
            continue
        homo = max(m0, key=m0.get)
        for d in sorted({t["delay_req"] for t in ts}):
            md = mean_at(d)
            if homo not in md or len(md) < len(m0):  # compare only where every configuration was measured
                continue
            best = max(md, key=md.get)
            p = float(stats.ttest_ind(vals(best, d), vals(homo, d), equal_var=False).pvalue) if best != homo else None
            ranking = sorted(md, key=md.get, reverse=True)
            res.append(dict(setting=s, model=m, metric=metric, delay_req=d, n_configs=len(md),
                            homo_opt=f"{homo[0]}/{homo[1]}MB", best_at_delay=f"{best[0]}/{best[1]}MB",
                            homo_value=md[homo], best_value=md[best],
                            regret_pct=100 * (md[best] - md[homo]) / md[best], homo_rank=ranking.index(homo) + 1,
                            welch_p_best_vs_homo=p, differs=best != homo))
    return res


def slack_model(trials):
    """How well does A ~= max(0, delta - slack)/delta (slack = straggler's exposed wait at delay 0) explain A?"""
    out = {}
    for model in sorted({t["model"] for t in trials}):
        pairs = [(t["A"], t["A_slack_model"]) for t in trials
                 if t["model"] == model and t.get("A") is not None and t.get("A_slack_model") is not None]
        if len(pairs) > 2:
            a, p = np.array(pairs).T
            out[model] = dict(n=len(pairs), rmse=float(np.sqrt(np.mean((a - p) ** 2))),
                              pearson_r=float(stats.pearsonr(a, p)[0]) if np.std(p) > 0 else None,
                              mean_A=float(a.mean()), mean_pred=float(p.mean()))
    return out


def robustness(trials):
    """A from mean vs median step period; Auto vs Ring where NCCL verifiably selected the same algorithm."""
    pairs = [(t["A"], t["A_median"]) for t in trials if t.get("A") is not None and t.get("A_median") is not None
             and t["delay_req"] >= 10]
    a, m = np.array(pairs).T if pairs else (np.array([]), np.array([]))
    out = {"A_mean_vs_median_step_delay_ge10": dict(n=len(pairs), max_abs_diff=float(np.max(np.abs(a - m)))
                                                       if pairs else None,
                                                       mean_abs_diff=float(np.mean(np.abs(a - m))) if pairs else None)}
    same = []
    ver = {(v["setting"], v["model"], v["algo_requested"], float(v["bucket"])): v["allreduce_algos"]
           for v in nccl_verification()}
    for (s_, m_, b) in sorted({(t["setting"], t["model"], t["bucket"]) for t in trials if t["algo"] == "Ring"}):
        if ver.get((s_, m_, "Auto", float(b))) != "Ring":
            continue
        for d in sorted({t["delay_req"] for t in trials if t["algo"] == "Ring"}):
            au = [t["step_ms"] for t in sel(trials, setting=s_, model=m_, bucket=b, delay_req=d, algo="Auto")]
            ri = [t["step_ms"] for t in sel(trials, setting=s_, model=m_, bucket=b, delay_req=d, algo="Ring")]
            if len(au) > 1 and len(ri) > 1:
                same.append(dict(setting=s_, model=m_, bucket=b, delay_req=d, auto=float(np.mean(au)),
                                 ring=float(np.mean(ri)), diff_pct=100 * (np.mean(au) - np.mean(ri)) / np.mean(ri),
                                 welch_p=float(stats.ttest_ind(au, ri, equal_var=False).pvalue)))
    out["auto_vs_ring_where_nccl_chose_ring"] = dict(
        n=len(same), max_abs_diff_pct=max((abs(x["diff_pct"]) for x in same), default=None),
        n_p_lt_0_05=sum(1 for x in same if x["welch_p"] < 0.05), rows=same)
    return out


def calibration(trials):
    out = []
    for (s, m, d) in sorted({(t["setting"], t["model"], t["delay_req"]) for t in trials if t["delay_req"] > 0}):
        ts = sel(trials, setting=s, model=m, delay_req=d)
        meas = [t["delay_meas_ms"] for t in ts]
        out.append(dict(setting=s, model=m, delay_req=d, delay_meas_mean=float(np.mean(meas)),
                        delay_meas_min=float(np.min(meas)), delay_meas_max=float(np.max(meas)),
                        within_step_sd_mean=_m([t["delay_meas_sd"] for t in ts]),
                        rel_err_pct=100 * (float(np.mean(meas)) - d) / d, n=len(ts)))
    return out


def instrumentation_overhead(trials):
    out = []
    for (s, m, a, b) in sorted({(t["setting"], t["model"], t["algo"], t["bucket"])
                                for t in trials if t["instrument"] == 0}):
        on = [t["step_ms"] for t in sel(trials, setting=s, model=m, algo=a, bucket=b, delay_req=0, instrument=1)]
        off = [t["step_ms"] for t in sel(trials, setting=s, model=m, algo=a, bucket=b, delay_req=0, instrument=0)]
        if on and off:
            out.append(dict(setting=s, model=m, algo=a, bucket=b, step_ms_instrumented=float(np.mean(on)),
                            step_ms_plain=float(np.mean(off)),
                            overhead_pct=100 * (np.mean(on) - np.mean(off)) / np.mean(off), n_on=len(on),
                            n_off=len(off),
                            welch_p=float(stats.ttest_ind(on, off, equal_var=False).pvalue)
                            if len(on) > 1 and len(off) > 1 else None))
    return out


def nccl_verification():
    rows = []
    p = RAW / "verify.jsonl"
    if p.exists():
        with open(p) as f:
            records = [json.loads(line) for line in f]
        for r in records:
            if r["type"] != "nccl_verify":
                continue
            ar = {k.split(":", 1)[1]: v for k, v in r["selected"].items() if k.startswith("AllReduce:")}
            algos = sorted({k.split("/")[0] for k in ar})
            req = r["algo_requested"]
            honored = (algos == [req]) if req != "Auto" else None
            rows.append(dict(setting=r["setting"], model=r["model"], algo_requested=req, bucket=r["bucket"],
                             allreduce_selected=json.dumps(ar), allreduce_algos="|".join(algos),
                             honored=honored, env_lines="|".join(r["env_lines"]),
                             transports="|".join(sorted(r["transports"]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="final")
    ap.add_argument("--outdir", default=str(OUT), help="where to write the processed tables")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    conds, launches, rows = load(a.stage)
    trials = [trial_row(r) for r in conds]
    add_paired(trials)
    main_trials = [t for t in trials if t["instrument"] == 1 and t["tag"] == ""]
    write_csv(out / f"{a.stage}_trials.csv", trials)
    write_csv(out / f"{a.stage}_conditions.csv", aggregate(trials))
    ver = nccl_verification()
    if ver:
        write_csv(out / "nccl_verification.csv", ver)
    # kernel names seen in profiled production windows (independent check of the algorithm actually run)
    kern = defaultdict(set)
    for t in trials:
        if t.get("nccl_kernels"):
            kern[f"{t['setting']}|{t['model']}|{t['algo']}"].update(
                n for n in t["nccl_kernels"].split("|") if "AllReduce" in n)
    outcomes = launch_outcomes(a.stage, rows, conds, load.flags)
    fwd_ref, anomalies = timing_anomalies(conds)  # on the data actually used (should be empty)
    h2_rows = h2(main_trials)
    summary = {
        "stage": a.stage,
        "n_trials": len(trials),
        "launch_status": outcomes["launch_status"],
        "launches": outcomes["launches"],
        "completeness": completeness(a.stage, conds),
        "incomplete_launches": outcomes["incomplete_launches"],
        "attempts": outcomes["attempts"],
        "data_quality": {
            "conditions_excluded_contaminated_or_failed_attempts": load.excluded,
            "forward_time_reference_ms": fwd_ref,
            "forward_time_anomalies_in_used_data": anomalies,
            "contaminated_attempts": sorted({l["launch_id"] for l in launches if l["status"] == "contaminated"}),
            "trials_with_foreign_gpu_procs": sum(1 for t in trials if t["foreign_procs"]),
            "throttle_reasons_seen": dict(Counter(x for t in trials for x in t["throttle"].split("|") if x)),
            "energy_sampled_vs_counter_pct": {
                "mean": _m([t["energy_sampled_vs_counter_pct"] for t in trials]),
                "max_abs": max((abs(t["energy_sampled_vs_counter_pct"]) for t in trials
                                if t["energy_sampled_vs_counter_pct"] is not None), default=None)},
            "max_rank_period_spread_ms": max((t["period_rank_spread_ms"] for t in trials), default=None),
            "temp_max_c": max((t["temp_max"] for t in trials if t["temp_max"] is not None), default=None),
        },
        "delay_calibration": calibration(main_trials),
        "robustness": robustness(main_trials),
        "instrumentation_overhead": instrumentation_overhead(trials),
        "nccl_verification_summary": {
            "n": len(ver), "forced_honored": sum(1 for v in ver if v["honored"]),
            "forced_not_honored": [v for v in ver if v["honored"] is False],
            "auto_selection": sorted({(v["setting"], v["model"], v["bucket"], v["allreduce_algos"])
                                      for v in ver if v["algo_requested"] == "Auto"})},
        "nccl_kernels_in_profiled_runs": {k: sorted(v) for k, v in kern.items()},
        "H1": h1(main_trials),
        "H1_slack_model": slack_model(main_trials),
        "straggler_placement": [dict(model=c[0], delay_req=c[1], **{
            f"A_{lab}": ci95([t["A"] for t in trials if t["model"] == c[0] and t["delay_req"] == c[1]
                              and t["setting"] == "4gpu" and t["algo"] == "Auto" and t["bucket"] == 25
                              and t["tag"] == tag and t.get("A") is not None])
            for lab, tag in (("rank3", ""), ("rank0", "strag0"))})
            for c in sorted({(t["model"], t["delay_req"]) for t in trials if t["tag"] == "strag0" and t["delay_req"]})],
        "H2_overview": h2_overview(h2_rows),
        "H2": h2_rows,
        "H3_throughput": h3(main_trials, "throughput"),
        "H3_images_per_J": h3(main_trials, "images_per_J"),
    }
    (out / f"{a.stage}_summary.json").write_text(json.dumps(summary, indent=1, default=str), newline="\n")
    if h2_rows:
        write_csv(out / f"{a.stage}_h2.csv", h2_table(h2_rows), sig=6)
    c = summary["completeness"] or {}
    print(json.dumps({k: summary[k] for k in ("n_trials", "launch_status", "data_quality")}, indent=1))
    print(f"completeness: {c.get('clean_conditions')}/{c.get('planned_conditions')} clean, "
          f"{len(c.get('missing', []))} missing")


if __name__ == "__main__":
    main()
