"""Key invariants of the published data and of the analysis built on it."""
import json
from collections import Counter
from pathlib import Path

import pytest

from hddp import analyze

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((ROOT / "results" / "processed" / "final_summary.json").read_text())


def test_one_clean_attempt_per_launch(final_trials):
    attempts = {}
    for r in final_trials["conds"]:
        attempts.setdefault(r["launch_id"], set()).add(r.get("attempt"))
    assert all(len(a) == 1 for a in attempts.values())
    keys = Counter((t["launch_id"], t["bucket"], t["delay_req"]) for t in final_trials["trials"])
    assert max(keys.values()) == 1


def test_excluded_attempts_never_used(final_trials):
    flags = analyze.attempt_flags(final_trials["rows"])
    used = {(r["launch_id"], r.get("attempt")) for r in final_trials["conds"]}
    bad = flags["foreign"] | flags["anomalous"] | flags["baseline_outliers"]
    assert used <= flags["good"] and not used & bad
    status = {(l["launch_id"], l.get("attempt")): l["status"] for l in final_trials["launches"]}
    assert all(status.get(u) == "ok" for u in used)
    assert all(t["foreign_procs"] == 0 for t in final_trials["trials"])


def test_kept_attempts_within_consistency_tolerance():
    dq = SUMMARY["data_quality"]["conditions_excluded_contaminated_or_failed_attempts"]
    assert dq["kept_attempts_max_condition_dev_pct"] <= analyze.CONSISTENCY_TOL_PCT
    assert dq["kept_attempts_max_condition_dev_pct"] == pytest.approx(2.93, abs=0.005)
    assert SUMMARY["data_quality"]["forward_time_anomalies_in_used_data"] == []


def test_completeness_reports_the_missing_launch():
    c = SUMMARY["completeness"]
    assert (c["planned_conditions"], c["clean_conditions"]) == (810, 792)
    assert len(c["missing"]) == 18
    assert {m.split(" ")[0] for m in c["missing"]} == {"final|4gpu|resnet50|Ring|t1"}


def test_paired_metrics_use_same_launch_baseline(final_trials):
    base = {(t["launch_id"], t["bucket"]): t for t in final_trials["trials"] if t["delay_req"] == 0}
    n = 0
    for t in final_trials["trials"]:
        if t["delay_req"] == 0:
            assert "A" not in t
            continue
        b = base[(t["launch_id"], t["bucket"])]
        assert t["A"] == pytest.approx((t["step_ms"] - b["step_ms"]) / t["delay_meas_ms"], rel=1e-12)
        assert t["thr_deg_pct"] == pytest.approx(100 * (b["throughput"] - t["throughput"]) / b["throughput"])
        n += 1
    assert n == sum(1 for t in final_trials["trials"] if t["delay_req"] > 0)


def test_measured_delay_close_to_requested():
    for c in SUMMARY["delay_calibration"]:
        assert abs(c["rel_err_pct"]) < 0.8


def test_forced_algorithms_confirmed():
    v = SUMMARY["nccl_verification_summary"]
    assert v["forced_honored"] == 72 and v["forced_not_honored"] == []


def test_h2_design_and_corrections():
    rows, ov = SUMMARY["H2"], SUMMARY["H2_overview"]
    assert ov["n_cells"] == len(rows) == 16
    for r in rows:
        for mtr in analyze.H2_METRICS:
            c = r[mtr]
            expected_n = 4 if (r["model"], r["algo"]) == ("resnet50", "Ring") else 5
            assert c["n_launches"] == expected_n and c["n_launches_incomplete_excluded"] == 0
            rm = c["rm_anova"]
            assert 1 / (len(r["buckets"]) - 1) <= rm["gg_epsilon"] <= 1
            if rm["F"] > 2:  # with F near or below 1 the corrected p can legitimately be the smaller one
                assert rm["p_gg"] >= rm["p"] - 1e-15
            assert rm["p_gg_holm"] >= rm["p_gg"] - 1e-15 and rm["p_holm"] >= rm["p"] - 1e-15
            assert c["friedman"]["p_holm"] >= c["friedman"]["p"] - 1e-15
            assert 0 <= rm["generalized_eta2"] <= rm["partial_eta2"] <= 1
    for mtr in analyze.H2_METRICS:
        assert ov[mtr]["n_tested"] == 16
        assert ov[mtr]["rm_anova_gg_holm"] <= ov[mtr]["rm_anova_gg_unadjusted"]


def test_h2_recomputes_from_trials(final_trials):
    rows = analyze.h2(final_trials["main"])
    for new, old in zip(rows, SUMMARY["H2"]):
        assert new["A"]["rm_anova"]["F"] == pytest.approx(old["A"]["rm_anova"]["F"], rel=1e-9)
        assert new["A"]["rm_anova"]["p_gg_holm"] == pytest.approx(old["A"]["rm_anova"]["p_gg_holm"], rel=1e-9)
