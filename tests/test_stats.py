"""Statistical helpers: reference values and mathematical invariants."""
import numpy as np
import pytest
from scipy import stats

from hddp.analyze import ci95, h2_cell, holm, rm_anova

# 5 subjects x 4 levels; reference values computed independently with pingouin 0.7.0 (rm_anova, correction=True)
Y = np.array([[10, 12, 13, 15], [8, 9, 12, 11], [11, 13, 14, 17], [9, 9, 10, 13], [12, 15, 15, 16]], float)


def test_rm_anova_reference_values():
    r = rm_anova(Y)
    assert (r["df1"], r["df2"]) == (3, 12)
    assert r["F"] == pytest.approx(21.89473684210524, rel=1e-12)
    assert r["p"] == pytest.approx(3.710898290397979e-05, rel=1e-9)
    assert r["gg_epsilon"] == pytest.approx(0.7400574005740058, rel=1e-12)
    assert r["p_gg"] == pytest.approx(0.0003106095160029434, rel=1e-9)
    assert r["generalized_eta2"] == pytest.approx(0.4088050314465409, rel=1e-12)
    assert r["partial_eta2"] == pytest.approx(0.8455284552845527, rel=1e-12)


def test_rm_anova_ignores_subject_offsets():
    """Between-launch differences must not affect the within-launch bucket test."""
    shifted = Y + np.array([[100.0], [-40.0], [3.0], [0.0], [7.5]])
    a, b = rm_anova(Y), rm_anova(shifted)
    for k in ("F", "p", "gg_epsilon", "p_gg", "partial_eta2"):
        assert a[k] == pytest.approx(b[k], rel=1e-9)


def test_rm_anova_two_levels_equals_paired_t():
    two = Y[:, :2]
    t, p = stats.ttest_rel(two[:, 0], two[:, 1])
    r = rm_anova(two)
    assert r["F"] == pytest.approx(t ** 2, rel=1e-12)
    assert r["p"] == pytest.approx(p, rel=1e-9)
    assert r["gg_epsilon"] == 1.0 and r["p_gg"] == pytest.approx(r["p"], rel=1e-12)


def test_gg_correction_is_conservative():
    r = rm_anova(Y)
    assert 1 / (Y.shape[1] - 1) <= r["gg_epsilon"] <= 1
    assert r["p_gg"] >= r["p"]


def test_holm_reference_and_properties():
    p = [0.01, 0.04, 0.03, 0.005, 0.5]
    assert holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.025, 0.5])  # statsmodels multipletests(method="holm")
    assert holm([0.2, None, 0.01]) == [0.2, None, 0.02]
    adj = holm(list(np.linspace(0.001, 0.9, 16)))
    assert all(a >= q for a, q in zip(adj, np.linspace(0.001, 0.9, 16))) and max(adj) <= 1


def test_ci95_matches_t_interval():
    m, h, n = ci95([1.0, 2.0, 4.0, None])
    assert (m, n) == (pytest.approx(7 / 3), 3)
    assert h == pytest.approx(stats.t.ppf(0.975, 2) * np.std([1, 2, 4], ddof=1) / np.sqrt(3))


def test_h2_cell_uses_only_complete_launches():
    trials = [dict(launch_id=f"L{i}", bucket=b, A=float(Y[i, j]))
              for i in range(5) for j, b in enumerate((1, 4, 16, 25))]
    trials = [t for t in trials if not (t["launch_id"] == "L4" and t["bucket"] == 16)]  # one incomplete launch
    c = h2_cell(trials, [1, 4, 16, 25], "A")
    assert c["n_launches"] == 4 and c["n_launches_incomplete_excluded"] == 1
    assert c["rm_anova"]["F"] == pytest.approx(rm_anova(Y[:4])["F"])
    ct = c["contrast_max_vs_min_bucket"]
    assert (ct["high_bucket"], ct["low_bucket"]) == (25, 1)


def test_h2_cell_rejects_duplicate_launch_bucket():
    trials = [dict(launch_id="L0", bucket=1, A=1.0), dict(launch_id="L0", bucket=1, A=2.0)]
    with pytest.raises(AssertionError):
        h2_cell(trials, [1, 4, 16], "A")
