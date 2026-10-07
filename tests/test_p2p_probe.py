"""P2P correctness probe: exact expected values, failure parsing, and which GPU sets are probed."""
import json
from pathlib import Path

import numpy as np
import pytest

from hddp import env

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("world", [2, 3, 4, 8])
def test_pattern_sum_is_exact_in_float32(world):
    n = 50_000
    total = np.zeros(n, dtype=np.float32)
    for r in range(world):
        total += env.p2p_pattern(r, n)
    assert np.array_equal(total, env.p2p_expected(world, n))
    assert env.p2p_expected(world).max() < 2 ** 24  # every partial sum is exactly representable


def test_pattern_detects_shifted_or_stale_data():
    good = env.p2p_pattern(0, 5000) + env.p2p_pattern(1, 5000)
    assert not np.array_equal(np.roll(good, 1), env.p2p_expected(2, 5000))
    assert not np.array_equal(env.p2p_pattern(0, 5000), env.p2p_expected(2, 5000))


def line(rank, world=2, mism=0, first=-1, cs=10.0, ref=10.0, s=0.5):
    return (f"P2PCHECK rank={rank} world={world} mismatches={mism} first_bad={first} checksum={cs:.1f} "
            f"expected_checksum={ref:.1f} first_allreduce_s={s:.2f}\n")


def test_parse_success():
    r = env.parse_p2p_output("noise\n" + line(0) + line(1, s=0.7), 2)
    assert r["correct"] and r["failure"] is None and r["ranks_reported"] == 2 and r["first_allreduce_s"] == 0.7


def test_parse_wrong_elements_on_one_rank():
    r = env.parse_p2p_output(line(0) + line(1, mism=4_999_999, first=1, cs=5.0), 2)
    assert not r["correct"] and r["mismatched_elements"] == 4_999_999
    assert "rank 1: 4999999 mismatched elements (first at index 1), checksum differs" in r["failure"]


def test_parse_checksum_only_mismatch_fails():
    assert not env.parse_p2p_output(line(0) + line(1, cs=9.0), 2)["correct"]


def test_parse_missing_rank_and_hang():
    assert env.parse_p2p_output(line(0), 2)["failure"] == "only ranks [0] of 2 reported"
    r = env.parse_p2p_output("timeout after 90 s", 2)
    assert not r["correct"] and r["ranks_reported"] == 0 and "no rank reported" in r["failure"]


def test_targets_cover_every_non_nvlink_pair_and_the_group():
    topo = json.loads((ROOT / "results" / "env" / "topology.json").read_text())
    link = {tuple(map(int, k.split("-"))): v for k, v in topo["matrix"].items()}
    targets = env.p2p_targets(topo["gpus"], link, topo["allowed_gpus"], topo["settings"])
    pairs = sorted(tuple(g) for g, kind, _ in targets if kind == "pair")
    assert pairs == [(0, 2), (0, 3), (1, 2), (1, 3)]  # all PCIe pairs among GPUs 0-3; NVLink pairs excluded
    assert [(g, kind) for g, kind, _ in targets if kind == "group"] == [([0, 1, 2, 3], "group")]
    assert all(cls != "nvlink" for _, kind, cls in targets if kind == "pair")


def test_default_p2p_probe_is_opt_in():
    import inspect
    src = inspect.getsource(env.main)
    assert 'levels = ([None] if args.test_default_p2p else []) + ["NVL"]' in src
