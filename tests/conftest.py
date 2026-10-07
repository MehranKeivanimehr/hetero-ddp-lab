import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def final_trials():
    """Trial rows of the final stage exactly as the analysis builds them (unrounded)."""
    from hddp import analyze
    conds, launches, rows = analyze.load("final")
    trials = [analyze.trial_row(r) for r in conds]
    analyze.add_paired(trials)
    return dict(conds=conds, launches=launches, rows=rows, trials=trials,
                main=[t for t in trials if t["instrument"] == 1 and t["tag"] == ""])
