"""Orchestrator safety: side-effect-free dry runs and no orphaned workers when start-up fails."""
import hashlib
import signal
import sys
from pathlib import Path

import pytest

from hddp import nvml, run

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "results").rglob("*") if p.is_file()}


@pytest.fixture
def keep_signals():
    names = [n for n in ("SIGTERM", "SIGHUP", "SIGINT") if hasattr(signal, n)]
    saved = {n: signal.getsignal(getattr(signal, n)) for n in names}
    yield
    for n, h in saved.items():
        signal.signal(getattr(signal, n), h)


def test_throttle_dry_run_touches_nothing(monkeypatch, capsys, keep_signals):
    def forbidden(*a, **k):
        raise AssertionError("a dry run must not call NVML")
    for fn in ("init", "can_set_power_limit", "set_power_limit_w", "set_locked_clocks"):
        monkeypatch.setattr(nvml, fn, forbidden)
    monkeypatch.setattr(nvml, "ThrottleGuard", forbidden)
    monkeypatch.setattr(run, "launch", forbidden)
    before = snapshot()
    monkeypatch.setattr(sys, "argv", ["hddp.run", "throttle", "--dry"])
    run.main()
    out = capsys.readouterr().out
    assert "dry run" in out and out.count("throttle|4gpu|") == 4
    assert snapshot() == before


def test_matrix_dry_run_touches_nothing(monkeypatch, capsys, keep_signals):
    monkeypatch.setattr(run, "gpu_state", lambda: pytest.fail("a dry run must not query the GPUs"))
    monkeypatch.setattr(run, "launch", lambda *a, **k: pytest.fail("a dry run must not launch"))
    before = snapshot()
    monkeypatch.setattr(sys, "argv", ["hddp.run", "final", "--dry"])
    run.main()
    # only the launch without any clean attempt (final|4gpu|resnet50|Ring|t1) would be re-run
    assert "final: 60 launches, 1 to run, 18 conditions" in capsys.readouterr().out
    assert snapshot() == before


class FakePopen:
    pid = 999_999_999
    returncode = None

    def __init__(self, *a, **k):
        pass

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None


def test_workers_killed_when_watch_fails_to_start(monkeypatch, tmp_path):
    killed = []

    class BrokenWatch:
        def __init__(self, *a, **k):
            raise RuntimeError("NVML init failed")

    monkeypatch.setattr(run.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(run, "Watch", BrokenWatch)
    monkeypatch.setattr(run, "_kill", lambda p, token: killed.append((p, token)))
    Lx = run.L("final", "4gpu", [0, 1, 2, 3], "resnet50", "Auto", 0, [(25, 0)])
    with open(tmp_path / "log", "w") as log, pytest.raises(RuntimeError, match="NVML init failed"):
        run.launch(Lx, tmp_path / "out.jsonl", log)
    assert len(killed) == 1 and isinstance(killed[0][0], FakePopen) and killed[0][1]
    assert run.CURRENT is None


def test_workers_killed_on_interrupt(monkeypatch, tmp_path):
    killed = []

    class InterruptedPopen(FakePopen):
        def wait(self, timeout=None):
            raise KeyboardInterrupt

    class IdleWatch:
        def __init__(self, *a, **k):
            self.events = []
            self.halt = type("E", (), {"set": lambda self: None})()

    monkeypatch.setattr(run.subprocess, "Popen", InterruptedPopen)
    monkeypatch.setattr(run, "Watch", IdleWatch)
    monkeypatch.setattr(run, "_kill", lambda p, token: killed.append(p))
    Lx = run.L("final", "4gpu", [0, 1, 2, 3], "resnet50", "Auto", 0, [(25, 0)])
    with open(tmp_path / "log", "w") as log, pytest.raises(KeyboardInterrupt):
        run.launch(Lx, tmp_path / "out.jsonl", log)
    assert len(killed) == 1 and run.CURRENT is None


def test_launch_log_has_no_private_paths(monkeypatch, tmp_path):
    class IdleWatch:
        def __init__(self, *a, **k):
            self.events = []
            self.halt = type("E", (), {"set": lambda self: None})()

    monkeypatch.setattr(run.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(run, "Watch", IdleWatch)
    monkeypatch.setattr(run, "_kill", lambda p, token: None)
    monkeypatch.setattr(run.sys, "executable", "/home/someone/miniforge3/bin/python")
    Lx = run.L("final", "4gpu", [0, 1, 2, 3], "resnet50", "Auto", 0, [(25, 0)])
    with open(tmp_path / "log", "w") as log:
        assert run.launch(Lx, "/home/someone/x/hetero-ddp-lab/results/raw/final.jsonl", log) == 0
    text = (tmp_path / "log").read_text()
    assert "someone" not in text and "<python-env>/bin/python" in text
