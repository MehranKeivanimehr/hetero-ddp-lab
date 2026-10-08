"""Re-run the analysis from results/raw into a temporary directory and compare with results/processed.

  python scripts/check_reproducibility.py [--stages final pilot]

CSV files are compared byte for byte and, only if the bytes differ, cell by cell (text equal, numbers within
a relative 1e-5 of the rounded value). The JSON summaries are compared value by value with a relative
tolerance of 1e-9 for floats, because different SciPy builds can differ in the last floating-point digit of
a p-value. Exit status 0 = reproduced, 1 = mismatch.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "results" / "processed"
REL_TOL = 1e-9


def json_diffs(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ ({sorted(set(a) ^ set(b))})"]
        return [d for k in a for d in json_diffs(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in json_diffs(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        if a == b or (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=REL_TOL, abs_tol=0.0):
            return []
        return [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def csv_diffs(new, old):
    """Cell-by-cell comparison, used only when the bytes differ. Rounded values may differ by one unit in the
    last written digit when a library build changes the last bit of the unrounded value."""
    with open(new, newline="") as fa, open(old, newline="") as fb:
        a, b = list(csv.reader(fa)), list(csv.reader(fb))
    if len(a) != len(b) or a[0] != b[0]:
        return [": header or row count differs"]
    out = []
    for i, (ra, rb) in enumerate(zip(a[1:], b[1:]), start=2):
        for col, x, y in zip(a[0], ra, rb):
            if x != y:
                try:
                    if math.isclose(float(x), float(y), rel_tol=1e-5, abs_tol=1e-5):
                        continue
                except ValueError:
                    pass
                out.append(f" line {i}, {col}: {x!r} != {y!r}")
    return out


def check(stages):
    problems = []
    with tempfile.TemporaryDirectory() as td:
        for stage in stages:
            subprocess.run([sys.executable, "-m", "hddp.analyze", "--stage", stage, "--outdir", td], cwd=ROOT,
                           check=True, capture_output=True, text=True)
        for new in sorted(Path(td).iterdir()):
            old = PROC / new.name
            if not old.exists():
                problems.append(f"{new.name}: produced by the analysis but not in results/processed")
            elif new.suffix == ".csv":
                if new.read_bytes() != old.read_bytes():
                    problems += [f"{new.name}{x}" for x in csv_diffs(new, old)[:20]]
            else:
                d = json_diffs(json.loads(new.read_text()), json.loads(old.read_text()))
                problems += [f"{new.name}{x}" for x in d[:20]]
            print(("MISMATCH " if any(p.startswith(new.name) for p in problems) else "ok       ") + new.name)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["final", "pilot"])
    problems = check(ap.parse_args().stages)
    for p in problems:
        print("  " + p)
    print("reproduced" if not problems else f"{len(problems)} mismatches")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
