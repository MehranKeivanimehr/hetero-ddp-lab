"""Repository hygiene: every data file is readable, Markdown links resolve, no file is too large for Git."""
import csv
import gzip
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = [p for p in ROOT.rglob("*") if p.is_file()
         and not {".git", "__pycache__", ".pytest_cache", ".venv", "venv"} & set(p.relative_to(ROOT).parts)]


def test_json_files_parse():
    for p in [p for p in FILES if p.suffix == ".json"]:
        json.loads(p.read_text(encoding="utf-8"))


def test_jsonl_files_parse_and_have_types():
    for p in [p for p in FILES if p.suffix == ".jsonl"]:
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                assert "type" in json.loads(line), f"{p.name}:{i}"


def test_csv_files_are_rectangular():
    for p in [p for p in FILES if p.suffix == ".csv"]:
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) > 1 and all(len(r) == len(rows[0]) for r in rows), p.name


def test_gzip_files_decompress():
    for p in [p for p in FILES if p.suffix == ".gz"]:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            assert sum(1 for _ in f) > 0


def test_png_files_are_png():
    for p in [p for p in FILES if p.suffix == ".png"]:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", p.name


def _anchor(heading):
    a = re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")
    return a


@pytest.mark.parametrize("md", [p for p in FILES if p.suffix == ".md"], ids=lambda p: p.name)
def test_markdown_links_resolve(md):
    text = md.read_text(encoding="utf-8")
    anchors = {_anchor(h) for h in re.findall(r"^#+ (.+)$", text, flags=re.M)}
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if re.match(r"https?://", target):
            continue
        path, _, frag = target.partition("#")
        if path:
            assert (md.parent / path).exists(), f"{md.name}: broken link {target}"
        elif frag:
            assert frag in anchors, f"{md.name}: missing anchor #{frag}"


def test_no_file_too_large_for_plain_git():
    # GitHub warns above 50 MB and rejects files above 100 MB
    big = [(p.relative_to(ROOT), p.stat().st_size) for p in FILES if p.stat().st_size > 25 * 2 ** 20]
    assert not big, big
