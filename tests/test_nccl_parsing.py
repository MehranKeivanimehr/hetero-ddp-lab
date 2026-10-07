"""The archived NCCL logs re-parse to exactly the algorithm selections recorded in the raw data."""
import gzip
import json
from collections import defaultdict
from pathlib import Path

from hddp.run import parse_nccl_logs

RAW = Path(__file__).resolve().parents[1] / "results" / "raw"


def split_archive():
    """nccl_logs.txt.gz holds one section per process log, each headed '##### <launch_id> <file name>'."""
    sections, cur = defaultdict(list), None
    with gzip.open(RAW / "nccl_logs.txt.gz", "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("##### "):
                lid, name = line[6:].rsplit(" ", 1)
                cur = (lid, name.strip())
            elif cur is not None:
                sections[cur].append(line)
    return sections


def test_nccl_log_archive_reproduces_recorded_selection(tmp_path):
    sections = split_archive()
    verify = [json.loads(l) for stage in ("verify", "smoke") for l in open(RAW / f"{stage}.jsonl")]
    launches = {r["launch_id"]: r for r in verify if r["type"] == "launch" and r.get("marker")}
    recorded = defaultdict(list)
    for r in verify:
        if r["type"] == "nccl_verify":
            recorded[r["launch_id"]].append(r)
    assert set(recorded) == {lid for lid, _ in sections} == set(launches)
    for lid, recs in recorded.items():
        files = []
        for (l2, name), lines in sections.items():
            if l2 == lid:
                p = tmp_path / f"{len(files)}_{name}"
                p.write_text("".join(lines), encoding="utf-8")
                files.append(p)
        per_cond, env_lines, transports = parse_nccl_logs(files, launches[lid]["conditions"])
        assert [dict(per_cond[i]) for i in range(len(recs))] == [r["selected"] for r in recs]
        assert env_lines == recs[0]["env_lines"] and transports == recs[0]["transports"]


def test_every_allreduce_bucket_classified():
    for line in open(RAW / "verify.jsonl"):
        r = json.loads(line)
        if r["type"] == "nccl_verify":
            assert any(k.startswith("AllReduce:") for k in r["selected"]), r["launch_id"]
