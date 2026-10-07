"""No private identifiers in the published files, and the redaction itself is correct and idempotent."""
import gzip
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hddp import privacy

ROOT = Path(__file__).resolve().parents[1]
RAW, ENV = ROOT / "results" / "raw", ROOT / "results" / "env"

FORBIDDEN = {
    "GPU UUID": privacy.GPU_UUID_RE,
    "IPv4 address": privacy.IPV4_RE,
    "home directory": re.compile(r"/home/(?!<user>)[^\s/\"']+"),
    "Windows user directory": re.compile(r"[A-Za-z]:\\+Users\\+(?!<user>)[^\\\s\"']+"),
    "calendar date": re.compile(r"\b20\d\d-[01]\d-[0-3]\d\b|\b20\d{6}T\d{6}\b"),
    "e-mail address": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.I),
    "credential": re.compile(r"(api[_-]?key|passw(or)?d|secret[_-]?key)\s*[=:]\s*['\"]?\w{6,}|ghp_\w{20,}|"
                             r"github_pat_\w+|hf_\w{20,}|AKIA[0-9A-Z]{16}|xox[bp]-\w+|-----BEGIN [A-Z ]*PRIVATE KEY",
                             re.I),
}
CLOCK = re.compile(r"(?<![\d:.])(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?![\d:])")


def published_text_files():
    skip = {".png", ".pyc"}
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_file() and p.suffix not in skip and not {".git", "__pycache__", ".pytest_cache"} & set(rel.parts):
            yield p


def read(p):
    return gzip.open(p, "rt", encoding="utf-8").read() if p.suffix == ".gz" else p.read_text(encoding="utf-8")


def test_no_forbidden_patterns_in_published_files():
    hits = []
    for p in published_text_files():
        if p.parent.name == "tests" or p.name in ("privacy.py", "sanitize_data.py"):
            continue  # these define the patterns themselves
        text = read(p)
        for what, rx in FORBIDDEN.items():
            if rx.search(text):
                hits.append(f"{p.relative_to(ROOT)}: {what}: {rx.search(text).group(0)!r}")
        if p.parent in (RAW, ENV) and CLOCK.search(text):
            hits.append(f"{p.relative_to(ROOT)}: clock time {CLOCK.search(text).group(0)!r}")
    assert not hits, "\n".join(hits)


def test_raw_record_identifiers_are_placeholders():
    attempt_ok = re.compile(r"^(attempt-\d{3}|[0-9a-f]{8})$")
    pseudonym = re.compile(r"^P\w+$")
    for p in RAW.glob("*.jsonl"):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            assert "time" not in r and "hostname" not in r
            assert r.get("attempt") in (None, "") or attempt_ok.match(r["attempt"]), r.get("attempt")
            for k in r.get("ranks", []):
                assert all(pseudonym.match(str(x)) for x in k["foreign_pids"])
            for side in ("gpu_pre", "gpu_post"):
                for g in (r.get(side) or {}).values():
                    assert all(pseudonym.match(x.split(":")[0]) and "/" not in x for x in g["procs"])
            for t, gpu, pid in r.get("foreign_watch") or []:
                assert isinstance(t, int) and pseudonym.match(pid)


def test_nccl_log_prefixes_are_anonymous():
    prefix = re.compile(r"^([^\s:]+):([^\s:]+):([^\s:]+) \[\d+\] NCCL")
    with gzip.open(RAW / "nccl_logs.txt.gz", "rt", encoding="utf-8") as f:
        for line in f:
            m = prefix.match(line)
            if m:
                assert m.groups() == (privacy.HOST, "<pid>", "<tid>"), line
            if line.startswith("##### "):
                assert re.search(r" nccl\.host\.p\d+\.log$", line.strip()), line


def test_environment_record_has_no_host_identity():
    topo = json.loads((ENV / "topology.json").read_text())
    assert "hostname" not in topo["versions"]
    assert "UUID: GPU-<uuid>" in (ENV / "machine.txt").read_text(encoding="utf-8")


SYNTHETIC_UUID = "GPU-" + "-".join("0" * n for n in (8, 4, 4, 4, 12))


def test_redact_text_examples():
    s = ("/data/home/jdoe/miniforge3/bin/python --out /data/home/jdoe/proj/hetero-ddp-lab/results/raw/x.jsonl "
         f"node7.example:123:456 Using eth0:10.1.2.3<0> {SYNTHETIC_UUID} /data/home/jdoe/x /home/jdoe "
         r"C:\Users\jdoe\repo")
    out = privacy.redact_text(s, ["node7.example"])
    assert out == ("<python-env>/bin/python --out <repo>/results/raw/x.jsonl host:123:456 Using eth0:<ip><0> "
                   r"GPU-<uuid> /home/<user> /home/<user> C:\Users\<user>\repo")
    assert privacy.redact_text(out, ["node7.example"]) == out  # idempotent
    assert privacy.redact_text("NCCL version 2.21.5+cuda12.4 driver 580.173.02") == \
        "NCCL version 2.21.5+cuda12.4 driver 580.173.02"  # version numbers are not addresses


def test_machine_text_redaction():
    banner = time.strftime("%a %b %d %H:%M:%S %Y", time.gmtime(0))  # synthetic nvidia-smi banner (epoch)
    text = (banner + "\n"
            "|    0   N/A  N/A            4242      C   python                                 9972MiB |\n"
            "Vulnerability Spectre v1: Mitigation\nVulnerability Mds: Not affected\nModel name: AMD EPYC 7713\n")
    out = privacy.redact_machine_text(text)
    assert banner not in out and "4242" not in out and "Spectre" not in out
    assert out.count("Vulnerability") == 1 and "AMD EPYC 7713" in out


def test_pid_pseudonyms_are_stable_and_salted():
    assert privacy.pid_pseudonym(1234, "a") == privacy.pid_pseudonym(1234, "a")
    assert privacy.pid_pseudonym(1234, "a") != privacy.pid_pseudonym(1234, "b")
    assert "1234" not in privacy.proc_label(1234, "/home/x/bin/python", "10 MiB", "a")


def test_sanitizer_is_idempotent_on_published_data(tmp_path):
    copy = tmp_path / "repo"
    shutil.copytree(ROOT / "results", copy / "results")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "sanitize_data.py"), "--root", str(copy)],
                       capture_output=True, text=True, check=True)
    assert "sanitized" not in r.stdout, r.stdout
    for p in (ROOT / "results").rglob("*"):
        if p.is_file():
            assert p.read_bytes() == (copy / p.relative_to(ROOT)).read_bytes(), p
