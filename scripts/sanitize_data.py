"""Remove private identifiers from results/ in place, without touching any measurement.

  python scripts/sanitize_data.py [--root REPO]

This is the exact procedure that converted the historical raw data into the published form. It is
idempotent: on already sanitized data it changes nothing. What it does:

* host name -> "host" (discovered from results/env/topology.json and the NCCL log line prefixes);
* home-directory, Python-installation and checkout paths -> "/home/<user>", "<python-env>", "<repo>";
* IPv4 addresses -> "<ip>"; GPU UUIDs -> "GPU-<uuid>"; CPU vulnerability status lines are dropped;
* wall-clock timestamps are removed (log headers, torch log prefixes, the per-condition "time" field,
  the nvidia-smi banner date);
* launch-attempt ids "<YYYYmmddTHHMMSS>-<pid>" -> "attempt-NNN" (numbered in order of first appearance,
  consistently across JSONL and logs);
* process ids of other tenants -> dataset-wide pseudonyms "P1", "P2", ... (same process, same pseudonym);
  the clock time of each continuous-watch sighting becomes seconds since the attempt started;
* process ids of this study's own processes in logs -> "<pid>".

Numbers in measurement fields are never modified; tests/test_sanitization.py and
scripts/check_reproducibility.py verify that the processed tables are unchanged.
"""
import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hddp import privacy  # noqa: E402

OLD_ATTEMPT_RE = re.compile(r"\b(\d{8})T(\d{2})(\d{2})(\d{2})-\d+\b")
CLOCK_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")
NCCL_PREFIX_RE = re.compile(r"^([A-Za-z0-9._-]+):(\d+):(\d+) (\[\d+\] )")
NCCL_FILE_RE = re.compile(r"nccl\.([A-Za-z0-9._-]+)\.(\d+)\.log")
LOG_HEADER_RE = re.compile(r"^(===== \S+)  \d{4}-\d\d-\d\d \d\d:\d\d:\d\d  ")
GLOG_PREFIX_RE = re.compile(r"^([WEIF])\d{4} \d\d:\d\d:\d\d\.\d+ +\d+ ")
OWN_PID_RES = [re.compile(r"(Sending process )\d+( closing signal)"), re.compile(r"(Process )\d+( got signal)")]
LOG_PROC_RE = re.compile(r"'(\d+):([^:'\s]+):(\d+ MiB)'")


class Maps:
    """Dataset-wide identifier maps, so the same attempt / foreign process keeps one placeholder."""

    def __init__(self):
        self.attempts, self.pids, self.nccl_files = {}, {}, {}

    def attempt(self, a):
        if not isinstance(a, str) or not OLD_ATTEMPT_RE.fullmatch(a):
            return a
        return self.attempts.setdefault(a, f"attempt-{len(self.attempts) + 1:03d}")

    def pid(self, p):
        if isinstance(p, str) and not p.isdigit():
            return p  # already a pseudonym
        return self.pids.setdefault(int(p), f"P{len(self.pids) + 1}")

    def proc(self, entry):
        pid, name, mem = entry.split(":", 2)
        if not pid.isdigit():
            return entry
        return f"{self.pid(pid)}:{os.path.basename(name)}:{mem}"


def _seconds(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)


def _strings(obj, fn):
    if isinstance(obj, dict):
        return {k: _strings(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strings(v, fn) for v in obj]
    return fn(obj) if isinstance(obj, str) else obj


def sanitize_record(r, maps, hosts):
    old_attempt = r.get("attempt")
    start = OLD_ATTEMPT_RE.fullmatch(old_attempt) if isinstance(old_attempt, str) else None
    if "attempt" in r:
        r["attempt"] = maps.attempt(old_attempt)
    r.pop("time", None)
    for k in r.get("ranks") or []:
        k["foreign_pids"] = [maps.pid(p) for p in k.get("foreign_pids", [])]
    for side in ("gpu_pre", "gpu_post"):
        for g in (r.get(side) or {}).values():
            g["procs"] = [maps.proc(p) for p in g.get("procs", [])]
    watch = []
    for t, gpu, pid in r.get("foreign_watch") or []:
        m = CLOCK_RE.match(t) if isinstance(t, str) else None
        if m:
            t = (_seconds(*m.groups()) - _seconds(*start.groups()[1:])) % 86400 if start else None
        watch.append([t, gpu, maps.pid(pid)])
    if "foreign_watch" in r:
        r["foreign_watch"] = watch
    return _strings(r, lambda s: privacy.redact_text(s, hosts))


def sanitize_log_line(line, maps, hosts):
    line = LOG_HEADER_RE.sub(r"\1  ", line)
    line = GLOG_PREFIX_RE.sub(r"\1 ", line)
    for rx in OWN_PID_RES:
        line = rx.sub(r"\1<pid>\2", line)
    line = LOG_PROC_RE.sub(lambda m: "'" + maps.proc(":".join(m.groups())) + "'", line)
    line = OLD_ATTEMPT_RE.sub(lambda m: maps.attempt(m.group(0)), line)
    return privacy.redact_text(line, hosts)


def sanitize_nccl_line(line, maps, hosts):
    line = NCCL_PREFIX_RE.sub(lambda m: f"{privacy.HOST}:<pid>:<tid> {m.group(4)}", line)
    line = NCCL_FILE_RE.sub(lambda m: "nccl.{}.p{}.log".format(
        privacy.HOST, maps.nccl_files.setdefault(m.group(0), len(maps.nccl_files) + 1)), line)
    return privacy.redact_text(line, hosts)


def discover_hosts(root):
    hosts = set()
    topo = root / "results" / "env" / "topology.json"
    if topo.exists():
        hosts.add(json.loads(topo.read_text(encoding="utf-8")).get("versions", {}).get("hostname"))
    gz = root / "results" / "raw" / "nccl_logs.txt.gz"
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            for line in f:
                for rx in (NCCL_PREFIX_RE, NCCL_FILE_RE):
                    m = rx.search(line)
                    if m:
                        hosts.add(m.group(1))
    return {h for h in hosts if h and h != privacy.HOST}


def write_if_changed(path, data):
    old = path.read_bytes()
    if old != data:
        path.write_bytes(data)
        print("sanitized", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    root = Path(ap.parse_args().root)
    raw, env = root / "results" / "raw", root / "results" / "env"
    hosts, maps = discover_hosts(root), Maps()

    for p in sorted(raw.glob("*.jsonl"), key=lambda p: (p.stem != "final", p.stem)):
        with open(p, encoding="utf-8") as f:
            lines = [json.dumps(sanitize_record(json.loads(l), maps, hosts)) + "\n" for l in f if l.strip()]
        write_if_changed(p, "".join(lines).encode())
    for p in sorted(raw.glob("*.log")):
        with open(p, encoding="utf-8", newline="") as f:
            text = "".join(sanitize_log_line(l, maps, hosts) for l in f)
        write_if_changed(p, text.encode())
    gz = raw / "nccl_logs.txt.gz"
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8", newline="") as f:
            text = "".join(sanitize_nccl_line(l, maps, hosts) for l in f)
        buf = gzip.compress(text.encode(), compresslevel=9, mtime=0)
        if gzip.decompress(gz.read_bytes()) != text.encode():
            gz.write_bytes(buf)
            print("sanitized", gz)
    topo = env / "topology.json"
    if topo.exists():
        t = json.loads(topo.read_text(encoding="utf-8"))
        t.get("versions", {}).pop("hostname", None)
        t = _strings(t, lambda s: privacy.redact_text(s, hosts))
        write_if_changed(topo, json.dumps(t, indent=1).encode())
    mt = env / "machine.txt"
    if mt.exists():
        write_if_changed(mt, privacy.redact_machine_text(mt.read_bytes().decode("utf-8"), hosts).encode())
    print(f"{len(maps.attempts)} attempt ids, {len(maps.pids)} foreign process ids, "
          f"{len(maps.nccl_files)} NCCL log names replaced")


if __name__ == "__main__":
    main()
