"""Redaction helpers for anything written to results/ (raw records, logs, machine characterization).

The published data must not contain host names, user names, home-directory paths, network addresses,
GPU serial identifiers (UUIDs), process ids of other tenants, or wall-clock timestamps. The helpers here
are used both by the experiment code (so new runs write privacy-safe records) and by
scripts/sanitize_data.py (which converted the historical raw data). None of them touch numeric
measurements.
"""
import hashlib
import os
import re
import secrets

HOST = "host"  # placeholder for the machine's host name
PY_ENV = "<python-env>"  # placeholder for the Python installation prefix
REPO = "<repo>"  # placeholder for the repository checkout path

GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
IPV4_RE = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
# absolute path of a Python installation (conda, mamba, venv) up to and including the environment directory
PY_ENV_RE = re.compile(r"(?:/[^\s/\"']+)*/(?:miniforge3|miniconda3|anaconda3|mambaforge|micromamba|\.?venv|"
                       r"envs/[^\s/\"']+)(?=/)")
# absolute path of the repository checkout
REPO_RE = re.compile(r"(?:/[^\s/\"']+)+/hetero-ddp-lab(?=[/\s\"']|$)")
HOME_RE = re.compile(r"(?:/[^\s/\"']+)*?/home/[^\s/\"']+(?:/[^\s/\"']+)?")
WIN_USER_RE = re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+")


def redact_text(text, hostnames=()):
    """Replace host names, user paths, network addresses and GPU UUIDs in free text."""
    for h in sorted({h for h in hostnames if h}, key=len, reverse=True):
        text = text.replace(h, HOST)
    text = PY_ENV_RE.sub(PY_ENV, text)
    text = REPO_RE.sub(REPO, text)
    text = HOME_RE.sub("/home/<user>", text)
    text = WIN_USER_RE.sub(r"C:\\Users\\<user>", text)
    text = GPU_UUID_RE.sub("GPU-<uuid>", text)
    return IPV4_RE.sub("<ip>", text)


SMI_BANNER_RE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \w{3} +\d+ \d\d:\d\d:\d\d \d{4}\s*$")
SMI_PROC_ROW_RE = re.compile(r"^\|\s+\d+\s+\S+\s+\S+\s+\d+\s+[CGM+]+\s+.*\|\s*$")


def redact_machine_text(text, hostnames=()):
    """Machine characterization (nvidia-smi, lscpu, ...): additionally drop the nvidia-smi clock banner, the
    per-process rows of the nvidia-smi process table, and the CPU vulnerability status lines."""
    out, dropped = [], False
    for line in text.splitlines(keepends=True):
        if SMI_BANNER_RE.match(line):
            continue
        if SMI_PROC_ROW_RE.match(line):
            out.append("|  <process row redacted>" + " " * 64 + "|\n")
            continue
        if line.startswith("Vulnerability "):
            if not dropped:
                out.append("Vulnerability status:                    <redacted>\n")
            dropped = True
            continue
        out.append(redact_text(line, hostnames))
    return "".join(out)


def new_attempt_id():
    """Opaque launch-attempt id (no clock time, no process id)."""
    return secrets.token_hex(4)


def pid_salt():
    """Per-launch secret shared with the workers through the environment; never written to disk."""
    return os.environ.get("HDDP_PID_SALT") or secrets.token_hex(16)


def pid_pseudonym(pid, salt):
    """Stable, non-reversible stand-in for a foreign process id (stable within one launch)."""
    return "P" + hashlib.sha256(f"{salt}:{pid}".encode()).hexdigest()[:8]


def proc_label(pid, name, mem, salt):
    """Process-list entry without the process id or any directory part of the executable path."""
    return f"{pid_pseudonym(pid, salt)}:{os.path.basename(name.strip())}:{mem}"
