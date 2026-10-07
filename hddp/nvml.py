"""Minimal ctypes NVML binding (no pynvml dependency) + optional GPU throttling with best-effort restoration.

Only read-only queries are used by the training path. The throttling helpers are used solely by the
optional hardware-throttle stage. They restore the original limits on normal exit, on exceptions and on
SIGINT/SIGTERM/SIGHUP (finally/atexit/signal handlers). No user-space program can restore them after
SIGKILL, a host crash or a power loss; check `nvidia-smi -q -d POWER` after any abnormal termination.
"""
import atexit
import ctypes
import signal
import sys

_lib = None


class NVMLError(RuntimeError):
    def __init__(self, fn, rc):
        self.rc = rc
        super().__init__(f"{fn} failed: rc={rc} ({error_string(rc)})")


NVML_ERROR_NO_PERMISSION = 4
NVML_ERROR_NOT_SUPPORTED = 3

THROTTLE_BITS = {
    0x1: "gpu_idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
    0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clock",
}


def error_string(rc):
    if _lib is None:
        return "?"
    _lib.nvmlErrorString.restype = ctypes.c_char_p
    return _lib.nvmlErrorString(rc).decode()


def _call(fn, *args):
    rc = getattr(_lib, fn)(*args)
    if rc != 0:
        raise NVMLError(fn, rc)


def init():
    global _lib
    if _lib is not None:
        return
    for name in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
        try:
            _lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if _lib is None:
        raise RuntimeError("libnvidia-ml not found")
    _call("nvmlInit_v2")


def handle(index):
    h = ctypes.c_void_p()
    _call("nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint(index), ctypes.byref(h))
    return h


def _uint(fn, h, *pre):
    v = ctypes.c_uint()
    _call(fn, h, *pre, ctypes.byref(v))
    return v.value


def power_w(h):
    return _uint("nvmlDeviceGetPowerUsage", h) / 1000.0


class _FieldValue(ctypes.Structure):
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_uint), ("nvmlReturn", ctypes.c_uint),
                ("value", ctypes.c_ulonglong)]


def power_instant_w(h):
    """NVML_FI_DEV_POWER_INSTANT (186); falls back to nvmlDeviceGetPowerUsage (1 s average) if unsupported."""
    fv = _FieldValue(fieldId=186)
    if _lib.nvmlDeviceGetFieldValues(h, 1, ctypes.byref(fv)) == 0 and fv.nvmlReturn == 0:
        return fv.value / 1000.0
    return power_w(h)


def energy_j(h):
    """Cumulative energy since driver load (Volta+), millijoule counter -> joules."""
    v = ctypes.c_ulonglong()
    _call("nvmlDeviceGetTotalEnergyConsumption", h, ctypes.byref(v))
    return v.value / 1000.0


def temp_c(h):
    return _uint("nvmlDeviceGetTemperature", h, ctypes.c_uint(0))


def sm_clock_mhz(h):
    return _uint("nvmlDeviceGetClockInfo", h, ctypes.c_uint(1))


def mem_clock_mhz(h):
    return _uint("nvmlDeviceGetClockInfo", h, ctypes.c_uint(2))


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def util_pct(h):
    u = _Util()
    _call("nvmlDeviceGetUtilizationRates", h, ctypes.byref(u))
    return u.gpu


def throttle_reasons(h):
    v = ctypes.c_ulonglong()
    _call("nvmlDeviceGetCurrentClocksThrottleReasons", h, ctypes.byref(v))
    return v.value


def decode_throttle(mask):
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


def uuid(h):
    buf = ctypes.create_string_buffer(96)
    _call("nvmlDeviceGetUUID", h, buf, ctypes.c_uint(96))
    return buf.value.decode()


class _PciInfo(ctypes.Structure):
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


def pci_bus_id(h):
    """Bus id in sysfs form, e.g. '0000:01:00.0'."""
    p = _PciInfo()
    _call("nvmlDeviceGetPciInfo_v3", h, ctypes.byref(p))
    return p.busId.decode()[-12:].lower()


class _ProcInfo(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


def compute_pids(h):
    n = ctypes.c_uint(64)
    arr = (_ProcInfo * 64)()
    _call("nvmlDeviceGetComputeRunningProcesses_v3", h, ctypes.byref(n), arr)
    return [arr[i].pid for i in range(n.value)]


def power_limit_w(h):
    return _uint("nvmlDeviceGetPowerManagementLimit", h) / 1000.0


def power_limit_range_w(h):
    lo, hi = ctypes.c_uint(), ctypes.c_uint()
    _call("nvmlDeviceGetPowerManagementLimitConstraints", h, ctypes.byref(lo), ctypes.byref(hi))
    return lo.value / 1000.0, hi.value / 1000.0


def set_power_limit_w(h, watts):
    _call("nvmlDeviceSetPowerManagementLimit", h, ctypes.c_uint(int(round(watts * 1000))))


def set_locked_clocks(h, lo_mhz, hi_mhz):
    _call("nvmlDeviceSetGpuLockedClocks", h, ctypes.c_uint(lo_mhz), ctypes.c_uint(hi_mhz))


def reset_locked_clocks(h):
    _call("nvmlDeviceResetGpuLockedClocks", h)


def can_set_power_limit(index):
    """Non-destructive permission probe: re-apply the *current* limit (a no-op if permitted)."""
    init()
    h = handle(index)
    try:
        set_power_limit_w(h, power_limit_w(h))
        return True, "ok"
    except NVMLError as e:
        return False, str(e)


class ThrottleGuard:
    """Apply a power limit and/or locked SM clocks to ONE GPU; best-effort restore on exit/exception/signal
    (see the module docstring for what cannot be covered)."""

    def __init__(self, index, power_w=None, sm_clock_mhz=None):
        init()
        self.index, self.h = index, handle(index)
        self.power_w, self.sm_clock = power_w, sm_clock_mhz
        self.original = None
        self.applied = {}

    def __enter__(self):
        self.original = {"power_limit_w": power_limit_w(self.h)}
        atexit.register(self.restore)
        sigs = [getattr(signal, n) for n in ("SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, n)]
        self._prev = {s: signal.signal(s, self._on_signal) for s in sigs}
        try:
            if self.power_w is not None:
                set_power_limit_w(self.h, self.power_w)
            if self.sm_clock is not None:
                set_locked_clocks(self.h, self.sm_clock, self.sm_clock)
                self.original["locked_clocks"] = True
            self.applied = {"power_limit_w": power_limit_w(self.h), "sm_clock_mhz_now": sm_clock_mhz(self.h)}
            if self.power_w is not None and abs(self.applied["power_limit_w"] - self.power_w) > 1.0:
                raise RuntimeError(f"power limit verify failed: {self.applied}")
        except BaseException:
            self.restore()
            raise
        return self

    def _on_signal(self, signum, frame):
        try:
            self.restore()
        except RuntimeError as e:  # still exit (and let callers clean up) even if a restore step failed
            print(e, file=sys.stderr)
        raise SystemExit(128 + signum)

    def restore(self):
        if self.original is None:
            return
        errors = []
        if self.original.get("locked_clocks"):
            try:
                reset_locked_clocks(self.h)
            except NVMLError as e:
                errors.append(str(e))
        try:
            set_power_limit_w(self.h, self.original["power_limit_w"])
            if abs(power_limit_w(self.h) - self.original["power_limit_w"]) > 1.0:
                errors.append("power limit restore verify failed")
        except NVMLError as e:
            errors.append(str(e))
        self.original = None
        if errors:
            raise RuntimeError(f"GPU {self.index} restore problems: {errors}")

    def __exit__(self, *exc):
        try:
            self.restore()
        finally:
            for s, prev in getattr(self, "_prev", {}).items():
                signal.signal(s, prev)
            atexit.unregister(self.restore)
        return False
