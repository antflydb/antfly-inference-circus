"""Non-suspending Darwin ledger/mapped-storage observations (no dependencies).

Metric ``darwin-ledger-mapped-storage-interval-v1`` is an INTERVAL, not
``phys_footprint + mapped file size``.  Let F be the raw footprint ledger and
R the resident pages reachable through verified model-file mappings and unlinked
regular-file mappings in the owned process. This includes temporary repacked
weights whose pages need not be charged to the footprint ledger.
The ledger overlap is not exposed by libproc.  For measured R in [L, U], the
inclusive accounting interval is [max(F, L), F + U].  This deliberately retains
uncertainty about already charged dirty/private pages instead of charging them
twice in a purported exact total.  Only equal endpoints yield corrected_fp.
Intervals from different processes MUST NOT be summed as unique physical memory:
shared file pages can occur in multiple processes.  ``combine_samples`` handles
that uncertainty conservatively for a rooted, caller-owned process set.

Residency means VM-object residency in the mapped window, including file-cache
pages without a task PTE; it is not RSS, touched bytes, or file size.  Dirty,
private, and clean-candidate counters remain separate.  A clean candidate is NOT
proof that a page is external or absent from the footprint ledger.

ABI: macOS SDK sys/proc_info.h proc_regionwithpathinfo (1272 bytes),
sys/resource.h rusage_info_v4 (296 bytes), libproc.h proc_pidinfo uint64_t arg.
Layout was checked against the local SDK with sizeof/offsetof, including the
64-bit address/size/offset fields.  Kernel semantics: Apple XNU bsd/kern/proc_info.c
proc_pidregionpathinfo and osfmk/vm/bsd_vm.c fill_procregioninfo; counters come from
osfmk/vm/vm_map.c vm_map_region_walk (per-window) and vm_map_region_top_walk
(object-wide private/shared diagnostics, NOT additive window counters).

Limitations are part of the validity contract, not zero-valued fallbacks:
* libproc is a top-level walk, not a recursive submap walk. By default any
  submap makes model coverage invalid. Callers MAY explicitly select
  mapping_scope="posix-mmap-top-level" ONLY with source-backed evidence that
  original model-file and temporary-storage loaders use ordinary POSIX mappings.
  Evidence is required and retained in every sample. The audit covers retained
  original mappings, not Apple driver-created aliases or arbitrary submap
  contents; this metric does not attest total physical memory.
  The API returns EINVAL both at the end and for a missing task map; a typed
  RegionEnd is accepted only with matching live rusage identity before/after.
  EOF provenance remains explicit in the result.
* There is no atomic cross-region/ledger snapshot. Samples bracket the walk;
  identity changes, partial reads, count inconsistencies and limits invalidate
  the interval. Successful samples are observations over observer_seconds.
* Work is bounded by max_regions and a deadline BETWEEN syscalls. A libproc
  call itself scans a whole region in the kernel and cannot be preempted by
  Python. This is not a hard-real-time observer.
  Isolated benchmark sampling allows two seconds per process (the standalone
  reader defaults to 150 ms); the retained budget and actual observer time make
  this distinction explicit. Exhausted budgets remain invalid observations.
* ESRCH or an exit-marked ledger is only a candidate teardown race, never valid
  coverage. The process-tree caller may discard that incomplete observation
  only with independent positive process-identity/liveness evidence, retaining
  the discarded observation. Live/unconfirmed failures still invalidate peaks.
* task_read_for_pid / task_for_pid are denied for ordinary owned unsigned
  children on a stock host. The public SDK has no PROC_PIDREGIONPAGEINFO.
  No privileged helper, debugger, suspension, target reads, or VM writes occur.
* Caller-only vm.self_region_footprint is forced off and restored. Each read
  uses a short-lived observer thread with vm.self_region_page_size=4096;
  the thread-local override disappears when it exits. A module lock serializes
  readers; do not change region controls elsewhere in the SAME observer process.
  Neither control changes the target.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import errno
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Iterable

METRIC_VERSION = "darwin-ledger-mapped-storage-interval-v1"
PAGE_BYTES = 4096
UINT64_LIMIT = 1 << 64
SM_COW, SM_PRIVATE, SM_EMPTY, SM_SHARED, SM_TRUESHARED = 1, 2, 3, 4, 5
SM_PRIVATE_ALIASED, SM_SHARED_ALIASED, SM_LARGE_PAGE = 6, 7, 8
_CONTROL_LOCK = threading.Lock()


class _RegionInfo(ct.Structure):
    _fields_ = [
        ("protection", ct.c_uint32), ("max_protection", ct.c_uint32),
        ("inheritance", ct.c_uint32), ("flags", ct.c_uint32),
        ("offset", ct.c_uint64),
        *[(name, ct.c_uint32) for name in (
            "behavior", "user_wired_count", "user_tag", "pages_resident",
            "pages_shared_now_private", "pages_swapped_out", "pages_dirtied",
            "ref_count", "shadow_depth", "share_mode", "private_pages_resident",
            "shared_pages_resident", "obj_id", "depth")],
        ("address", ct.c_uint64), ("size", ct.c_uint64),
    ]


class _VinfoStat(ct.Structure):
    _fields_ = [
        ("dev", ct.c_uint32), ("mode", ct.c_uint16), ("nlink", ct.c_uint16),
        ("ino", ct.c_uint64), ("uid", ct.c_uint32), ("gid", ct.c_uint32),
        *[(name, ct.c_int64) for name in (
            "atime", "atimensec", "mtime", "mtimensec", "ctime", "ctimensec",
            "birthtime", "birthtimensec", "size", "blocks")],
        ("blksize", ct.c_int32), ("flags", ct.c_uint32),
        ("gen", ct.c_uint32), ("rdev", ct.c_uint32), ("qspare", ct.c_int64 * 2),
    ]


class _VnodeInfo(ct.Structure):
    _fields_ = [("stat", _VinfoStat), ("type", ct.c_int), ("pad", ct.c_int),
                ("fsid", ct.c_int32 * 2)]


class _VnodePath(ct.Structure):
    _fields_ = [("info", _VnodeInfo), ("path", ct.c_char * 1024)]


class _RegionPath(ct.Structure):
    _fields_ = [("region", _RegionInfo), ("vnode", _VnodePath)]


class _RusageV4(ct.Structure):
    _fields_ = [("uuid", ct.c_ubyte * 16), *[(name, ct.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
        "pageins", "wired_size", "resident_size", "phys_footprint",
        "proc_start_abstime", "proc_exit_abstime", "child_user_time",
        "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
        "child_pageins", "child_elapsed_abstime", "diskio_bytesread",
        "diskio_byteswritten", "cpu_time_qos_default", "cpu_time_qos_maintenance",
        "cpu_time_qos_background", "cpu_time_qos_utility", "cpu_time_qos_legacy",
        "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive",
        "billed_system_time", "serviced_system_time", "logical_writes",
        "lifetime_max_phys_footprint", "instructions", "cycles", "billed_energy",
        "serviced_energy", "interval_max_phys_footprint", "runnable_time")]]


class RegionEnd(Exception):
    """libproc EINVAL: candidate end, requiring a live-identity postcheck."""


class CoverageError(RuntimeError):
    """An incomplete or inconsistent observation, never an empty mapping set."""


class TargetUnavailable(CoverageError):
    """Possible exit/reuse; callers must confirm process identity/liveness."""


@dataclass(frozen=True)
class Region:
    address: int
    size: int
    offset: int
    resident: int
    dirty: int
    private: int
    shared: int
    share_mode: int
    shadow_depth: int
    flags: int
    dev: int
    ino: int
    path: str
    nlink: int | None = None
    file_mode: int = 0

    @property
    def unlinked_file(self) -> bool:
        return self.nlink == 0 and stat.S_ISREG(self.file_mode)

    @property
    def identity(self) -> tuple[int, int]:
        return self.dev, self.ino

    def validate(self) -> None:
        if (self.address < 0 or self.size <= 0 or self.offset < 0
                or self.address + self.size >= UINT64_LIMIT
                or self.offset + self.size > UINT64_LIMIT
                or self.address % PAGE_BYTES or self.offset % PAGE_BYTES
                or self.size % PAGE_BYTES):
            raise CoverageError("invalid region address/size/offset")
        if (not 0 <= self.dirty <= self.resident <= self.size
                or self.resident % PAGE_BYTES or self.dirty % PAGE_BYTES):
            raise CoverageError("inconsistent resident/dirty counters")
        if self.share_mode not in range(1, 9):
            raise CoverageError("unknown region sharing mode")


class _Libproc:
    def __init__(self):
        if sys.platform != "darwin":
            raise RuntimeError("DarwinMemoryReader requires macOS")
        if (ct.sizeof(_RegionInfo), ct.sizeof(_VinfoStat), ct.sizeof(_VnodeInfo),
                ct.sizeof(_RegionPath), ct.sizeof(_RusageV4),
                _RegionInfo.address.offset, _RegionPath.vnode.offset + _VnodePath.path.offset,
                _RusageV4.lifetime_max_phys_footprint.offset) != (96, 136, 152, 1272, 296, 80, 248, 240):
            raise RuntimeError("unsupported libproc structure ABI")
        self.lib = ct.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.lib.proc_pidinfo.argtypes = [ct.c_int, ct.c_int, ct.c_uint64, ct.c_void_p, ct.c_int]
        self.lib.proc_pidinfo.restype = ct.c_int
        self.lib.proc_pid_rusage.argtypes = [ct.c_int, ct.c_int, ct.c_void_p]
        self.lib.proc_pid_rusage.restype = ct.c_int
        self.system = ct.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        self.system.sysctlbyname.argtypes = [ct.c_char_p, ct.c_void_p,
                                             ct.POINTER(ct.c_size_t), ct.c_void_p, ct.c_size_t]
        self.system.sysctlbyname.restype = ct.c_int

    def ledger(self, pid: int) -> dict:
        buf = _RusageV4()
        ct.set_errno(0)
        if self.lib.proc_pid_rusage(pid, 4, ct.byref(buf)) != 0:
            error = ct.get_errno()
            kind = TargetUnavailable if error == errno.ESRCH else CoverageError
            raise kind(f"rusage errno={error}")
        if buf.proc_exit_abstime:
            raise TargetUnavailable("target has exited")
        return {"fp": int(buf.phys_footprint), "rss": int(buf.resident_size),
                "lifetime_max": int(buf.lifetime_max_phys_footprint),
                "identity": (int(buf.proc_start_abstime), bytes(buf.uuid).hex())}

    def control(self, name: str, value: int | None = None) -> int:
        old, size = ct.c_int(), ct.c_size_t(ct.sizeof(ct.c_int))
        new = ct.c_int(value) if value is not None else None
        ct.set_errno(0)
        rc = self.system.sysctlbyname(name.encode(), ct.byref(old), ct.byref(size),
                                      ct.byref(new) if new is not None else None,
                                      ct.sizeof(new) if new is not None else 0)
        if rc != 0 or size.value != ct.sizeof(old):
            raise CoverageError(f"{name} errno={ct.get_errno()}")
        return old.value

    def region(self, pid: int, address: int) -> Region:
        buf = _RegionPath()
        ct.set_errno(0)
        n = self.lib.proc_pidinfo(pid, 8, address, ct.byref(buf), ct.sizeof(buf))
        error = ct.get_errno()
        if n == 0 and error == errno.EINVAL:
            raise RegionEnd()
        if n != ct.sizeof(buf):
            kind = TargetUnavailable if n == 0 and error == errno.ESRCH else CoverageError
            raise kind(f"region read size={n} errno={error}")
        r, v = buf.region, buf.vnode
        native_page = os.sysconf("SC_PAGESIZE")
        # top_walk uses PAGE_SIZE_64, unlike extended region counters, which
        # honor the caller's effective page-size override. Superpages also use
        # the native page shift in vm_map_region_walk.
        resident_unit = native_page if r.share_mode == SM_LARGE_PAGE else PAGE_BYTES
        return Region(int(r.address), int(r.size), int(r.offset),
                      int(r.pages_resident) * resident_unit, int(r.pages_dirtied) * PAGE_BYTES,
                      int(r.private_pages_resident) * native_page,
                      int(r.shared_pages_resident) * native_page,
                      int(r.share_mode), int(r.shadow_depth), int(r.flags),
                      int(v.info.stat.dev), int(v.info.stat.ino), os.fsdecode(bytes(v.path)),
                      int(v.info.stat.nlink), int(v.info.stat.mode))


class _Artifacts:
    def __init__(self, paths: Iterable[Path | str]):
        self.files: set[tuple[int, int]] = set()
        self.roots: list[tuple[Path, tuple[int, int]]] = []
        self.paths: list[Path] = []
        for raw in paths:
            path = Path(raw).resolve(strict=True)
            stat = path.stat()
            self.paths.append(path)
            identity = (stat.st_dev & 0xFFFFFFFF, stat.st_ino)
            if path.is_dir():
                self.roots.append((path, identity))
            elif path.is_file():
                self.files.add(identity)
            else:
                raise ValueError(f"artifact is not a regular file/directory: {path}")
        if not self.paths:
            raise ValueError("at least one verified model artifact path/root is required")

    def match(self, region: Region) -> bool:
        if not region.ino:
            return False
        if region.unlinked_file:
            return True  # Kernel vnode identity survives unlink; a pathname stat cannot.
        if region.identity in self.files:
            return True  # The vnode identity also recognizes hard-linked/renamed aliases.
        if not self.roots:
            return False
        if not region.path or len(os.fsencode(region.path)) >= 1024:
            raise CoverageError("file mapping path unavailable/truncated under root matching")
        path = Path(region.path).resolve()
        if not any(path.is_relative_to(root) for root, _ in self.roots):
            return False
        try:
            stat = path.stat()
        except OSError as exc:
            raise CoverageError(f"model mapping identity unavailable: {exc.errno}") from exc
        if (stat.st_dev & 0xFFFFFFFF, stat.st_ino) != region.identity:
            raise CoverageError("model mapping path/vnode identity mismatch")
        return True

    def verify_roots(self) -> None:
        for path, expected in self.roots:
            stat = path.stat()
            if (stat.st_dev & 0xFFFFFFFF, stat.st_ino) != expected:
                raise CoverageError("model artifact root identity changed")


def _offset_bounds(regions: list[Region], *, clean: bool = False) -> tuple[int, int]:
    """Conservative physical-residency bounds, never proportional alias guesses.

    Unshadowed file windows share the vnode's VM object. In each overlapping
    interval component, max(count) <= union resident bytes <= min(sum(count),
    union window capacity). Disjoint windows add. COW/shadow/large-page windows
    do not have a verified vnode-relative offset: retain their full upper bound,
    and only use a max for the identity's lower bound (they may still alias).
    """
    groups: dict[tuple[int, int], list[Region]] = {}
    for region in regions:
        region.validate()
        groups.setdefault(region.identity, []).append(region)
    total_lower = total_upper = 0
    for group in groups.values():
        regular, uncertain = [], []
        for r in group:
            count = r.resident - r.dirty if clean else r.resident
            if r.shadow_depth or r.share_mode in (SM_COW, SM_LARGE_PAGE):
                uncertain.append(count)
            else:
                regular.append((r.offset, r.offset + r.size, count))
        lower = upper = 0
        end = capacity = summed = largest = 0
        for start, stop, count in sorted(regular):
            if start >= end:
                lower += largest
                upper += min(summed, capacity)
                end, capacity, summed, largest = stop, stop - start, count, count
            else:
                capacity += max(0, stop - end)
                end = max(end, stop)
                summed += count
                largest = max(largest, count)
        lower += largest
        upper += min(summed, capacity)
        total_lower += max([lower, *uncertain])
        total_upper += upper + sum(uncertain)
    return total_lower, total_upper


def combine_samples(samples: Iterable[dict]) -> dict:
    """Conservative interval for one concurrently observed owned process tree.

    Across processes, the same resident file page can be shared. Therefore the
    tree's resident lower bound is max(per-process lower), NOT their sum. Upper
    bounds may sum; the uncertainty explicitly includes cross-process aliasing.
    Raw ledgers sum under the caller's existing owned-process accounting policy.
    Never combine independently peaked samples with this function.
    """
    rows = list(samples)
    if not rows or not all(r.get("coverage_valid") for r in rows):
        return {"valid": False, "lower_bytes": None, "upper_bytes": None}
    fp = sum(r["fp"] for r in rows)
    lower = max(r["mapped_storage_resident_lower_bytes"] for r in rows)
    upper = sum(r["mapped_storage_resident_upper_bytes"] for r in rows)
    return {"valid": True, "lower_bytes": max(fp, lower), "upper_bytes": fp + upper}


class DarwinMemoryReader:
    def __init__(self, model_paths: Iterable[Path | str], *, max_regions: int = 8192,
                 max_seconds: float = 0.15, mapping_scope: str = "all-model-mappings",
                 mapping_scope_evidence: str | None = None):
        if max_regions <= 0 or not max_seconds > 0:
            raise ValueError("positive region and time budgets are required")
        if mapping_scope not in ("all-model-mappings", "posix-mmap-top-level"):
            raise ValueError("unknown model mapping coverage scope")
        if mapping_scope == "posix-mmap-top-level" and not (
                mapping_scope_evidence and mapping_scope_evidence.strip()):
            raise ValueError("source evidence is required for scoped model coverage")
        self.mapping_scope = mapping_scope
        self.mapping_scope_evidence = mapping_scope_evidence
        self.artifacts = _Artifacts(model_paths)
        self.max_regions, self.max_seconds = max_regions, max_seconds
        self.backend = _Libproc()
        self.last_error: str | None = None
        self.last_target_unavailable = False

    def read(self, pid: int) -> dict | None:
        """Return raw ledger plus measured components; None means no ledger.

        Invalid coverage retains observed subtotals but sets all model-residency
        bounds and corrected accounting endpoints to None, never fallback zero.
        """
        started = time.monotonic()
        output, failures = [], []

        def observe():
            try:
                output.append(self._read(pid))
            except BaseException as exc:
                failures.append(exc)

        observer = threading.Thread(target=observe, name="darwin-memory-observer")
        observer.start()
        observer.join()
        if failures:
            raise failures[0]
        result = output[0]
        if result is not None:
            result["observer_seconds"] = time.monotonic() - started
        return result

    def _read(self, pid: int) -> dict | None:
        started = time.monotonic()
        self.last_target_unavailable = False
        try:
            before = self.backend.ledger(pid)
        except CoverageError as exc:
            self.last_error = str(exc)
            self.last_target_unavailable = isinstance(exc, TargetUnavailable)
            return None
        errors: list[str] = []
        matched: list[Region] = []
        count = submaps = 0
        complete = False
        eof = "not_reached"
        after = before
        target_unavailable = False
        with _CONTROL_LOCK:
            saved: list[tuple[str, int]] = []
            try:
                # Footprint mode is process-local; page-size mode is thread-local.
                old = self.backend.control("vm.self_region_footprint", 0)
                saved.append(("vm.self_region_footprint", old))
                self.backend.control("vm.self_region_page_size", PAGE_BYTES)
                self.artifacts.verify_roots()
                cursor = 0
                while count < self.max_regions:
                    if time.monotonic() - started >= self.max_seconds:
                        raise CoverageError("region walk time budget exhausted")
                    try:
                        region = self.backend.region(pid, cursor)
                    except RegionEnd:
                        eof = "libproc_einval_live_identity_required"
                        complete = count > 0
                        if not complete:
                            raise CoverageError("EINVAL before any region: not verified EOF")
                        break
                    region.validate()
                    if region.address < cursor:
                        raise CoverageError("region walk did not advance")
                    cursor = region.address + region.size
                    count += 1
                    if region.flags & 1:
                        submaps += 1
                    if self.artifacts.match(region):
                        matched.append(region)
                else:
                    raise CoverageError("region walk count budget exhausted")
                after = self.backend.ledger(pid)
                if after["identity"] != before["identity"]:
                    raise TargetUnavailable("target identity changed during observation")
                self.artifacts.verify_roots()
                if submaps and self.mapping_scope == "all-model-mappings":
                    raise CoverageError("submaps are not recursively visible through libproc")
            except (CoverageError, OSError) as exc:
                errors.append(str(exc))
                target_unavailable = isinstance(exc, TargetUnavailable)
            finally:
                for name, value in reversed(saved):
                    try:
                        self.backend.control(name, value)
                    except CoverageError as exc:
                        errors.append(f"observer control restoration failed: {exc}")
                        target_unavailable = False
        valid = complete and not errors
        lower, upper = _offset_bounds(matched)
        clean_lower, clean_upper = _offset_bounds(matched, clean=True)
        # Ledger is read at the end of the same observation, never a lifetime peak.
        fp = after["fp"]
        inclusive_lower, inclusive_upper = max(fp, lower), fp + upper
        result = {key: after[key] for key in ("fp", "rss", "lifetime_max")}
        result.update({
            "metric_version": METRIC_VERSION,
            "coverage_valid": valid,
            "coverage_scope": self.mapping_scope,
            "coverage_evidence": self.mapping_scope_evidence,
            "top_level_walk_complete": complete,
            "eof": eof,
            "errors": errors,
            "target_identity": before["identity"],
            "target_unavailable": target_unavailable,
            "region_walk_budget_seconds": self.max_seconds,
            "region_walk_budget_count": self.max_regions,
            "region_count": count,
            "mapped_storage_region_count": len(matched),
            "unlinked_storage_region_count": sum(r.unlinked_file for r in matched),
            "unlinked_storage_observed_resident_bytes": sum(r.resident for r in matched if r.unlinked_file),
            "submap_count": submaps,
            "page_bytes": PAGE_BYTES,
            "observer_seconds": time.monotonic() - started,
            "ledger_before_bytes": before["fp"],
            "mapped_storage_observed_resident_bytes": sum(r.resident for r in matched),
            "mapped_storage_observed_dirty_bytes": sum(r.dirty for r in matched),
            "mapped_storage_observed_clean_candidate_bytes": sum(r.resident - r.dirty for r in matched),
            "mapped_storage_observed_private_object_resident_bytes": sum(r.private for r in matched),
            "mapped_storage_observed_shared_object_resident_bytes": sum(r.shared for r in matched),
            "mapped_storage_resident_lower_bytes": lower if valid else None,
            "mapped_storage_resident_upper_bytes": upper if valid else None,
            "mapped_storage_clean_candidate_lower_bytes": clean_lower if valid else None,
            "mapped_storage_clean_candidate_upper_bytes": clean_upper if valid else None,
            "corrected_fp_interval_valid": valid,
            "corrected_fp_lower_bytes": inclusive_lower if valid else None,
            "corrected_fp_upper_bytes": inclusive_upper if valid else None,
            "corrected_fp": inclusive_lower if valid and inclusive_lower == inclusive_upper else None,
            "corrected_fp_valid": valid and inclusive_lower == inclusive_upper,
        })
        self.last_error = "; ".join(errors) or None
        return result


def _self_check_child(path: str, mode: str) -> None:
    import mmap
    heartbeat = {"ticks": 0, "max_gap_seconds": 0.0}
    stop = threading.Event()

    def beat():
        previous = time.monotonic()
        while not stop.wait(0.002):
            now = time.monotonic()
            heartbeat["ticks"] += 1
            heartbeat["max_gap_seconds"] = max(heartbeat["max_gap_seconds"], now - previous)
            previous = now

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    allocation = None
    print(json.dumps({"ready": True}), flush=True)
    try:
        for command in sys.stdin:
            if command.strip() == "allocate":
                if mode == "mapped":
                    with open(path, "rb") as source:
                        allocation = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
                    checksum = sum(allocation[i] for i in range(0, len(allocation), PAGE_BYTES))
                else:
                    with open(path, "rb") as source:
                        allocation = bytearray(os.fstat(source.fileno()).st_size)
                        if source.readinto(allocation) != len(allocation):
                            raise RuntimeError("short controlled model read")
                    checksum = sum(allocation[::PAGE_BYTES])
                print(json.dumps({"allocated": len(allocation), "checksum": checksum, **heartbeat}), flush=True)
            elif command.strip() == "status":
                print(json.dumps(heartbeat), flush=True)
                heartbeat["max_gap_seconds"] = 0.0
            elif command.strip() == "quit":
                break
    finally:
        stop.set()
        thread.join()
        if isinstance(allocation, mmap.mmap):
            allocation.close()


def self_check() -> dict:
    """Exercise actual owned mmap/copy children and report observer/heartbeat cost."""
    if sys.platform != "darwin":
        raise RuntimeError("--self-check requires macOS")
    size = 64 * 1024 * 1024
    evidence = {"metric_version": METRIC_VERSION, "model_bytes": size, "scenarios": {}}
    with tempfile.TemporaryDirectory(prefix="termite-memory-", dir="/tmp") as temporary:
        path = Path(temporary) / "controlled-model.bin"
        with path.open("wb") as output:
            block = os.urandom(1024 * 1024)
            for _ in range(size // len(block)):
                output.write(block)
        reader = DarwinMemoryReader(
            [path], max_seconds=2.0, mapping_scope="posix-mmap-top-level",
            mapping_scope_evidence=(
                "Controlled _self_check_child uses only Python mmap.mmap(ACCESS_READ) "
                "or file.readinto(bytearray); no submap construction/remapping."))
        unrestricted_reader = DarwinMemoryReader([path], max_seconds=2.0)
        for mode in ("mapped", "copied"):
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                      "--self-check-child", str(path), mode],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            try:
                def receive() -> dict:
                    line = child.stdout.readline()
                    if not line:
                        raise RuntimeError(f"self-check child exited: {child.stderr.read()}")
                    return json.loads(line)

                def command(text: str) -> dict:
                    child.stdin.write(text + "\n")
                    child.stdin.flush()
                    return receive()

                receive()
                before = reader.read(child.pid)
                allocation = command("allocate")
                time.sleep(0.05)
                quiet_start = command("status")
                quiet_started = time.monotonic()
                time.sleep(0.1)
                quiet_end = command("status")
                quiet_seconds = time.monotonic() - quiet_started
                observed_started = time.monotonic()
                cpu_start = time.process_time()
                samples = []
                for _ in range(8):
                    samples.append(reader.read(child.pid))
                    time.sleep(0.01)
                observer_cpu = time.process_time() - cpu_start
                observed_end = command("status")
                observed_seconds = time.monotonic() - observed_started
                if before is None or any(sample is None for sample in samples):
                    raise RuntimeError("self-check could not read a live child ledger")
                last = samples[-1]
                if not all(sample["coverage_valid"] for sample in samples):
                    raise RuntimeError(f"controlled mapping scope was not covered: {last['errors']}")
                unrestricted = unrestricted_reader.read(child.pid)
                resident = last["mapped_storage_observed_resident_bytes"]
                ledger_delta = last["fp"] - before["fp"]
                if mode == "mapped" and resident < size * 0.9:
                    raise RuntimeError("touched mapping residency was not observed")
                if mode == "copied" and (resident != 0 or ledger_delta < size * 0.75):
                    raise RuntimeError("copied allocation was not distinguished from file mapping")
                if observed_end["ticks"] <= quiet_end["ticks"]:
                    raise RuntimeError("owned child heartbeat made no progress during observation")
                evidence["scenarios"][mode] = {
                    "allocation_bytes": allocation["allocated"],
                    "ledger_delta_bytes": ledger_delta,
                    "mapped_resident_bytes": resident,
                    "coverage_valid": last["coverage_valid"],
                    "coverage_errors": last["errors"],
                    "coverage_scope": last["coverage_scope"],
                    "unrestricted_coverage_valid": unrestricted["coverage_valid"],
                    "unrestricted_coverage_errors": unrestricted["errors"],
                    "inclusive_lower_bytes": last["corrected_fp_lower_bytes"],
                    "inclusive_upper_bytes": last["corrected_fp_upper_bytes"],
                    "quiet_wall_seconds": quiet_seconds,
                    "observed_wall_seconds": observed_seconds,
                    "quiet_heartbeat_ticks": quiet_end["ticks"] - quiet_start["ticks"],
                    "quiet_heartbeat_max_gap_seconds": quiet_end["max_gap_seconds"],
                    "observed_heartbeat_ticks": observed_end["ticks"] - quiet_end["ticks"],
                    "heartbeat_max_gap_seconds": observed_end["max_gap_seconds"],
                    "observer_cpu_seconds": observer_cpu,
                    "observer_mean_seconds": sum(s["observer_seconds"] for s in samples) / len(samples),
                    "observer_max_seconds": max(s["observer_seconds"] for s in samples),
                }
            finally:
                if child.poll() is None:
                    child.stdin.write("quit\n")
                    child.stdin.flush()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        child.wait(timeout=5)
                child.stdin.close()
                child.stdout.close()
                child.stderr.close()
    evidence["ok"] = True
    evidence["accounting_available"] = all(
        row["coverage_valid"] for row in evidence["scenarios"].values())
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--self-check-child", nargs=2, metavar=("PATH", "MODE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_check_child:
        _self_check_child(*args.self_check_child)
    elif args.self_check:
        print(json.dumps(self_check(), sort_keys=True))
    else:
        parser.error("use --self-check")


if __name__ == "__main__":
    main()
