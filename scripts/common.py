"""Shared lifecycle, sampling, and path utilities for portable inference benchmarks."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("TERMITE_RESULTS", ROOT / "results")).expanduser()
DATA = Path(os.environ.get("TERMITE_DATA", ROOT / ".data")).expanduser()
MODELS_DIR = Path(os.environ.get("TERMITE_MODELS_DIR", DATA / "models")).expanduser()
HF_HOME = Path(os.environ.get("TERMITE_HF_HOME", DATA / "huggingface")).expanduser()
CORPUS = Path(os.environ.get("TERMITE_CORPUS", DATA / "corpus" / "corpus.json")).expanduser()
MEDIA = Path(os.environ.get("TERMITE_MEDIA", DATA / "media")).expanduser()
REF_PY = Path(os.environ.get("TERMITE_REFERENCE_PYTHON", sys.executable)).expanduser()
REFERENCES = Path(os.environ.get("TERMITE_REFERENCES", DATA / "reference_snapshots.json")).expanduser()
DARWIN = sys.platform == "darwin"


def reference_precision(model, *, requested_dtype=None, default_policy="library_default", override_reason=None) -> dict:
    import precision
    return precision.reference_precision(model, requested_dtype=requested_dtype,
                                         default_policy=default_policy, override_reason=override_reason)


def native_precision(binary, model, models_dir=None) -> dict:
    import precision
    result = precision.native_precision(Path(binary), model, Path(models_dir or MODELS_DIR), ANTFLY_BACKEND)
    result["environment_overrides"] = {key: public_path(value) if os.path.isabs(value) else value
                                       for key, value in result["environment_overrides"].items()}
    return result


def api_error(body) -> dict:
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return {key: public_error(error[key]) if key == "message" else str(error[key])[:1000]
                for key in ("code", "message") if error.get(key) is not None}
    return {"message": public_error(error)} if isinstance(error, str) else {}


def call_refused(row: dict) -> bool:
    """A structured request rejection, not a transport/server error or universal support claim."""
    return row.get("ok") is False and row.get("status") in {400, 413, 422} and bool(row.get("response_error"))


class HttpResponseError(RuntimeError):
    """A received HTTP failure; status alone does not establish request refusal."""

    def __init__(self, status: int, body):
        self.status = status
        self.response_error = api_error(body)
        super().__init__(f"HTTP {status}: {self.response_error.get('message', 'request failed')}")

    def call_fields(self) -> dict:
        return {"ok": False, "status": self.status, "response_error": self.response_error,
                "error": public_error(self)}


def http_warmup(client, endpoint: str, requests, info: dict, cell: str) -> str:
    """Record actual planned warmups; do not admit timing after a hard failure."""
    observations, state = [], "complete"
    for payload in requests:
        started = time.perf_counter()
        try:
            status, body, _ = client.post(endpoint, payload)
            if status != 200:
                raise HttpResponseError(status, body)
            expected = len(payload.get("input", payload.get("inputs", [])))
            if not isinstance(body, dict) or not isinstance(body.get("data"), list) or len(body["data"]) != expected:
                raise RuntimeError("warmup returned malformed successful response")
            observation = {"ok": True, "status": status}
        except Exception as error:
            observation = error.call_fields() if isinstance(error, HttpResponseError) else {
                "ok": False, "error": public_error(error)}
            state = "refused" if call_refused(observation) else "failed"
        observation["ms"] = (time.perf_counter() - started) * 1000
        observations.append(observation)
        if state == "failed":
            info.setdefault("error", f"{cell} warmup failed: {observation['error']}")
            break
    info.setdefault("warmups", {})[cell] = {"state": state, "calls": observations}
    return state

def resolve_backend(value: str | None = None) -> tuple[str, str]:
    """Return the Antfly and PyTorch accelerator names for a requested backend."""
    requested = (value or os.environ.get("TERMITE_BACKEND", "auto")).lower()
    backend = ("metal" if DARWIN else "cuda") if requested == "auto" else requested
    if backend not in {"metal", "cuda"}:
        raise ValueError("backend must be auto, metal, or cuda")
    if backend == "metal" and not DARWIN:
        raise ValueError("the metal backend is only available on macOS")
    return backend, "mps" if backend == "metal" else "cuda"


def configure_backend(value: str | None = None) -> str:
    global ANTFLY_BACKEND, ACCEL
    ANTFLY_BACKEND, ACCEL = resolve_backend(value)
    return ANTFLY_BACKEND


ANTFLY_BACKEND, ACCEL = resolve_backend()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def public_path(path: str | Path) -> str:
    """Represent a path without exposing an absolute local directory."""
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.name


def public_command(argv: list[str | Path]) -> list[str]:
    public: list[str] = []
    for arg in argv:
        value = str(arg)
        candidate = Path(value).expanduser()
        public.append(public_path(candidate) if candidate.is_absolute() else value)
    return public


def public_error(error: object) -> str:
    """Remove absolute local paths from an exception or failure message."""
    text = repr(error) if isinstance(error, BaseException) else str(error)
    for root in (ROOT, RESULTS, DATA, MODELS_DIR, HF_HOME, CORPUS.parent, MEDIA):
        text = text.replace(str(root.expanduser()), f"<{root.name or 'path'}>")
    return re.sub(r"(?<![A-Za-z0-9:])/(?:[^\s'\"),;]+)", "<path>", text)[:1000]


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {"name": resolved.name, "size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def reference_paths() -> dict[str, str]:
    rows = json.loads(REFERENCES.read_text(encoding="utf-8"))
    paths: dict[str, str] = {}
    for row in rows:
        repo = str(row["repo"])
        if row.get("path"):
            paths[repo] = str(row["path"])
            continue
        from huggingface_hub import snapshot_download

        paths[repo] = snapshot_download(
            repo_id=repo,
            revision=str(row["revision"]),
            cache_dir=str(HF_HOME),
            local_files_only=True,
        )
    return paths


def load_corpus() -> dict[str, Any]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    minimums = {"sentences": 256, "paragraphs": 256, "pages": 128, "wiki_en": 64}
    observed = {key: len(corpus.get(key, [])) for key in (*minimums, "chapters")}
    short = {key: observed[key] for key, minimum in minimums.items() if observed[key] < minimum}
    if short or observed["chapters"] != 15:
        raise ValueError(f"corpus does not satisfy workload minimums: counts={observed}")
    sources = corpus.get("sources") or {}
    darwin = sources.get("darwin") or {}
    wikipedia = sources.get("wikipedia") or {}
    required_languages = {"en"}
    wiki_records = [
        record
        for records in wikipedia.values()
        if isinstance(records, list)
        for record in records
    ] if isinstance(wikipedia, dict) else []
    if (
        not isinstance(corpus.get("sampling"), str)
        or len(str(darwin.get("sha256", ""))) != 64
        or not isinstance(wikipedia, dict)
        or set(wikipedia) != required_languages
        or any(not isinstance(wikipedia[language], list) or not wikipedia[language] for language in required_languages)
        or any(
            not isinstance(record.get("resolved_title"), str)
            or not isinstance(record.get("pageid"), int)
            or not isinstance(record.get("lastrevid"), int)
            or not isinstance(record.get("url"), str)
            or len(str(record.get("extract_sha256", ""))) != 64
            for record in wiki_records
        )
    ):
        raise ValueError("corpus source snapshot provenance is incomplete")
    return corpus


def torch_sync(torch):
    return torch.mps.synchronize if ACCEL == "mps" else torch.cuda.synchronize


def torch_device_bytes(torch) -> dict[str, int]:
    if ACCEL == "mps":
        return {"mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()}
    return {
        "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def host_snapshot() -> dict[str, Any]:
    """Capture benchmark-relevant host state without recording a hostname."""
    load1, load5, _ = os.getloadavg()
    snapshot: dict[str, Any] = {"t": time.time(), "load1": round(load1, 2), "load5": round(load5, 2)}
    try:
        if DARWIN:
            battery = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
            snapshot["ac_power"] = "AC Power" in battery
            thermal = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
            snapshot["thermal_warning"] = "No thermal warning level has been recorded" not in thermal
            power = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
            snapshot["low_power_mode"] = any(
                line.split()[-1] == "1" for line in power.splitlines() if "lowpowermode" in line
            )
        else:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            snapshot["gpu"] = query.splitlines()
    except Exception:  # Host telemetry must not prevent a benchmark.
        pass
    return snapshot


def wait_quiet(max_load1: float | None = None, timeout_s: float = 600, log=print) -> dict[str, Any]:
    """Optionally wait for a load ceiling; with no ceiling, return immediately."""
    started = time.time()
    while True:
        snapshot = host_snapshot()
        if max_load1 is None or snapshot["load1"] <= max_load1 or time.time() - started > timeout_s:
            snapshot["waited_s"] = round(time.time() - started, 1)
            return snapshot
        log(f"host busy load1={snapshot['load1']}; waiting")
        time.sleep(15)


_LOCK = None


def exclusive_bench_lock():
    """Hold an exclusive process lock so benchmark workloads do not overlap."""
    global _LOCK
    import fcntl

    lock_dir = ROOT / ".cache"
    lock_dir.mkdir(parents=True, exist_ok=True)
    _LOCK = (lock_dir / "bench.lock").open("w", encoding="utf-8")
    started = time.time()
    fcntl.flock(_LOCK, fcntl.LOCK_EX)
    _LOCK.write(f"pid={os.getpid()}\n")
    _LOCK.flush()
    return round(time.time() - started, 2)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _SmapsNvml:
    """Linux smaps_rollup and optional per-process NVML memory reader."""

    def __init__(self):
        self.count_nvml = os.environ.get("TERMITE_LINUX_COUNT_NVML", "1") != "0"
        self.errors: list[str] = []
        self.read_failures = 0
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception as exc:
            self.nvml = None
            self.handles = []
            self.errors.append(f"NVML initialization: {exc!r}")

    def gpu_bytes(self) -> dict[int, int]:
        values: dict[int, int] = {}
        if self.nvml is None:
            return values
        for handle in self.handles:
            try:
                for process in self.nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    used = process.usedGpuMemory
                    if not isinstance(used, int) or not 0 <= used < (1 << 60):
                        raise RuntimeError(f"NVML memory unavailable for PID {process.pid}")
                    values[process.pid] = values.get(process.pid, 0) + used
            except Exception as exc:
                message = f"NVML process sampling: {exc!r}"
                if message not in self.errors:
                    self.errors.append(message)
        return values

    def read(self, pid: int, gpu: dict[int, int]):
        try:
            text = Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="utf-8")
        except OSError:
            self.read_failures += 1
            return None
        kb: dict[str, int] = {}
        for line in text.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 2 and fields[0].endswith(":"):
                kb[fields[0][:-1]] = int(fields[1]) * 1024
        anon, shared, file_backed = kb.get("Pss_Anon", 0), kb.get("Pss_Shmem", 0), kb.get("Pss_File", 0)
        device = gpu.get(pid, 0)
        footprint = anon + shared + file_backed + (device if self.count_nvml else 0)
        return {
            "fp": footprint,
            "rss": kb.get("Rss", 0),
            "lifetime_max": 0,
            "pss_anon": anon,
            "pss_shmem": shared,
            "pss_file": file_backed,
            "nvml": device,
        }


def sampled_model_paths() -> list[Path]:
    roots = {MODELS_DIR, DATA / "reference-verified"}
    roots.update(Path(value) for key in ("TERMITE_GEN_MODELS_DIR", "TERMITE_WHISPER_CPP_MODEL")
                 if (value := os.environ.get(key)))
    if REFERENCES.is_file():
        roots.update(Path(path) for path in reference_paths().values())
    pending = [path.resolve() for path in roots if path.exists()]
    observed = set()
    def unreadable(error):
        raise error
    while pending:
        root = pending.pop()
        if root in observed:
            continue
        observed.add(root)
        if root.is_dir():
            for directory, directories, files in os.walk(root, followlinks=False, onerror=unreadable):
                for name in (*directories, *files):
                    path = Path(directory) / name
                    if path.is_symlink():
                        pending.append(path.resolve(strict=True))
    return sorted(observed)


def memory_sampling_enabled() -> bool:
    return os.environ.get("TERMITE_SAMPLE_MEMORY", "0") == "1"


class Sampler:
    """Non-suspending concurrent process-tree observations; independent peaks never sum."""

    _DARWIN_EXIT_CONFIRM_SECONDS = 0.2

    def __init__(self, pid: int, out_dir: Path, tag: str, interval: float = 0.2):
        import threading
        import psutil

        self.root, self.interval, self.psutil = pid, interval, psutil
        self.out = out_dir / f"{tag}.footprint.json"
        self.timeline = (out_dir / f"{tag}.footprint.jsonl").open("w", encoding="utf-8")
        if DARWIN:
            from memory_darwin import DarwinMemoryReader
            evidence = os.environ.get("TERMITE_METAL_MAPPING_EVIDENCE")
            # Python's loaded runtime needs more than the native-only 150 ms
            # allowance. This phase is isolated from latency measurements; keep
            # the full walk bounded and reject it, never truncate, at the limit.
            self.reader = DarwinMemoryReader(sampled_model_paths(), max_seconds=2.0,
                mapping_scope="posix-mmap-top-level" if evidence else "all-model-mappings",
                mapping_scope_evidence=evidence)
        else:
            self.reader = _SmapsNvml()
        self.peak = {
            "metric_version": "darwin-ledger-mapped-storage-interval-v1" if DARWIN else "linux-pss-nvml-v1",
            "memory_peak_lower_bytes": 0, "memory_peak_upper_bytes": 0,
            "raw_process_tree_footprint_peak_bytes": 0, "peak_rss_bytes": 0,
            "samples": 0, "invalid_samples": 0, "observer_seconds": 0.0,
            "max_sample_gap_seconds": 0.0, "coverage_errors": [],
        }
        self.sample_process_count = 0
        self.processes_seen: set[int] = set()
        if DARWIN:
            self.peak.update({
                "coverage_scope": self.reader.mapping_scope,
                "coverage_evidence": self.reader.mapping_scope_evidence,
                "region_walk_budget_seconds": self.reader.max_seconds,
                "region_walk_budget_count": self.reader.max_regions,
                "exit_confirmation_budget_seconds": self._DARWIN_EXIT_CONFIRM_SECONDS,
                "discarded_exit_observations": 0,
            })
        if not DARWIN:
            self.peak.update({"nvml_available": self.reader.nvml is not None, "nvml_counted": self.reader.count_nvml})
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _pids(self) -> list[int]:
        try:
            process = self.psutil.Process(self.root)
            return [process.pid] + [child.pid for child in process.children(recursive=True)]
        except self.psutil.NoSuchProcess:
            return []
        except self.psutil.Error as error:
            self.peak["invalid_samples"] += 1
            self.peak["coverage_errors"].append(public_error(error))
            return []

    def _run(self) -> None:
        try:
            self._sample_loop()
        except Exception as error:
            self.peak["invalid_samples"] += 1
            self.peak["coverage_errors"].append(public_error(error))

    def _darwin_observation(self, pid: int) -> tuple[dict | None, bool, dict | None]:
        """Discard only typed exit races backed by an owned process identity.

        ESRCH alone is insufficient: an unreadable live task, an exec, or an
        unconfirmed identity must still invalidate coverage. Retain all timeout,
        permission and observer-control errors even if the process later exits.
        """
        process = None
        try:
            process = self.psutil.Process(pid)
            create_time = process.create_time()  # Cache identity before the observation.
        except self.psutil.NoSuchProcess:
            return None, False, {"pid": pid, "evidence": "absent_before_read"}
        except self.psutil.Error:
            process = None
        value = self.reader.read(pid)
        unavailable = (self.reader.last_target_unavailable if value is None
                       else value.get("target_unavailable", False))
        if unavailable and process is not None:
            # XNU can make the task's VM map unavailable before libproc exposes
            # a zombie/absent process. Resolve only that typed race, without
            # rereading memory, signalling the process, or reaping Popen's child.
            started = time.monotonic()
            deadline = started + self._DARWIN_EXIT_CONFIRM_SECONDS
            confirmation = {"budget_seconds": self._DARWIN_EXIT_CONFIRM_SECONDS,
                            "checks": 0, "initial_state": None}
            evidence = None
            while True:
                confirmation["checks"] += 1
                state = "live"
                try:
                    if not process.is_running():
                        evidence = "original_process_exited_or_pid_reused"
                    elif process.status() == self.psutil.STATUS_ZOMBIE:
                        evidence = "original_process_is_zombie"
                except self.psutil.NoSuchProcess:
                    evidence = "original_process_no_longer_exists"
                except self.psutil.Error as error:
                    state = "unconfirmed"
                    confirmation["error"] = public_error(error)
                state = evidence or state
                if confirmation["initial_state"] is None:
                    confirmation["initial_state"] = state
                confirmation["final_state"] = state
                remaining = deadline - time.monotonic()
                if evidence or state == "unconfirmed" or remaining <= 0:
                    break
                time.sleep(min(0.01, remaining))
            confirmation["elapsed_seconds"] = time.monotonic() - started
            if evidence:
                return None, False, {
                    "pid": pid, "create_time": create_time,
                    "evidence": evidence, "exit_confirmation": confirmation,
                    "errors": [self.reader.last_error] if value is None else value["errors"],
                    "incomplete_observation": value,
                }
            if value is not None:
                value["exit_confirmation"] = {"create_time": create_time, **confirmation}
        return value, value is None, None

    def _sample_loop(self) -> None:
        started = previous = time.perf_counter()
        while not self.stop_event.is_set():
            now = time.perf_counter()
            self.peak["max_sample_gap_seconds"] = max(self.peak["max_sample_gap_seconds"], now - previous)
            previous = now
            gpu = {} if DARWIN else self.reader.gpu_bytes()
            pids = self._pids()
            self.sample_process_count = len(pids)
            values = []
            missing = []
            discarded = []
            for pid in pids:
                if DARWIN:
                    value, unreadable, exit_observation = self._darwin_observation(pid)
                    if exit_observation is not None:
                        discarded.append(exit_observation)
                    if unreadable:
                        missing.append(pid)
                        if self.reader.last_error:
                            self.peak["coverage_errors"] = sorted(set(
                                self.peak["coverage_errors"] + [self.reader.last_error]))
                else:
                    value = self.reader.read(pid, gpu)
                if value is not None:
                    value["pid"] = pid
                    self.processes_seen.add(pid)
                    values.append(value)
                elif not DARWIN and self.psutil.pid_exists(pid):
                    try:
                        if self.psutil.Process(pid).status() != self.psutil.STATUS_ZOMBIE:
                            missing.append(pid)
                    except self.psutil.NoSuchProcess:
                        pass
            if values:
                raw = sum(value["fp"] for value in values)
                rss = sum(value["rss"] for value in values)
                if DARWIN:
                    from memory_darwin import combine_samples
                    bounds = combine_samples(values)
                else:
                    bounds = {"valid": True, "lower_bytes": raw, "upper_bytes": raw}
                bounds["valid"] &= not missing
                row = {"t": now - started, "raw_footprint_bytes": raw, "rss_bytes": rss,
                       "memory": bounds, "processes": values, "missing_live_pids": missing,
                       "discarded_exit_observations": discarded}
                self.peak["raw_process_tree_footprint_peak_bytes"] = max(self.peak["raw_process_tree_footprint_peak_bytes"], raw)
                self.peak["peak_rss_bytes"] = max(self.peak["peak_rss_bytes"], rss)
                self.peak["samples"] += 1
                if bounds["valid"]:
                    for endpoint in ("lower", "upper"):
                        key = f"memory_peak_{endpoint}_bytes"
                        self.peak[key] = max(self.peak[key], bounds[f"{endpoint}_bytes"])
                else:
                    self.peak["invalid_samples"] += 1
                errors = [error for value in values for error in value.get("errors", [])]
                if missing:
                    errors.append("live process unreadable")
                self.peak["coverage_errors"] = sorted(set(self.peak["coverage_errors"] + errors))
                self._write_observation(row, now)
            elif missing:
                self.peak["invalid_samples"] += 1
                self.peak["coverage_errors"] = sorted(set(self.peak["coverage_errors"] + ["live process unreadable"]))
            if discarded:
                self.peak["discarded_exit_observations"] += len(discarded)
            if not values and (missing or discarded):
                self._write_observation({
                    "t": now - started, "processes": [], "missing_live_pids": missing,
                    "discarded_exit_observations": discarded,
                    "observation_status": "no_complete_process_observation",
                }, now)
            self.stop_event.wait(max(0, self.interval - (time.perf_counter() - now)))

    def _write_observation(self, row: dict, started: float) -> None:
        elapsed = time.perf_counter() - started
        self.peak["observer_seconds"] += elapsed
        row["observer_seconds"] = elapsed
        self.timeline.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.timeline.flush()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        timeout = 5
        if DARWIN:
            timeout += self.sample_process_count * (
                self.reader.max_seconds + self._DARWIN_EXIT_CONFIRM_SECONDS)
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise RuntimeError("memory sampler failed to stop")
        self.timeline.close()
        self.peak["valid"] = self.peak["samples"] > 0 and self.peak["invalid_samples"] == 0
        self.peak["interval_seconds"] = self.interval
        self.peak["root_pid"] = self.root
        self.peak["processes_seen"] = sorted(self.processes_seen)
        self.peak["method"] = ("concurrent ledger/mapped-storage interval; overlap unknown"
                               if DARWIN else "concurrent process-tree PSS plus discrete NVML")
        if not DARWIN and not self.reader.count_nvml:
            self.peak["metric_version"] = "linux-pss-v1"
            self.peak["method"] = "concurrent process-tree PSS; NVML excluded by operator"
        if not DARWIN:
            self.peak["nvml_errors"] = self.reader.errors
            self.peak["read_failures"] = self.reader.read_failures
            self.peak["valid"] &= not self.reader.count_nvml or (
                self.reader.nvml is not None and not self.reader.errors)
        if not self.peak["valid"]:
            self.peak["memory_peak_lower_bytes"] = self.peak["memory_peak_upper_bytes"] = None
        write_json(self.out, self.peak)
        return dict(self.peak)


def wait_http(proc, port: int, path: str, timeout_s: float, log_name: str, started: float) -> float:
    while time.perf_counter() - started < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}; see {Path(log_name).name}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", path)
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return round(time.perf_counter() - started, 3)
        except OSError:
            pass
        time.sleep(0.05)
    raise RuntimeError(f"server did not become ready; see {Path(log_name).name}")


def stop_proc(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


class AntflyServer:
    def __init__(
        self,
        binary: Path,
        out_dir: Path,
        tag: str,
        config: dict | None = None,
        env: dict | None = None,
        extra_args: list[str] | None = None,
        models_dir: Path | None = None,
    ):
        self.binary, self.out_dir, self.tag = Path(binary), out_dir, tag
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.port = free_port()
        self.config = config or {}
        self.env = env or {}
        self.extra_args = extra_args or []
        self.proc = None
        self.sampler = None
        self.log = None

    def start(self, sample: bool = True, timeout_s: float = 180) -> dict[str, Any]:
        config_path = self.out_dir / f"{self.tag}.config.json"
        config: dict[str, Any] = {"inference": {"keep_alive": "0"}}
        for key, value in self.config.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
        config_path.write_text(json.dumps(config, separators=(",", ":")) + "\n", encoding="utf-8")
        env = dict(os.environ)
        env.setdefault("ANTFLY_LOG_LEVEL", "info")
        env["ANTFLY_INFERENCE_REQUIRED_BACKEND"] = ANTFLY_BACKEND
        env["ANTFLY_INFERENCE_WORKER"] = str(self.binary)
        env.update(self.env)
        argv = [
            str(self.binary),
            "inference",
            "run",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--models-dir",
            str(self.models_dir),
            "--config",
            str(config_path),
            *self.extra_args,
        ]
        self.log = (self.out_dir / f"{self.tag}.server.log").open("w", encoding="utf-8")
        started = time.perf_counter()
        try:
            self.proc = subprocess.Popen(
                argv,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            if sample and memory_sampling_enabled():
                self.sampler = Sampler(self.proc.pid, self.out_dir, self.tag)
            ready = wait_http(self.proc, self.port, "/healthz", timeout_s, self.log.name, started)
        except Exception:
            self.stop()
            raise
        return {
            "ready_s": ready,
            "command": public_command(argv),
            "config": config,
            "required_backend": ANTFLY_BACKEND,
        }

    def stop(self) -> dict[str, Any]:
        try:
            return self.sampler.stop() if self.sampler else {}
        finally:
            self.sampler = None
            stop_proc(self.proc)
            if self.log is not None and not self.log.closed:
                self.log.close()


class Client:
    """Persistent keep-alive JSON client."""

    def __init__(self, port: int, timeout: float = 900):
        self.port, self.timeout = port, timeout
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)

    def post(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any], int]:
        data = json.dumps(body, ensure_ascii=False).encode()
        for attempt in range(2):
            try:
                self.conn.request("POST", path, body=data, headers={"content-type": "application/json"})
                response = self.conn.getresponse()
                raw = response.read()
                value = json.loads(raw) if raw else {}
                return response.status, value, len(raw)
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                self.conn.close()
                self.conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
                if attempt:
                    raise
        raise RuntimeError("unreachable")


class JsonlWriter:
    def __init__(self, path: Path):
        self.file = path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self.file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def python_arm_env() -> dict[str, str]:
    return dict(
        os.environ,
        HF_HOME=str(HF_HOME),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        TERMITE_BACKEND=ANTFLY_BACKEND,
    )


def run_python_arm(script: str, argv: list[str], out: Path, tag: str) -> dict[str, Any]:
    """Run an upstream implementation, retaining its result if cleanup fails."""
    log = out / f"{tag}.py.log"
    sampler = None
    process = None
    stdout = ""
    footprint, cleanup_errors = {}, []
    with log.open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(
                [str(REF_PY), script, "pyarm", *argv],
                stdout=subprocess.PIPE,
                stderr=stderr,
                env=python_arm_env(),
                text=True,
                start_new_session=True,
            )
            if memory_sampling_enabled():
                sampler = Sampler(process.pid, out, tag)
            stdout, _ = process.communicate()
        finally:
            try:
                footprint = sampler.stop() if sampler else {}
            except Exception as error:
                message = public_error(error)
                cleanup_errors.append(message)
                footprint = {**sampler.peak, "valid": False,
                             "memory_peak_lower_bytes": None, "memory_peak_upper_bytes": None,
                             "coverage_errors": [*sampler.peak.get("coverage_errors", []), message]}
            finally:
                if process is not None and process.poll() is None:
                    try:
                        stop_proc(process)
                    except Exception as error:
                        cleanup_errors.append(public_error(error))
    try:
        info = json.loads(stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "error": f"python arm rc={None if process is None else process.returncode}; see {log.name}",
            "footprint": footprint,
            "rc": None if process is None else process.returncode,
            **({"cleanup_errors": cleanup_errors} if cleanup_errors else {}),
        }
    info["footprint"] = footprint
    info["rc"] = process.returncode
    if process.returncode != 0 and "error" not in info:
        info["error"] = f"python arm exited {process.returncode}; see {log.name}"
    if cleanup_errors:
        info["cleanup_errors"] = cleanup_errors
        messages = [info["error"]] if info.get("error") else []
        messages.append("cleanup failed: " + "; ".join(cleanup_errors))
        info["error"] = "; ".join(messages)
    return info


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_output(out: Path) -> None:
    """Create a fresh output directory, refusing to append to prior measurements."""
    measurement_files = (
        "calls.jsonl",
        "lifecycles.jsonl",
        "outputs.jsonl",
        "vectors.jsonl",
        "agreement.jsonl",
        "summary.json",
    )
    stale = [name for name in measurement_files if (out / name).exists()]
    if stale:
        raise RuntimeError(f"output contains prior measurements: {', '.join(stale)}")
    out.mkdir(parents=True, exist_ok=True)


def benchmark_complete(
    out: Path,
    expected_arms: set[str],
    lifecycles: int,
    expected_calls_per_lifecycle: int | None = None,
) -> bool:
    """Require complete execution records, allowing explicitly recorded request refusals."""
    lifecycle_rows = load_jsonl(out / "lifecycles.jsonl")
    call_rows = load_jsonl(out / "calls.jsonl")
    expected = {(arm, lifecycle) for arm in expected_arms for lifecycle in range(lifecycles)}
    observed = {
        (str(row.get("arm")), int(row.get("lc", -1)))
        for row in lifecycle_rows
        if not row.get("error") and row.get("rc", 0) == 0
    }
    call_groups: dict[tuple[str, int], int] = {}
    for row in call_rows:
        key = (str(row.get("arm")), int(row.get("lc", -1)))
        call_groups[key] = call_groups.get(key, 0) + 1
    if expected_calls_per_lifecycle is not None and any(
        call_groups.get(key) != expected_calls_per_lifecycle for key in expected
    ):
        return False
    return bool(call_rows) and expected <= observed and expected <= set(call_groups) and all(
        row.get("ok") is True or call_refused(row) for row in call_rows
    )
