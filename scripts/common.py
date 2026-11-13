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


class _RusageV4:
    """macOS proc_pid_rusage reader that does not suspend the target."""

    def __init__(self):
        import ctypes

        self.ct = ctypes
        self.lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        self.buf = (ctypes.c_uint64 * 64)()

    def read(self, pid: int):
        if self.lib.proc_pid_rusage(pid, 4, self.ct.byref(self.buf)) != 0:
            return None
        fields = list(self.buf)[2:]
        return {"fp": fields[7], "rss": fields[6], "lifetime_max": fields[28]}


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


class Sampler:
    """Sample summed process-tree footprint every 0.2 seconds."""

    def __init__(self, pid: int, out_dir: Path, tag: str, interval: float = 0.2):
        import threading
        import psutil

        self.root, self.interval, self.psutil = pid, interval, psutil
        self.out = out_dir / f"{tag}.footprint.json"
        self.timeline = (out_dir / f"{tag}.footprint.jsonl").open("w", encoding="utf-8")
        self.reader = _RusageV4() if DARWIN else _SmapsNvml()
        self.peak: dict[str, Any] = {
            "process_tree_footprint_peak_bytes": 0,
            "peak_rss_bytes": 0,
            "lifetime_max_footprint_bytes": 0,
            "samples": 0,
        }
        if not DARWIN:
            self.peak.update(
                {
                    "pss_anon_peak_bytes": 0,
                    "pss_shmem_peak_bytes": 0,
                    "pss_file_peak_bytes": 0,
                    "nvml_peak_bytes": 0,
                    "nvml_available": self.reader.nvml is not None,
                    "nvml_counted": self.reader.count_nvml,
                }
            )
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _pids(self) -> list[int]:
        try:
            process = self.psutil.Process(self.root)
            return [process.pid] + [child.pid for child in process.children(recursive=True)]
        except Exception:
            return []

    def _run(self) -> None:
        started = time.time()
        parts = ("pss_anon", "pss_shmem", "pss_file", "nvml")
        while not self.stop_event.is_set():
            footprint = rss = 0
            components = dict.fromkeys(parts, 0)
            gpu = {} if DARWIN else self.reader.gpu_bytes()
            for pid in self._pids():
                value = self.reader.read(pid) if DARWIN else self.reader.read(pid, gpu)
                if not value:
                    continue
                footprint += value["fp"]
                rss += value["rss"]
                for key in parts:
                    components[key] += value.get(key, 0)
                self.peak["lifetime_max_footprint_bytes"] = max(
                    self.peak["lifetime_max_footprint_bytes"], value["lifetime_max"]
                )
            if footprint:
                self.peak["process_tree_footprint_peak_bytes"] = max(
                    self.peak["process_tree_footprint_peak_bytes"], footprint
                )
                self.peak["peak_rss_bytes"] = max(self.peak["peak_rss_bytes"], rss)
                if not DARWIN:
                    for key in parts:
                        peak_key = f"{key}_peak_bytes"
                        self.peak[peak_key] = max(self.peak[peak_key], components[key])
                self.peak["samples"] += 1
                row = {"t": round(time.time() - started, 2), "fp": footprint, "rss": rss}
                if not DARWIN:
                    row.update(components)
                self.timeline.write(json.dumps(row, separators=(",", ":")) + "\n")
                self.timeline.flush()
            self.stop_event.wait(self.interval)

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("memory sampler failed to stop")
        self.timeline.close()
        method = "proc_pid_rusage v4 phys_footprint" if DARWIN else "smaps_rollup PSS"
        if not DARWIN and self.reader.count_nvml:
            method += " plus NVML"
        self.peak["method"] = method + "; 0.2 s polling; summed process tree"
        self.peak["valid"] = self.peak["samples"] > 0
        if not DARWIN:
            self.peak["nvml_errors"] = self.reader.errors
            self.peak["read_failures"] = self.reader.read_failures
            self.peak["valid"] &= not self.reader.count_nvml or (
                self.reader.nvml is not None and not self.reader.errors
            )
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
            if sample:
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
        stop_proc(self.proc)
        try:
            return self.sampler.stop() if self.sampler else {}
        finally:
            self.sampler = None
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
    """Run an upstream implementation in a fresh sampled process."""
    log = out / f"{tag}.py.log"
    sampler = None
    process = None
    stdout = ""
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
            sampler = Sampler(process.pid, out, tag)
            stdout, _ = process.communicate()
        finally:
            if process is not None and process.poll() is None:
                stop_proc(process)
            footprint = sampler.stop() if sampler else {}
    try:
        info = json.loads(stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "error": f"python arm rc={None if process is None else process.returncode}; see {log.name}",
            "footprint": footprint,
            "rc": None if process is None else process.returncode,
        }
    info["footprint"] = footprint
    info["rc"] = process.returncode
    if process.returncode != 0 and "error" not in info:
        info["error"] = f"python arm exited {process.returncode}; see {log.name}"
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
    """Require every arm/lifecycle and every expected measured call to have succeeded."""
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
        row.get("ok") is True for row in call_rows
    )
