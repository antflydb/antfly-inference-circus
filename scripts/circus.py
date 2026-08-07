#!/usr/bin/env python3
"""Manual, supplied-binary Antfly inference benchmark campaign runner."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "matrix.json"
DEFAULT_FIXTURES = ROOT / "fixtures" / "cases.json"
DEFAULT_MODELS = Path.home() / ".antfly" / "inference" / "models"
CATEGORY_ORDER = [
    "generation",
    "dense embedding",
    "sparse embedding",
    "multimodal embedding",
    "reranking",
    "chunking",
    "extraction",
    "reading / OCR",
    "transcription",
]


class CampaignError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CampaignError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    if path.is_file():
        return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    files = []
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = str(child.relative_to(path))
        checksum = sha256_file(child)
        size = child.stat().st_size
        files.append({"path": relative, "size_bytes": size, "sha256": checksum})
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(checksum.encode())
        digest.update(b"\n")
    return {"path": str(path), "sha256": digest.hexdigest(), "files": files}


def resolve_command(value: str) -> str | None:
    path = Path(value).expanduser()
    if path.parent != Path(".") or path.is_absolute():
        return str(path.resolve()) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(value)


def antfly_binary(explicit: str | None) -> Path | None:
    value = explicit or os.environ.get("ANTFLY_BIN", "")
    if not value:
        return None
    resolved = resolve_command(value)
    return Path(resolved) if resolved else None


def command_output(args: list[str], timeout: int = 15) -> str:
    try:
        proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return proc.stdout.strip()


def validate_matrix(matrix: dict[str, Any]) -> None:
    if matrix.get("schema_version") != 1:
        raise CampaignError("matrix schema_version must be 1")
    if not isinstance(matrix.get("antfly_contract"), str) or not matrix["antfly_contract"]:
        raise CampaignError("matrix must name its antfly_contract")
    profiles = matrix.get("profiles")
    providers = matrix.get("providers")
    benchmarks = matrix.get("benchmarks")
    if not isinstance(profiles, dict) or not isinstance(providers, dict) or not isinstance(benchmarks, list):
        raise CampaignError("matrix must define profiles, providers, and benchmarks")
    ids: set[str] = set()
    for bench in benchmarks:
        bench_id = bench.get("id")
        if not isinstance(bench_id, str) or not bench_id or bench_id in ids:
            raise CampaignError(f"invalid or duplicate benchmark id: {bench_id!r}")
        ids.add(bench_id)
        for required in ("category", "model", "role", "profiles", "providers", "fixture"):
            if not bench.get(required):
                raise CampaignError(f"{bench_id}: missing {required}")
        unknown_profiles = set(bench["profiles"]) - set(profiles)
        unknown_providers = set(bench["providers"]) - set(providers)
        if unknown_profiles or unknown_providers:
            raise CampaignError(f"{bench_id}: unknown profiles/providers: {sorted(unknown_profiles | unknown_providers)}")


def select_benchmarks(matrix: dict[str, Any], profile: str, category: str | None) -> list[dict[str, Any]]:
    if profile not in matrix["profiles"]:
        raise CampaignError(f"unknown profile {profile!r}")
    selected = [bench for bench in matrix["benchmarks"] if profile in bench["profiles"]]
    if category:
        selected = [bench for bench in selected if bench["category"] == category]
        if not selected:
            raise CampaignError(f"profile {profile!r} has no category {category!r}")
    return selected


def providers_for(bench: dict[str, Any], profile: str) -> list[str]:
    return ["antfly"] if profile == "smoke" else list(bench["providers"])


def model_dir(models_dir: Path, bench: dict[str, Any]) -> Path:
    # rc.24 writes the current flat layout, while older installs may retain the
    # role-prefixed layout. Accept both without moving operator-owned models.
    flat = models_dir / bench["model"]
    role_prefixed = models_dir / bench["role"] / bench["model"]
    return flat if flat.exists() or not role_prefixed.exists() else role_prefixed


def named_model_dir(models_dir: Path, role: str, model: str) -> Path:
    return model_dir(models_dir, {"role": role, "model": model})


def artifact_path(models_dir: Path, bench: dict[str, Any]) -> Path:
    base = model_dir(models_dir, bench)
    artifact = bench.get("artifact") or ""
    return base / artifact if artifact else base


def module_available(python: str, module: str) -> bool:
    try:
        proc = subprocess.run(
            [python, "-c", f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return proc.returncode == 0


def cached_huggingface_model(python: str, model: str) -> str | None:
    code = (
        "from huggingface_hub import snapshot_download; "
        f"print(snapshot_download(repo_id={model!r}, local_files_only=True))"
    )
    try:
        proc = subprocess.run(
            [python, "-c", code],
            stdout=subprocess.PIPE,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    path = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return path if path and Path(path).exists() else None


def reference_cache_model(bench: dict[str, Any], provider_id: str) -> str | None:
    if provider_id in {"onnxruntime", "sentence-transformers", "gliner2", "transformers"}:
        return str(bench.get("reference_model") or bench["model"])
    if provider_id == "faster-whisper":
        return "Systran/faster-whisper-tiny"
    if provider_id == "tokenizers":
        if bench["model"] == "fixed_bert":
            return "google-bert/bert-base-uncased"
        if bench["model"] == "fixed_bpe":
            return "openai-community/gpt2"
        return str(bench["model"])
    return None


def configured_provider_model(bench: dict[str, Any], provider_id: str) -> str | None:
    env_key = re.sub(r"[^A-Za-z0-9]", "_", bench["id"]).upper()
    if provider_id == "ollama":
        return os.environ.get(f"OLLAMA_MODEL_{env_key}") or os.environ.get("OLLAMA_MODEL")
    if provider_id == "vllm-metal":
        return os.environ.get(f"VLLM_MODEL_{env_key}") or os.environ.get("VLLM_MODEL")
    if provider_id == "whisper.cpp":
        return os.environ.get("WHISPER_CPP_MODEL")
    return reference_cache_model(bench, provider_id)


def provider_provenance(matrix: dict[str, Any], provider_ids: set[str], python: str) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    package_names = {
        "onnxruntime": "onnxruntime",
        "sentence_transformers": "sentence-transformers",
        "gliner2": "gliner2",
        "transformers": "transformers",
        "PIL": "Pillow",
        "faster_whisper": "faster-whisper",
    }
    for provider_id in sorted(provider_ids):
        if provider_id == "antfly":
            continue
        provider = matrix["providers"][provider_id]
        if provider["kind"] == "command" or provider.get("command_env"):
            value = os.environ.get(provider.get("command_env", ""), provider.get("command_default", ""))
            resolved = resolve_command(value)
            versions[provider_id] = {"command": resolved or value, "version": command_output([resolved or value, "--version"]) if value else ""}
            continue
        modules = provider.get("python_modules", [])
        packages = [package_names.get(module, module) for module in modules]
        code = "import importlib.metadata,json; print(json.dumps({p:importlib.metadata.version(p) for p in " + repr(packages) + "}))"
        proc = subprocess.run([python, "-c", code], text=True, capture_output=True, check=False)
        versions[provider_id] = {"python": python, "packages": json.loads(proc.stdout) if proc.returncode == 0 else {}, "error": proc.stderr.strip() if proc.returncode else ""}
    return versions


def prerequisite_report(
    matrix: dict[str, Any],
    fixtures: dict[str, Any],
    profile: str,
    category: str | None,
    antfly: Path | None,
    models_dir: Path,
    python: str,
) -> dict[str, Any]:
    selected = select_benchmarks(matrix, profile, category)
    missing: list[str] = []
    warnings: list[str] = []
    if antfly is None:
        missing.append("ANTFLY_BIN (executable path)")
    else:
        version = command_output([str(antfly), "--version"])
        expected_contract = str(matrix["antfly_contract"])
        if expected_contract.removeprefix("v") not in version:
            warnings.append(f"binary reports {version!r}; request shapes are certified against {expected_contract}")
    for bench in selected:
        fixture = fixtures.get(bench["fixture"], {})
        for key, value in fixture.items():
            if (key == "audio" or key == "image" or key.endswith("_image")) and isinstance(value, str):
                if not (ROOT / value).is_file():
                    missing.append(f"fixture {value} (run make fixtures)")
        if bench["model"].startswith("fixed_"):
            continue
        path = artifact_path(models_dir, bench)
        if not path.exists():
            missing.append(f"model {path}")
        draft_model = bench.get("draft_model")
        if draft_model:
            draft_path = named_model_dir(models_dir, "generators", str(draft_model))
            if not draft_path.exists():
                missing.append(f"draft model {draft_path}")
    provider_ids = {provider for bench in selected for provider in providers_for(bench, profile)}
    for provider_id in sorted(provider_ids - {"antfly"}):
        provider = matrix["providers"][provider_id]
        if provider["kind"] == "command":
            value = os.environ.get(provider.get("command_env", ""), provider.get("command_default", ""))
            if not resolve_command(value):
                missing.append(f"provider command {provider_id}: {value}")
        elif provider["kind"] == "reference":
            for module in provider.get("python_modules", []):
                if not module_available(python, module):
                    missing.append(f"Python module {module} for {provider_id} ({python})")
            command_env = provider.get("command_env")
            if command_env:
                value = os.environ.get(command_env, provider.get("command_default", ""))
                if not resolve_command(value):
                    missing.append(f"provider command {provider_id}: {value}")
    cache_checks: set[str] = set()
    for bench in selected:
        env_key = re.sub(r"[^A-Za-z0-9]", "_", bench["id"]).upper()
        for provider_id in providers_for(bench, profile):
            if provider_id == "ollama" and not (os.environ.get(f"OLLAMA_MODEL_{env_key}") or os.environ.get("OLLAMA_MODEL")):
                missing.append(f"provider model OLLAMA_MODEL_{env_key}")
            if provider_id == "vllm-metal" and not (os.environ.get(f"VLLM_MODEL_{env_key}") or os.environ.get("VLLM_MODEL")):
                missing.append(f"provider model VLLM_MODEL_{env_key}")
            if provider_id == "whisper.cpp":
                whisper_model = Path(os.environ.get("WHISPER_CPP_MODEL", "")).expanduser()
                if not whisper_model.is_file():
                    missing.append("provider model WHISPER_CPP_MODEL")
            cache_model = reference_cache_model(bench, provider_id)
            if cache_model and cache_model not in cache_checks:
                cache_checks.add(cache_model)
                if cached_huggingface_model(python, cache_model) is None:
                    missing.append(f"offline Hugging Face model {cache_model} for {provider_id}")
    if profile == "full" and os.environ.get("QUIET_HOST_ACK") != "1":
        warnings.append("QUIET_HOST_ACK=1 not set; memory is collected but marked diagnostic")
    return {"status": "ok" if not missing else "failed", "profile": profile, "missing": sorted(set(missing)), "warnings": warnings}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def tree_rss_kb(root_pid: int) -> tuple[int, list[dict[str, Any]]]:
    proc = subprocess.run(["ps", "-axo", "pid=,ppid=,rss=,comm="], text=True, capture_output=True, check=False)
    rows: list[tuple[int, int, int, str]] = []
    for line in proc.stdout.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) < 3:
            continue
        try:
            rows.append((int(fields[0]), int(fields[1]), int(fields[2]), fields[3] if len(fields) > 3 else ""))
        except ValueError:
            continue
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid, _, _ in rows:
            if ppid in tree and pid not in tree:
                tree.add(pid)
                changed = True
    seen = [{"pid": pid, "ppid": ppid, "rss_kb": rss, "command": command} for pid, ppid, rss, command in rows if pid in tree]
    return sum(row["rss_kb"] for row in seen), seen


def system_demand_kb() -> int | None:
    if platform.system() != "Darwin":
        return None
    proc = subprocess.run(["vm_stat"], text=True, capture_output=True, check=False)
    if proc.returncode:
        return None
    page_match = re.search(r"page size of (\d+) bytes", proc.stdout)
    page_size = int(page_match.group(1)) if page_match else 16384
    pages = 0
    for label in ("Pages wired down", "Anonymous pages", "Pages occupied by compressor"):
        match = re.search(rf"{re.escape(label)}:\s+(\d+)\.", proc.stdout)
        if not match:
            return None
        pages += int(match.group(1))
    return pages * page_size // 1024


class MemorySampler:
    def __init__(self, pid: int, path: Path):
        self.pid = pid
        self.path = path
        self.stop_event = threading.Event()
        self.baseline = system_demand_kb()
        self.samples: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
        rss = max((sample["rss_kb"] for sample in self.samples), default=0)
        demand = max((sample.get("system_demand_delta_kb", 0) for sample in self.samples), default=0)
        return {
            "peak_rss_mb": rss / 1024,
            "peak_system_demand_delta_mb": demand / 1024,
            "peak_memory_mb": max(rss, demand) / 1024,
            "sample_count": len(self.samples),
            "timeline": str(self.path.relative_to(self.path.parents[1])),
        }

    def _run(self) -> None:
        while not self.stop_event.wait(0.2):
            rss, processes = tree_rss_kb(self.pid)
            demand = system_demand_kb()
            sample: dict[str, Any] = {"elapsed_ms": len(self.samples) * 200, "rss_kb": rss, "processes": processes}
            if demand is not None and self.baseline is not None:
                sample["system_demand_delta_kb"] = max(0, demand - self.baseline)
            self.samples.append(sample)


def http_json(url: str, body: dict[str, Any] | None = None, timeout: float = 600) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise CampaignError(f"HTTP {exc.code} from {url}: {detail[:1000]}") from exc
    except OSError as exc:
        raise CampaignError(f"request to {url} failed: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CampaignError(f"invalid JSON from {url}: {raw[:500]!r}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"expected object from {url}")
    return value


def wait_ready(base_url: str, process: subprocess.Popen[str], timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CampaignError(f"Antfly inference exited before readiness with status {process.returncode}")
        try:
            http_json(base_url + "/ai/v1/models", timeout=2)
            return
        except CampaignError:
            time.sleep(0.25)
    raise CampaignError("Antfly inference did not become ready within 90 seconds")


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def antfly_request(bench: dict[str, Any], fixture: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    model = bench["model"]
    category = bench["category"]
    if category == "generation":
        body = {
            "model": model,
            "messages": [{"role": "user", "content": fixture["prompt"]}],
            "max_tokens": fixture["max_tokens"],
            "temperature": 0,
            "stream": False,
        }
        if bench.get("draft_model"):
            body["draft_model"] = bench["draft_model"]
            body["speculative_k"] = int(bench.get("speculative_k", 2))
        return "/ai/v1/generate", body
    if category in {"dense embedding", "sparse embedding"}:
        return "/ai/v1/embed", {"model": model, "input": fixture["texts"]}
    if category == "multimodal embedding":
        return "/ai/v1/embed", {
            "model": model,
            "input": [
                fixture["text"],
                {"type": "image_url", "image_url": {"url": data_uri(ROOT / fixture["matching_image"], "image/png")}},
                {"type": "image_url", "image_url": {"url": data_uri(ROOT / fixture["mismatching_image"], "image/png")}},
            ],
        }
    if category == "reranking":
        return "/ai/v1/rerank", {"model": model, "query": fixture["query"], "prompts": fixture["documents"]}
    if category == "chunking":
        chunk_model = "fixed" if model.startswith("fixed_") else model
        return "/ai/v1/chunk", {
            "input": fixture["text"],
            "config": {"model": chunk_model, "text": {"target_tokens": fixture["target_tokens"], "overlap_tokens": fixture["overlap_tokens"]}},
        }
    if category == "extraction":
        structures = {
            name: {
                "fields": {
                    field.split("::", 1)[0]: "list" if "list" in field.split("::")[1:] else "str"
                    for field in fields
                }
            }
            for name, fields in fixture["schema"].items()
        }
        return "/ai/v1/extract", {
            "model": model,
            "inputs": [{"content": text} for text in fixture["texts"]],
            "schema": {"structures": structures},
            "options": {"include_spans": True},
        }
    if category == "reading / OCR":
        return "/ai/v1/read", {"model": model, "images": [{"url": data_uri(ROOT / fixture["image"], "image/png")}], "prompt": "<OCR>"}
    if category == "transcription":
        audio = base64.b64encode((ROOT / fixture["audio"]).read_bytes()).decode()
        return "/ai/v1/transcribe", {"model": model, "audio": audio, "language": "en"}
    raise CampaignError(f"unsupported category {category}")


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def quality_result(category: str, response: dict[str, Any], fixture: dict[str, Any]) -> tuple[bool, str, float | None, str | None]:
    if category == "generation":
        choices = response.get("choices") or []
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        ok = bool(content.strip()) and all(term.lower() in content.lower() for term in fixture.get("required_terms", []))
        return ok, "nonempty required-term completion", 1.0 if ok else 0.0, "pass"
    if category == "dense embedding":
        vectors = [item.get("embedding") for item in response.get("data", []) if isinstance(item, dict)]
        if len(vectors) < 3:
            return False, "semantic ordering", None, None
        relevant = cosine(vectors[0], vectors[1])
        irrelevant = cosine(vectors[0], vectors[2])
        return relevant > irrelevant, "relevant cosine margin", relevant - irrelevant, "cosine"
    if category == "sparse embedding":
        vectors = [item.get("embedding") for item in response.get("data", []) if isinstance(item, dict)]
        nonzero = sum(len(vector.get("indices", vector.get("values", []))) for vector in vectors if isinstance(vector, dict))
        ok = len(vectors) == len(fixture["texts"]) and nonzero >= fixture["minimum_nonzero"]
        return ok, "nonzero sparse entries", float(nonzero), "entries"
    if category == "multimodal embedding":
        vectors = [item.get("embedding") for item in response.get("data", []) if isinstance(item, dict)]
        if len(vectors) != 3 or not all(vectors):
            return False, "text/image alignment", None, None
        margin = cosine(vectors[0], vectors[1]) - cosine(vectors[0], vectors[2])
        return margin > 0, "matching image cosine margin", margin, "cosine"
    if category == "reranking":
        scored = [item for item in response.get("data", []) if isinstance(item, dict)]
        top = max(scored, key=lambda item: float(item.get("score", float("-inf")))).get("index", -1) if scored else -1
        ok = top == fixture["expected_top_index"]
        return ok, "expected document ranked first", 1.0 if ok else 0.0, "pass"
    if category == "chunking":
        chunks = response.get("data") or response.get("chunks") or []
        ok = bool(chunks)
        return ok, "nonempty ordered chunks", float(len(chunks)), "chunks"
    if category == "extraction":
        raw = json.dumps(response, sort_keys=True).lower()
        found = sum(value.lower() in raw for value in fixture["expected"])
        ok = found == len(fixture["expected"])
        return ok, "expected schema values", found / len(fixture["expected"]), "recall"
    if category == "reading / OCR":
        text = " ".join(str(item.get("text", "")) for item in response.get("data", []) if isinstance(item, dict))
        expected = fixture["expected_text"].lower().split()
        found = sum(word in text.lower() for word in expected)
        return found == len(expected), "OCR expected words", found / len(expected), "word recall"
    if category == "transcription":
        text = " ".join(str(item.get("text", "")) for item in response.get("data", []) if isinstance(item, dict)).lower()
        expected = fixture["expected_text"].lower().split()
        found = sum(word in text for word in expected)
        return found >= max(1, len(expected) - 1), "transcription expected words", found / len(expected), "word recall"
    return False, "unknown gate", None, None


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_antfly_trial(
    antfly: Path,
    models_dir: Path,
    bench: dict[str, Any],
    fixture: dict[str, Any],
    iterations: int,
    raw_dir: Path,
    trial: int,
) -> dict[str, Any]:
    api_port = free_port()
    health_port = free_port()
    base_url = f"http://127.0.0.1:{api_port}"
    env = os.environ.copy()
    env["ANTFLY_LOG_LEVEL"] = "warn"
    log_path = raw_dir / f"{bench['id']}-antfly-trial-{trial}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                str(antfly), "inference", "run",
                "--host", "127.0.0.1",
                "--port", str(api_port),
                "--health-port", str(health_port),
                "--models-dir", str(models_dir),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        sampler = MemorySampler(process.pid, raw_dir / f"{bench['id']}-antfly-trial-{trial}-memory.jsonl")
        sampler.start()
        latencies: list[float] = []
        responses: list[dict[str, Any]] = []
        error = ""
        try:
            wait_ready(base_url, process)
            endpoint, body = antfly_request(bench, fixture)
            http_json(base_url + endpoint, body)
            for _ in range(iterations):
                started = time.perf_counter()
                responses.append(http_json(base_url + endpoint, body))
                latencies.append((time.perf_counter() - started) * 1000)
        except Exception as exc:  # campaign rows preserve provider failures
            error = str(exc)
        finally:
            terminate_process(process)
            memory = sampler.stop()
    if error or not responses:
        return {"status": "failed", "error": error or "no response", "latencies_ms": latencies, "memory": memory, "raw_log": str(log_path.relative_to(raw_dir.parent))}
    gate, gate_name, quality, quality_unit = quality_result(bench["category"], responses[-1], fixture)
    return {
        "status": "ok" if gate else "failed",
        "error": "" if gate else f"quality gate failed: {gate_name}",
        "latencies_ms": latencies,
        "quality_gate": gate_name,
        "quality_value": quality,
        "quality_unit": quality_unit,
        "throughput_per_second": (1000 * len(latencies) / sum(latencies)) if sum(latencies) else None,
        "memory": memory,
        "raw_log": str(log_path.relative_to(raw_dir.parent)),
        "response": responses[-1],
    }


def run_reference_trial(
    python: str,
    provider: str,
    models_dir: Path,
    bench: dict[str, Any],
    fixture: dict[str, Any],
    iterations: int,
    raw_dir: Path,
    trial: int,
) -> dict[str, Any]:
    request = {
        "provider": provider,
        "benchmark": bench,
        "fixture": fixture,
        "iterations": iterations,
        "model_path": str(artifact_path(models_dir, bench)),
        "root": str(ROOT),
    }
    log_path = raw_dir / f"{bench['id']}-{provider}-trial-{trial}.log"
    process = subprocess.Popen(
        [python, str(ROOT / "providers" / "reference.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    sampler = MemorySampler(process.pid, raw_dir / f"{bench['id']}-{provider}-trial-{trial}-memory.jsonl")
    sampler.start()
    try:
        stdout, stderr = process.communicate(json.dumps(request), timeout=1800)
    except subprocess.TimeoutExpired:
        terminate_process(process)
        stdout, stderr = "", "provider timed out after 1800 seconds"
    memory = sampler.stop()
    log_path.write_text(stderr, encoding="utf-8")
    if process.returncode != 0:
        return {"status": "failed", "error": stderr.strip()[-2000:] or f"provider exited {process.returncode}", "memory": memory, "raw_log": str(log_path.relative_to(raw_dir.parent))}
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return {"status": "failed", "error": f"provider returned invalid JSON: {stdout[-1000:]}", "memory": memory, "raw_log": str(log_path.relative_to(raw_dir.parent))}
    result["memory"] = memory
    result["raw_log"] = str(log_path.relative_to(raw_dir.parent))
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def aggregate_row(
    bench: dict[str, Any],
    provider: str,
    trials: list[dict[str, Any]],
    memory_publishable: bool,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    successes = [trial for trial in trials if trial.get("status") == "ok"]
    latencies = [float(value) for trial in successes for value in trial.get("latencies_ms", [])]
    memories = [float(trial.get("memory", {}).get("peak_memory_mb", 0)) for trial in successes]
    throughputs = [float(trial["throughput_per_second"]) for trial in successes if trial.get("throughput_per_second") is not None]
    quality_values = [float(trial["quality_value"]) for trial in successes if trial.get("quality_value") is not None]
    status = "ok" if len(successes) == len(trials) and trials else "failed"
    row: dict[str, Any] = {
        "benchmark_id": bench["id"],
        "category": bench["category"],
        "model": bench["model"],
        "variant": bench.get("antfly_variant" if provider == "antfly" else "reference_variant", bench.get("variant", "")),
        "artifact": bench.get("artifact", "") or "provider-native",
        "model_artifact": artifact,
        "provider": provider,
        "status": status,
        "trials": trials,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "throughput_per_second": statistics.fmean(throughputs) if throughputs else None,
        "quality_gate": successes[0].get("quality_gate") if successes else None,
        "quality_value": statistics.fmean(quality_values) if quality_values else None,
        "quality_unit": successes[0].get("quality_unit") if successes else None,
        "memory_publishable": memory_publishable and bool(memories),
        "peak_memory_mean_mb": statistics.fmean(memories) if memories else None,
        "peak_memory_range_mb": [min(memories), max(memories)] if memories else None,
    }
    if status != "ok":
        row["error"] = next((trial.get("error") for trial in trials if trial.get("status") != "ok"), "incomplete trials")
    return row


def format_number(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def render_report(result: dict[str, Any], provenance: dict[str, Any], receipt: dict[str, Any]) -> str:
    lines = [
        f"# Antfly inference campaign — {result['run_id']}",
        "",
        f"- Status: **{receipt['status']}**",
        f"- Profile: `{result['profile']}`",
        f"- Antfly: `{provenance['subject']['version']}` (`{provenance['subject']['sha256'][:12]}`)",
        f"- Host: `{provenance['host']['platform']}` / `{provenance['host']['machine']}`",
        "",
        "Provider-native artifacts are compared as the same logical model only; the artifact and quantization columns are part of the result.",
        "",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result["rows"]:
        if isinstance(row, dict):
            grouped[str(row.get("category") or "uncategorized")].append(row)
    for category in CATEGORY_ORDER:
        rows = grouped.get(category)
        if not rows:
            continue
        lines.extend([
            f"## {category.title()}",
            "",
            "| Model | Variant | Provider | Status | p50 ms | p95 ms | Throughput/s | Quality | Peak mem MB |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in rows:
            quality = format_number(row.get("quality_value"), 3)
            if row.get("quality_unit"):
                quality += f" {row['quality_unit']}"
            memory = format_number(row.get("peak_memory_mean_mb"))
            if memory != "—" and not row.get("memory_publishable"):
                memory += "†"
            lines.append("| " + " | ".join([
                row["model"], row["variant"], row["provider"], row["status"],
                format_number(row.get("latency_p50_ms")), format_number(row.get("latency_p95_ms")),
                format_number(row.get("throughput_per_second")), quality, memory,
            ]) + " |")
        lines.append("")
    lines.extend([
        "† Diagnostic only: the operator did not acknowledge a quiet host or memory sampling was unavailable.",
        "",
        "Rewriters are intentionally not timed for rc.24 because the documented encoder/decoder path is not release-safe.",
        "",
    ])
    return "\n".join(lines)


def write_campaign_bundle(
    campaign_dir: Path,
    result: dict[str, Any],
    provenance: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    write_json(campaign_dir / "result.json", result)
    write_json(campaign_dir / "provenance.json", provenance)
    write_json(campaign_dir / "receipt.json", receipt)
    (campaign_dir / "report.md").write_text(render_report(result, provenance, receipt), encoding="utf-8")


def run_campaign(args: argparse.Namespace, matrix: dict[str, Any], fixtures: dict[str, Any]) -> int:
    antfly = antfly_binary(args.antfly)
    if antfly is None:
        raise CampaignError("ANTFLY_BIN or --antfly must name an executable")
    models_dir = Path(args.models_dir).expanduser().resolve()
    python = args.python
    selected = select_benchmarks(matrix, args.profile, args.category)
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{args.profile}-{sha256_file(antfly)[:12]}"
    campaign_dir = Path(args.output or ROOT / "results" / "campaigns" / run_id).resolve()
    raw_dir = campaign_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    version = command_output([str(antfly), "--version"])
    subject_id = args.subject_id or os.environ.get("ANTFLY_SUBJECT_ID") or version
    model_artifacts = {bench["id"]: fingerprint_path(artifact_path(models_dir, bench)) for bench in selected if not bench["model"].startswith("fixed_")}
    for bench in selected:
        if bench.get("draft_model"):
            model_artifacts[f"{bench['id']}:draft"] = fingerprint_path(
                named_model_dir(models_dir, "generators", str(bench["draft_model"]))
            )
    reference_artifacts: dict[str, Any] = {}
    for bench in selected:
        for provider_id in providers_for(bench, args.profile):
            cache_model = reference_cache_model(bench, provider_id)
            if cache_model and cache_model not in reference_artifacts:
                cache_path = cached_huggingface_model(python, cache_model)
                reference_artifacts[cache_model] = fingerprint_path(Path(cache_path)) if cache_path else {"status": "missing"}
    provider_ids = {provider for bench in selected for provider in providers_for(bench, args.profile)}
    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started,
        "subject": {"path": str(antfly), "subject_id": subject_id, "version": version, "sha256": sha256_file(antfly)},
        "host": {"platform": platform.platform(), "machine": platform.machine(), "hostname": platform.node(), "python": sys.version.split()[0]},
        "matrix_sha256": sha256_file(Path(args.matrix)),
        "antfly_contract": matrix["antfly_contract"],
        "fixtures_sha256": sha256_file(Path(args.fixtures)),
        "models_dir": str(models_dir),
        "models": model_artifacts,
        "reference_models": reference_artifacts,
        "providers": provider_provenance(matrix, provider_ids, python),
        "profile": args.profile,
        "category": args.category,
        "quiet_host_acknowledged": os.environ.get("QUIET_HOST_ACK") == "1",
    }
    write_json(campaign_dir / "provenance.json", provenance)
    config = matrix["profiles"][args.profile]
    iterations = int(config["iterations"])
    trial_count = int(config["trials"])
    memory_publishable = args.profile == "full" and os.environ.get("QUIET_HOST_ACK") == "1"
    rows: list[dict[str, Any]] = []
    expected_rows = sum(len(providers_for(bench, args.profile)) for bench in selected)
    result = {"schema_version": 1, "run_id": run_id, "profile": args.profile, "category": args.category, "rows": rows}
    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at": started,
        "completed_at": None,
        "expected_rows": expected_rows,
        "completed_rows": 0,
        "failed_rows": [],
    }
    write_campaign_bundle(campaign_dir, result, provenance, receipt)
    for bench_index, bench in enumerate(selected):
        provider_ids = providers_for(bench, args.profile)
        trial_results: dict[str, list[dict[str, Any]]] = {provider: [] for provider in provider_ids}
        for trial in range(1, trial_count + 1):
            order = provider_ids if (bench_index + trial) % 2 else list(reversed(provider_ids))
            for provider in order:
                fixture = fixtures[bench["fixture"]]
                if provider == "antfly":
                    trial_result = run_antfly_trial(antfly, models_dir, bench, fixture, iterations, raw_dir, trial)
                else:
                    trial_result = run_reference_trial(python, provider, models_dir, bench, fixture, iterations, raw_dir, trial)
                trial_results[provider].append(trial_result)
        for provider in provider_ids:
            if provider in {"antfly", "llama.cpp"}:
                artifact_label = str(artifact_path(models_dir, bench))
                artifact_fingerprint = model_artifacts.get(bench["id"])
            else:
                artifact_label = configured_provider_model(bench, provider) or "provider-native"
                artifact_fingerprint = reference_artifacts.get(str(artifact_label))
                if provider == "whisper.cpp" and artifact_label != "provider-native":
                    artifact_fingerprint = fingerprint_path(Path(artifact_label).expanduser())
            row = aggregate_row(bench, provider, trial_results[provider], memory_publishable, artifact_fingerprint)
            row["artifact"] = artifact_label
            rows.append(row)
        receipt["completed_rows"] = sum(row["status"] == "ok" for row in rows)
        receipt["failed_rows"] = [row["benchmark_id"] + "/" + row["provider"] for row in rows if row["status"] != "ok"]
        write_campaign_bundle(campaign_dir, result, provenance, receipt)
    status = "ok" if rows and all(row["status"] == "ok" for row in rows) else "failed"
    receipt["status"] = status
    receipt["completed_at"] = utc_now()
    write_campaign_bundle(campaign_dir, result, provenance, receipt)
    print(campaign_dir)
    return 0 if status == "ok" else 1


def pull_models(args: argparse.Namespace, matrix: dict[str, Any]) -> int:
    antfly = antfly_binary(args.antfly)
    if antfly is None:
        raise CampaignError("ANTFLY_BIN or --antfly must name an executable")
    models_dir = Path(args.models_dir).expanduser().resolve()
    for bench in select_benchmarks(matrix, args.profile, args.category):
        pull = bench.get("pull") or []
        if not pull:
            continue
        command = [str(antfly), "inference", "pull", "--models-dir", str(models_dir), *pull]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    return 0


def validate_campaign(path: Path) -> int:
    required = ["report.md", "result.json", "provenance.json", "receipt.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        print(json.dumps({"status": "failed", "missing": missing}, indent=2))
        return 1
    result = read_json(path / "result.json")
    receipt = read_json(path / "receipt.json")
    ok = result.get("schema_version") == 1 and receipt.get("schema_version") == 1 and isinstance(result.get("rows"), list)
    print(json.dumps({"status": "ok" if ok else "failed", "rows": len(result.get("rows", [])), "receipt_status": receipt.get("status")}, indent=2))
    return 0 if ok else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    result.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("prereqs", "models", "run"):
        command = sub.add_parser(name)
        command.add_argument("--profile", choices=("smoke", "full"), default="smoke")
        command.add_argument("--category")
        command.add_argument("--antfly")
        command.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", str(DEFAULT_MODELS)))
        command.add_argument("--python", default=os.environ.get("REFERENCE_PYTHON", str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable))
    run = sub.choices["run"]
    run.add_argument("--run-id")
    run.add_argument("--output")
    run.add_argument("--subject-id")
    validate = sub.add_parser("validate")
    validate.add_argument("result", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    matrix = read_json(Path(args.matrix))
    validate_matrix(matrix)
    fixtures = read_json(Path(args.fixtures))
    if args.command == "prereqs":
        report = prerequisite_report(matrix, fixtures, args.profile, args.category, antfly_binary(args.antfly), Path(args.models_dir).expanduser(), args.python)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "ok" else 1
    if args.command == "models":
        return pull_models(args, matrix)
    if args.command == "run":
        return run_campaign(args, matrix, fixtures)
    if args.command == "validate":
        return validate_campaign(args.result)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
