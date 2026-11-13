#!/usr/bin/env python3
"""Run current paired inference workloads with a supplied Antfly binary."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import fetch_reference  # noqa: E402
import summarize  # noqa: E402

SCHEMA_VERSION = 1
MAX_RECORD_BYTES = 1 << 20
WORKLOAD_GROUPS = {
    "embed": ["embed"],
    "gliner": ["gliner-base", "gliner-decide"],
    "gemma4": ["gemma4"],
    "whisper": ["whisper"],
    "ocr": ["ocr"],
}
PULLS = {
    "embed": ("google/embeddinggemma-2:safetensors", []),
    "gliner-base": ("fastino/gliner2.5-base-v1:safetensors", []),
    "gliner-decide": ("fastino/GLiNER2.5-Decide:safetensors", []),
    "gemma4": ("google/gemma-4-E4B-it-qat-q4_0-gguf:gguf", []),
    "whisper": ("openai/whisper-small:safetensors", []),
    "ocr": ("lightonai/LightOnOCR-3-0.8B:safetensors", []),
}
REFERENCE_MODULES = {
    "embed": {"torch", "sentence_transformers", "huggingface_hub", "psutil"},
    "gliner-base": {"torch", "gliner2", "huggingface_hub", "psutil"},
    "gliner-decide": {"torch", "gliner2", "huggingface_hub", "psutil"},
    "gemma4": {"psutil"},
    "whisper": {"torch", "transformers", "numpy", "huggingface_hub", "psutil"},
    "ocr": {"torch", "transformers", "PIL", "huggingface_hub", "psutil"},
}


class RunnerError(RuntimeError):
    pass


def resolve_binary(value: str | None) -> Path:
    if not value:
        raise RunnerError("--binary or TERMITE_BINARY is required")
    candidate = Path(value).expanduser()
    resolved = str(candidate.resolve()) if candidate.parent != Path(".") or candidate.is_absolute() else shutil.which(value)
    if not resolved:
        raise RunnerError(f"binary is not executable: {candidate.name}")
    path = Path(resolved)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RunnerError(f"binary is not executable: {path.name}")
    return path


def command_version(command: Path) -> str:
    try:
        process = subprocess.run([str(command), "--version"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return C.public_error(exc)
    return " ".join((process.stdout + process.stderr).split())[:300]


def selected_keys(workloads: list[str] | None, models: list[str] | None) -> list[str]:
    groups = workloads or list(WORKLOAD_GROUPS)
    keys = [key for group in groups for key in WORKLOAD_GROUPS[group]]
    keys = list(dict.fromkeys(keys))
    if not models:
        return keys
    requested = set(models)
    matched = [
        key
        for key in keys
        if requested
        & {
            key,
            summarize.WORKLOADS[key]["model"],
            summarize.WORKLOADS[key]["model_key"],
        }
    ]
    recognized = {
        value
        for key in matched
        for value in (key, summarize.WORKLOADS[key]["model"], summarize.WORKLOADS[key]["model_key"])
        if value in requested
    }
    unknown = requested - recognized
    if unknown:
        raise RunnerError(f"unknown or unselected model: {', '.join(sorted(unknown))}")
    if not matched:
        raise RunnerError("no workloads selected")
    return matched


def validate_selection(keys: list[str], only_cell: str | None, lifecycles: int) -> None:
    if lifecycles < 1:
        raise RunnerError("--lifecycles must be positive")
    if only_cell is not None:
        invalid = [key for key in keys if only_cell not in summarize.WORKLOADS[key]["cells"]]
        if invalid:
            raise RunnerError(f"unknown cell {only_cell!r} for {', '.join(invalid)}")


def fresh_output(output: Path) -> None:
    protected = ("result.json", "provenance.json", "inputs.provenance.json", "terminal.json", "speed", "memory")
    stale = [name for name in protected if (output / name).exists()]
    if stale:
        raise RunnerError(f"output already contains benchmark records: {', '.join(stale)}")


def sanitized_log(text: str) -> str:
    for path in (C.ROOT, C.DATA, C.MODELS_DIR, C.HF_HOME, C.CORPUS, C.MEDIA, Path.home()):
        text = text.replace(str(path.expanduser()), C.public_path(path))
    return text


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    process = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(sanitized_log(process.stdout + process.stderr), encoding="utf-8")
    return process.returncode


def setup_assets(keys: list[str], binary: Path, env: dict[str, str], output: Path) -> list[dict]:
    steps = []
    for key in keys:
        spec = summarize.WORKLOADS[key]
        pull, extra = PULLS[key]
        models_dir = (
            Path(os.environ["TERMITE_GEN_MODELS_DIR"]).expanduser()
            if key == "gemma4" and os.environ.get("TERMITE_GEN_MODELS_DIR")
            else C.MODELS_DIR
        )
        command = [str(binary), "inference", "pull", pull, "--models-dir", str(models_dir), *extra]
        returncode = run_logged(command, output / "execution" / f"setup-{key}.txt", env)
        steps.append(
            {
                "workload": spec["workload"],
                "model": spec["model"],
                "provider": "assets",
                "phase": "setup",
                "status": "complete" if returncode == 0 else "failed",
                **({} if returncode == 0 else {"error": f"model pull exited {returncode}"}),
            }
        )
        if returncode != 0:
            continue
    if any(step["status"] != "complete" for step in steps):
        return steps
    reference_repos = {summarize.WORKLOADS[key]["model"] for key in keys if key != "gemma4"}
    if reference_repos:
        command = [
            str(C.REF_PY),
            str(C.ROOT / "scripts" / "fetch_reference.py"),
            "--models-dir",
            str(C.MODELS_DIR),
            "--output",
            str(C.REFERENCES),
        ]
        for repo in sorted(reference_repos):
            command.extend(["--model", repo])
        returncode = run_logged(command, output / "execution" / "setup-references.txt", env)
        steps.append(
            {
                "workload": "setup",
                "model": "reference snapshots",
                "provider": "huggingface",
                "phase": "setup",
                "status": "complete" if returncode == 0 else "failed",
                **({} if returncode == 0 else {"error": f"reference fetch exited {returncode}"}),
            }
        )
    needs_corpus = any(key in {"embed", "gliner-base", "gliner-decide", "gemma4"} for key in keys)
    corpus_ready = False
    if needs_corpus and C.CORPUS.is_file():
        try:
            C.load_corpus()
            corpus_ready = True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if needs_corpus and not corpus_ready:
        command = [str(C.REF_PY), str(C.ROOT / "scripts" / "corpus.py"), "--output", str(C.CORPUS)]
        returncode = run_logged(command, output / "execution" / "setup-corpus.txt", env)
        steps.append(
            {
                "workload": "setup",
                "model": "text corpus",
                "provider": "public fixtures",
                "phase": "setup",
                "status": "complete" if returncode == 0 else "failed",
                **({} if returncode == 0 else {"error": f"corpus build exited {returncode}"}),
            }
        )
    needs_media = any(key in {"gemma4", "whisper"} for key in keys)
    if needs_media:
        command = [str(C.REF_PY), str(C.ROOT / "scripts" / "media.py"), "fetch"]
        returncode = run_logged(command, output / "execution" / "setup-media.txt", env)
        steps.append(
            {
                "workload": "setup",
                "model": "media fixtures",
                "provider": "public fixtures",
                "phase": "setup",
                "status": "complete" if returncode == 0 else "failed",
                **({} if returncode == 0 else {"error": f"media fetch exited {returncode}"}),
            }
        )
    return steps


def verify_python_modules(keys: list[str]) -> dict[str, Any]:
    modules = sorted({module for key in keys for module in REFERENCE_MODULES[key]})
    code = f"""
import importlib
import importlib.util
import json

modules = {modules!r}
missing = [name for name in modules if importlib.util.find_spec(name) is None]
versions = {{}}
for name in modules:
    if name in missing:
        continue
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, "__version__", "unknown"))
print(json.dumps({{"missing": missing, "versions": versions}}))
raise SystemExit(bool(missing))
"""
    process = subprocess.run([str(C.REF_PY), "-c", code], capture_output=True, text=True, timeout=60, check=False)
    try:
        report = json.loads(process.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise RunnerError(f"reference Python dependency report failed: {C.public_error(exc)}") from exc
    if process.returncode:
        raise RunnerError(f"reference Python is missing modules: {', '.join(report['missing'])}")
    return {
        "name": C.REF_PY.name,
        "version": command_version(C.REF_PY),
        "packages": report["versions"],
    }


def verify_media() -> dict:
    manifest = json.loads((C.ROOT / "media_manifest.json").read_text(encoding="utf-8"))
    files = []
    for entry in [*manifest["images"], *manifest["audio"]]:
        path = C.MEDIA / entry["name"]
        if not path.is_file() or C.sha256_file(path) != entry["sha256"]:
            raise RunnerError(f"media fixture is missing or mismatched: {entry['name']}")
        files.append({"name": entry["name"], "sha256": entry["sha256"], "size_bytes": entry["bytes"]})
    return {"manifest": C.file_identity(C.ROOT / "media_manifest.json"), "files": files}


def import_script(name: str):
    import importlib.util

    path = C.ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"termite_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RunnerError(f"cannot import {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(keys: list[str], binary: Path, backend: str) -> dict[str, Any]:
    C.configure_backend(backend)
    for key in keys:
        script = C.ROOT / "scripts" / summarize.WORKLOADS[key]["script"]
        if not script.is_file():
            raise RunnerError(f"missing workload script {script.name}")
    python = verify_python_modules(keys)
    assets: dict[str, Any] = {"reference_python": python}
    if any(key in {"embed", "gliner-base", "gliner-decide", "gemma4"} for key in keys):
        if not C.CORPUS.is_file():
            raise RunnerError("text corpus is missing; run --phase setup")
        corpus = C.load_corpus()
        assets["corpus"] = {
            "artifact": C.file_identity(C.CORPUS),
            "seed": corpus.get("seed"),
            "sampling": corpus.get("sampling"),
            "sources": corpus.get("sources"),
        }
    if any(key in {"gemma4", "whisper"} for key in keys):
        assets["media"] = verify_media()
    safetensor_repos = {summarize.WORKLOADS[key]["model"] for key in keys if key != "gemma4"}
    if safetensor_repos:
        assets["reference_models"] = fetch_reference.verify_cached(C.MODELS_DIR, safetensor_repos)
    if "gemma4" in keys:
        gemma = import_script("gemma4_bench.py")
        assets["gemma4_files"] = gemma.public_verified_files(gemma.verify_files())
        llama = os.environ.get("TERMITE_LLAMA_SERVER", "llama-server")
        llama_resolved = shutil.which(llama)
        if not llama_resolved:
            raise RunnerError("llama-server is required for Gemma 4")
        llama_path = Path(llama_resolved)
        assets["llama_server"] = {
            **C.file_identity(llama_path),
            "version": command_version(llama_path),
        }
    if "whisper" in keys:
        whisper = import_script("whisper_bench.py")
        assets["whisper_files"] = whisper.verify_same_weights()
        cpp_binary, cpp_model, cpp_provenance = whisper._cpp_paths()
        if not os.access(cpp_binary, os.X_OK):
            raise RunnerError("TERMITE_WHISPER_CPP must be executable")
        assets["whisper_cpp"] = whisper.validate_cpp_provenance(cpp_model, cpp_provenance)
        assets["whisper_cpp_binary"] = C.file_identity(cpp_binary)
    if "ocr" in keys:
        ocr = import_script("ocr_bench.py")
        assets["ocr_files"] = ocr.verify_same_weights()
    assets["binary"] = {**C.file_identity(binary), "version": command_version(binary)}
    return assets


def workload_command(
    key: str,
    binary: Path,
    backend: str,
    lifecycles: int,
    out: Path,
    only_cell: str | None,
) -> list[str]:
    spec = summarize.WORKLOADS[key]
    command = [
        str(C.REF_PY),
        str(C.ROOT / "scripts" / spec["script"]),
        "run",
        *spec["script_args"],
        "--antfly",
        str(binary),
        "--backend",
        backend,
        "--lifecycles",
        str(lifecycles),
        "--out",
        str(out),
    ]
    if only_cell:
        command.extend(["--only-cell", only_cell])
    return command


def run_measurements(
    root: Path,
    keys: list[str],
    phases: set[str],
    binary: Path,
    backend: str,
    lifecycles: int,
    only_cell: str | None,
    env: dict[str, str],
) -> dict[tuple[str, str], int]:
    failures = {}
    if "speed" in phases:
        for key in keys:
            out = root / "speed" / key
            command = workload_command(key, binary, backend, lifecycles, out, only_cell)
            rc = run_logged(command, root / "execution" / f"speed-{key}.txt", env)
            if rc:
                failures[(summarize.WORKLOADS[key]["model"], "speed")] = rc
    if "memory" in phases:
        for key in keys:
            cells = summarize.selected_cells(summarize.WORKLOADS[key], only_cell)
            for cell in cells:
                out = root / "memory" / key / cell
                command = workload_command(key, binary, backend, 1, out, cell)
                rc = run_logged(command, root / "execution" / f"memory-{key}-{cell}.txt", env)
                if rc:
                    failures[(summarize.WORKLOADS[key]["model"], "memory")] = rc
    return failures


def failure_steps(keys: list[str], phases: set[str], error: str) -> list[dict]:
    steps = []
    measured = phases & {"speed", "memory"}
    if not measured:
        measured = {"check"}
    for key in keys:
        spec = summarize.WORKLOADS[key]
        for phase in sorted(measured):
            for provider in spec["arms"].values():
                steps.append(
                    {
                        "workload": spec["workload"],
                        "model": spec["model"],
                        "provider": provider,
                        "phase": phase,
                        "status": "failed",
                        "error": error,
                    }
                )
    return steps




def write_records(output: Path, records: dict[str, dict]) -> None:
    encoded = {
        name: json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for name, record in records.items()
    }
    oversized = [name for name, text in encoded.items() if len(text.encode()) >= MAX_RECORD_BYTES]
    if oversized:
        raise RunnerError(f"record exceeds 1 MiB: {', '.join(oversized)}")
    output.mkdir(parents=True, exist_ok=True)
    for name, text in encoded.items():
        (output / name).write_text(text, encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", default=os.environ.get("TERMITE_BINARY"))
    result.add_argument("--backend", choices=("auto", "metal", "cuda"), default=os.environ.get("TERMITE_BACKEND", "auto"))
    result.add_argument("--output", type=Path)
    result.add_argument("--models-dir", type=Path, default=Path(os.environ.get("TERMITE_MODELS_DIR", C.MODELS_DIR)))
    result.add_argument("--workload", action="append", choices=sorted(WORKLOAD_GROUPS))
    result.add_argument("--model", action="append")
    result.add_argument("--phase", action="append", choices=("setup", "speed", "memory"))
    result.add_argument("--lifecycles", type=int, default=3)
    result.add_argument("--only-cell")
    result.add_argument("--run-id")
    result.add_argument("--check", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    started_at = C.utc_now()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    phases = {"check"} if args.check else set(args.phase or ["speed", "memory"])
    output = (args.output or Path(os.environ.get("TERMITE_OUTPUT", C.RESULTS / run_id))).expanduser().resolve()
    try:
        fresh_output(output)
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    keys: list[str] = []
    binary: Path | None = None
    error = ""
    steps: list[dict] = []
    comparisons: list[dict] = []
    assets: dict[str, Any] = {}
    exit_code = 1
    try:
        keys = selected_keys(args.workload, args.model)
        validate_selection(keys, args.only_cell, args.lifecycles)
        binary = resolve_binary(args.binary)
        backend = C.configure_backend(args.backend)
        C.MODELS_DIR = args.models_dir.expanduser().resolve()
        env = dict(os.environ)
        env.update(
            {
                "TERMITE_BINARY": str(binary),
                "TERMITE_BACKEND": backend,
                "TERMITE_RESULTS": str(output),
                "TERMITE_MODELS_DIR": str(C.MODELS_DIR),
            }
        )
        if "setup" in phases:
            setup_steps = setup_assets(keys, binary, env, output)
            steps.extend(setup_steps)
            if any(step["status"] != "complete" for step in setup_steps):
                raise RunnerError("one or more setup steps failed")
        assets = preflight(keys, binary, backend)
        if args.check:
            steps = [
                {
                    "workload": summarize.WORKLOADS[key]["workload"],
                    "model": summarize.WORKLOADS[key]["model"],
                    "provider": provider,
                    "phase": "check",
                    "status": "complete",
                }
                for key in keys
                for provider in summarize.WORKLOADS[key]["arms"].values()
            ]
        else:
            measured = phases & {"speed", "memory"}
            execution_failures = run_measurements(output, keys, measured, binary, backend, args.lifecycles, args.only_cell, env)
            if measured:
                summary = summarize.build_summary(output, keys, measured, args.lifecycles, args.only_cell)
                for step in summary["steps"]:
                    rc = execution_failures.get((step["model"], step["phase"]))
                    if rc:
                        step["status"] = "failed"
                        step.pop("metrics", None)
                        step["error"] = f"workload process exited {rc}"
                steps.extend(summary["steps"])
                comparisons.extend(
                    row for row in summary["comparisons"]
                    if (row["model"], row.get("phase", "speed")) not in execution_failures
                )
                if execution_failures or summary["status"] != "complete":
                    raise RunnerError("one or more measured provider steps failed")
        exit_code = 0
    except Exception as exc:
        error = C.public_error(exc)
        if not any(step.get("status") == "failed" for step in steps):
            steps.extend(failure_steps(keys, phases, error) if keys else [])
    completed_at = C.utc_now()
    complete = exit_code == 0 and bool(steps) and all(step.get("status") == "complete" for step in steps)
    if not complete:
        exit_code = 1
    status = "complete" if complete else "failed"
    binary_record = C.file_identity(binary) | {"version": command_version(binary)} if binary and binary.is_file() else None
    selection = {
        "workloads": [summarize.WORKLOADS[key]["workload"] for key in keys],
        "models": [summarize.WORKLOADS[key]["model"] for key in keys],
        "phases": sorted(phases),
        "lifecycles": args.lifecycles,
        "cell": args.only_cell,
        "backend": C.ANTFLY_BACKEND,
    }
    records = {
        "result.json": {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "backend": C.ANTFLY_BACKEND,
            "steps": steps,
            "comparisons": comparisons,
            **({"error": error} if error else {}),
        },
        "provenance.json": {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "subject": binary_record,
            "runner": C.file_identity(Path(__file__)),
            "protocol_sources": [
                C.file_identity(path) for path in sorted((C.ROOT / "scripts").glob("*.py"))
            ],
            "backend": C.ANTFLY_BACKEND,
            "platform": {"system": platform.system(), "architecture": platform.machine()},
            "python": sys.version.split()[0],
        },
        "inputs.provenance.json": {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": status,
            "selection": selection,
            "assets": assets,
        },
        "terminal.json": {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": status,
            "exit_code": exit_code,
            "started_at": started_at,
            "completed_at": completed_at,
            "steps_total": len(steps),
            "steps_complete": sum(step.get("status") == "complete" for step in steps),
            **({"error": error} if error else {}),
        },
    }
    try:
        write_records(output, records)
    except RunnerError as exc:
        print(f"error: {C.public_error(exc)}", file=sys.stderr)
        return 1
    print(output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
