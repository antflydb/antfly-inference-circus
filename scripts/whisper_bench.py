#!/usr/bin/env python3
"""Paired Whisper-small benchmark: Antfly, Transformers, and whisper.cpp.

All arms use openai/whisper-small, forced English transcription, greedy decoding,
and text-only output.  The cells cover one short LibriSpeech clip, a cohort of
four clips, and one deterministic recording longer than two 30-second windows.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import http.client
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODEL = "openai/whisper-small"
MODELS = [MODEL]
WORKLOAD = "whisper"
PROVIDERS = ["antfly", "transformers", "whisper.cpp"]
CELLS = ["short", "batch4", "multiwindow60"]
REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
SOURCE_WEIGHT_SHA256 = "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b"
MEDIA_MANIFEST = C.ROOT / "media_manifest.json"


def public_error(exc: BaseException) -> str:
    return C.public_error(exc)

def sanitize_log(path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    private_paths = {C.ROOT, C.RESULTS, C.DATA, C.MODELS_DIR, C.HF_HOME,
                     C.CORPUS.parent, C.MEDIA, Path.home()}
    for private in sorted((str(path) for path in private_paths), key=len, reverse=True):
        text = text.replace(private, C.public_path(private))
    path.write_text(text, encoding="utf-8")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text))


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, rvalue in enumerate(right, 1):
        current = [row]
        for column, lvalue in enumerate(left, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (lvalue != rvalue)))
        previous = current
    return previous[-1]


def quality(texts: list[str], references: list[str]) -> dict[str, Any]:
    if len(texts) != len(references):
        raise ValueError("transcript/reference count differs")
    rows = []
    total_edits = total_words = 0
    for text, reference in zip(texts, references, strict=True):
        observed = normalize_text(text).split()
        expected = normalize_text(reference).split()
        edits = edit_distance(observed, expected)
        total_edits += edits
        total_words += len(expected)
        rows.append({"word_edits": edits, "reference_words": len(expected),
                     "wer": edits / max(1, len(expected)),
                     "exact_normalized": observed == expected})
    wer = total_edits / max(1, total_words)
    return {"wer": wer, "word_accuracy": max(0.0, 1.0 - wer),
            "word_edits": total_edits, "reference_words": total_words,
            "exact_normalized_count": sum(row["exact_normalized"] for row in rows),
            "items": rows}


def _verified_manifest() -> dict[str, Any]:
    manifest = json.loads(MEDIA_MANIFEST.read_text(encoding="utf-8"))
    audio = manifest.get("audio")
    source = manifest.get("librispeech") or {}
    if not isinstance(audio, list) or len(audio) < 8:
        raise ValueError("media_manifest.json must contain at least eight audio fixtures")
    if source.get("revision") != "5be91486e11a2d616f4ec5db8d3fd248585ac07a":
        raise ValueError("unexpected LibriSpeech fixture revision")
    for entry in audio[:8]:
        path = C.MEDIA / entry["name"]
        if C.sha256_file(path) != entry["sha256"]:
            raise ValueError(f"audio fixture checksum differs: {entry['name']}")
    return manifest


def concatenate_wav(entries: list[dict[str, Any]], destination: Path) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames: list[bytes] = []
    params = None
    total_frames = 0
    for entry in entries:
        with wave.open(str(C.MEDIA / entry["name"]), "rb") as source:
            current = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
            if current != (1, 2, 16000, "NONE"):
                raise ValueError(f"unexpected WAV format: {entry['name']} {current}")
            if params is None:
                params = current
            data = source.readframes(source.getnframes())
            frames.append(data)
            total_frames += len(data) // 2
    with wave.open(str(destination), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"".join(frames))
    return total_frames / 16000


def make_cells(only: str | None, out: Path) -> list[dict[str, Any]]:
    audio = _verified_manifest()["audio"][:8]
    long_path = out / "inputs" / "libri-multiwindow60.wav"
    long_seconds = concatenate_wav(audio, long_path)
    if long_seconds < 60:
        raise ValueError(f"multiwindow fixture is only {long_seconds:.2f}s")

    def call(entry: dict[str, Any]) -> dict[str, Any]:
        return {"id": entry["id"], "path": C.MEDIA / entry["name"],
                "ref": entry["text"], "seconds": float(entry["seconds"]),
                "sha256": entry["sha256"]}

    cells = [
        {"id": "short", "calls": [call(audio[0])]},
        {"id": "batch4", "calls": [call(entry) for entry in audio[:4]]},
        {"id": "multiwindow60", "calls": [{"id": "multiwindow60", "path": long_path,
             "ref": " ".join(entry["text"] for entry in audio), "seconds": long_seconds,
             "sha256": C.sha256_file(long_path)}]},
    ]
    selected = [cell for cell in cells if only in (None, cell["id"])]
    if not selected:
        raise ValueError(f"unknown cell: {only}")
    return selected


def _antfly_request(call: dict[str, Any]) -> bytes:
    body = {"model": MODEL,
            "audio": base64.b64encode(Path(call["path"]).read_bytes()).decode("ascii"),
            "language": "en", "timestamps": False, "audio_context": "full"}
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


def _antfly_one(port: int, data: bytes) -> tuple[str, float]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    started = time.perf_counter()
    try:
        connection.request("POST", "/ai/v1/dictate", body=data, headers={"content-type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        connection.close()
    payload = json.loads(raw) if raw else {}
    if response.status != 200 or not isinstance(payload.get("text"), str):
        raise RuntimeError(f"Antfly dictate failed status={response.status} error={payload.get('error')!r}")
    return payload["text"], elapsed_ms


def _start_workers(pool: concurrent.futures.ThreadPoolExecutor, count: int) -> None:
    gate = threading.Barrier(count + 1)
    workers = [pool.submit(gate.wait, 30) for _ in range(count)]
    gate.wait(timeout=30)
    for worker in workers:
        worker.result()


def _cell_row(arm: str, cell: dict[str, Any], texts: list[str], elapsed_ms: float,
              lc: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    references = [call["ref"] for call in cell["calls"]]
    audio_seconds = sum(float(call["seconds"]) for call in cell["calls"])
    row: dict[str, Any] = {"workload": WORKLOAD, "model": MODEL, "provider": arm, "arm": arm,
        "lc": lc, "cell": cell["id"], "n_inputs": len(texts), "audio_seconds": audio_seconds,
        "words": sum(len(normalize_text(reference).split()) for reference in references),
        "ms": round(elapsed_ms, 3), "cell_wall_s": elapsed_ms / 1000,
        "audio_x_realtime": audio_seconds / max(elapsed_ms / 1000, 1e-9),
        "quality": quality(texts, references), "ok": True}
    if extra:
        row.update(extra)
    return row


def run_antfly(args: argparse.Namespace, cells: list[dict[str, Any]], out: Path, lc: int) -> dict[str, Any]:
    tag = f"lc{lc}-antfly"
    requests = {cell["id"]: [_antfly_request(call) for call in cell["calls"]] for cell in cells}
    server = C.AntflyServer(args.antfly, out, tag, models_dir=C.MODELS_DIR,
                           extra_args=["--max-concurrent-requests", "4", "--max-work-units", "1024"])
    info: dict[str, Any] = {"provider": "antfly", "arm": "antfly", "model": MODEL,
                            "binary": C.file_identity(args.antfly), "host_before": C.host_snapshot(),
                            "decoding": {"language": "en", "strategy": "greedy",
                                         "timestamps": False, "audio_context": "full"}}
    info.update(server.start())
    pool: concurrent.futures.ThreadPoolExecutor | None = None
    try:
        if any(cell["id"] == "batch4" for cell in cells):
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            _start_workers(pool, 4)
        warm_text, warm_ms = _antfly_one(server.port, requests[cells[0]["id"]][0])
        info["warmup"] = {"performed": True, "ms": round(warm_ms, 3),
                          "nonempty": bool(normalize_text(warm_text))}
        rows = []
        for cell in cells:
            if cell["id"] == "batch4":
                if pool is None:
                    raise RuntimeError("Antfly batch worker pool was not initialized")
                started = time.perf_counter()
                values = list(pool.map(lambda data: _antfly_one(server.port, data), requests[cell["id"]]))
                wall_ms = (time.perf_counter() - started) * 1000
                texts = [value[0] for value in values]
                request_ms = [round(value[1], 3) for value in values]
            else:
                value = _antfly_one(server.port, requests[cell["id"]][0])
                texts, request_ms = [value[0]], [round(value[1], 3)]
                wall_ms = value[1]
            semantics = ("four concurrent HTTP requests eligible for Antfly tensor microbatching"
                         if cell["id"] == "batch4" else "one HTTP request")
            rows.append({"call": _cell_row("antfly", cell, texts, wall_ms, lc,
                         {"request_ms": request_ms, "batch_semantics": semantics,
                          "warmup_state": "same server process after one inference warmup",
                          "timing_basis": "pre-encoded HTTP request dispatch through complete response-body read; WAV/base64/JSON preparation and worker startup excluded"}),
                         "texts": texts, "refs": [call["ref"] for call in cell["calls"]]})
        info["rows"] = rows
        return info
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        info["host_after"] = C.host_snapshot()
        info["footprint"] = server.stop()
        sanitize_log(out / f"{tag}.server.log")


def _model_dir() -> Path:
    owner, name = MODEL.split("/", 1)
    roots = [C.MODELS_DIR / owner / name,
             C.MODELS_DIR / "transcribers" / owner / name]
    roots.extend(sorted((C.MODELS_DIR / owner).glob(f"{name}--antfly-*")))
    found = [path for path in roots if (path / "model.safetensors").is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one installed {MODEL} model, found {[C.public_path(p) for p in found]}")
    return found[0]


def verify_same_weights() -> dict[str, Any]:
    antfly = _model_dir()
    receipt = json.loads((antfly / ".antfly-download-complete.json").read_text(encoding="utf-8"))
    source = receipt.get("source") or {}
    if source.get("resolved_revision") != REVISION:
        raise ValueError("Antfly Whisper source revision differs")
    reference = Path(C.reference_paths()[MODEL])
    antfly_hash = C.sha256_file(antfly / "model.safetensors")
    reference_hash = C.sha256_file(reference / "model.safetensors")
    if antfly_hash != SOURCE_WEIGHT_SHA256 or reference_hash != SOURCE_WEIGHT_SHA256:
        raise ValueError("Whisper Transformers and Antfly source weights are not identical")
    return {"model": MODEL, "revision": REVISION, "model_safetensors_sha256": SOURCE_WEIGHT_SHA256,
            "antfly_source_match": True, "transformers_source_match": True,
            "comparison": "SHA-256 of model.safetensors in both installed snapshots",
            "cuda_precision_note": "Antfly CUDA stores dense projections and the embedding table as FP16 with FP32 accumulation and logits; upstream Transformers keeps the matched source weights FP32."}


def validate_cpp_provenance(model_path: Path, provenance_path: Path) -> dict[str, Any]:
    value = json.loads(provenance_path.read_text(encoding="utf-8"))
    required = {"schema_version": 1, "model": MODEL, "source_revision": REVISION,
                "source_model_sha256": SOURCE_WEIGHT_SHA256}
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"whisper.cpp provenance {key} differs")
    converted = C.sha256_file(model_path)
    if value.get("converted_model_sha256") != converted:
        raise ValueError("whisper.cpp converted model checksum differs from provenance")
    converter = value.get("converter") or {}
    if converter.get("name") != "whisper.cpp" or not isinstance(converter.get("revision"), str) or not converter["revision"]:
        raise ValueError("whisper.cpp provenance needs converter name and revision")
    return {**required, "converted_model_sha256": converted,
            "converter": {"name": "whisper.cpp", "revision": converter["revision"]}}


def _cpp_paths() -> tuple[Path, Path, Path]:
    try:
        binary = Path(os.environ["TERMITE_WHISPER_CPP"]).expanduser()
        model = Path(os.environ["TERMITE_WHISPER_CPP_MODEL"]).expanduser()
        provenance = Path(os.environ["TERMITE_WHISPER_CPP_PROVENANCE"]).expanduser()
    except KeyError as exc:
        raise RuntimeError(f"missing required environment variable {exc.args[0]}") from exc
    for path in (binary, model, provenance):
        if not path.is_file():
            raise FileNotFoundError(C.public_path(path))
    return binary, model, provenance


def _parse_cpp_timings(log: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    for name, value in re.findall(r"whisper_print_timings:\s+([a-z ]+?)\s*=\s*([0-9.]+) ms", log):
        timings[name.strip().replace(" ", "_")] = float(value)
    return timings

def _cpp_compute_ms(timings: dict[str, float]) -> float:
    missing = {"load_time", "total_time"} - timings.keys()
    if missing:
        raise ValueError(f"whisper.cpp timing output is missing {', '.join(sorted(missing))}")
    load_ms = timings["load_time"]
    total_ms = timings["total_time"]
    if not math.isfinite(load_ms) or not math.isfinite(total_ms) or load_ms < 0 or total_ms <= load_ms:
        raise ValueError("whisper.cpp load_time/total_time timing output is malformed")
    return total_ms - load_ms


def run_cpp_arm(cells: list[dict[str, Any]], out: Path, lc: int) -> dict[str, Any]:
    binary, model, _ = _cpp_paths()
    rows = []
    arm_footprints = []
    commands = []
    for cell in cells:
        work = out / "cpp" / f"lc{lc}-{cell['id']}"
        work.mkdir(parents=True, exist_ok=True)
        output_bases = [work / f"result-{index}" for index in range(len(cell["calls"]))]
        command: list[str | Path] = [binary, "--model", model, "--language", "en",
            "--no-timestamps", "--best-of", "1", "--beam-size", "1", "--temperature", "0",
            "--temperature-inc", "0", "--no-fallback", "--output-json"]
        for call, output_base in zip(cell["calls"], output_bases, strict=True):
            command.extend(["--file", call["path"], "--output-file", output_base])
        log_path = out / f"lc{lc}-whisper-cpp-{cell['id']}.log"
        started = time.perf_counter()
        process: subprocess.Popen | None = None
        sampler: C.Sampler | None = None
        footprint: dict[str, Any] = {}
        with log_path.open("w", encoding="utf-8") as log:
            try:
                process = subprocess.Popen([str(part) for part in command], stdout=subprocess.DEVNULL,
                                           stderr=log, start_new_session=True)
                sampler = C.Sampler(process.pid, out, f"lc{lc}-whisper-cpp-{cell['id']}")
                returncode = process.wait(timeout=900)
            finally:
                active_exception = sys.exc_info()[0] is not None
                cleanup_error: Exception | None = None
                if process is not None:
                    try:
                        C.stop_proc(process)
                    except Exception as exc:
                        cleanup_error = exc
                if sampler is not None:
                    try:
                        footprint = sampler.stop()
                    except Exception as exc:
                        cleanup_error = cleanup_error or exc
                if cleanup_error is not None and not active_exception:
                    raise cleanup_error
        wall_ms = (time.perf_counter() - started) * 1000
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        log_path.write_text(log_text.replace(str(model), model.name), encoding="utf-8")
        sanitize_log(log_path)
        log_text = log_path.read_text(encoding="utf-8")
        if returncode != 0:
            raise RuntimeError(f"whisper.cpp exited {returncode}; see {log_path.name}")
        texts = []
        for output_base in output_bases:
            output_path = output_base.with_suffix(".json")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            texts.append("".join(segment["text"] for segment in payload["transcription"]).strip())
            output_path.unlink()
        timings = _parse_cpp_timings(log_text)
        compute_ms = _cpp_compute_ms(timings)
        semantics = ("one whisper-cli process loading the model once and processing four files sequentially; "
                     "whisper.cpp exposes no cross-file batch API"
                     if cell["id"] == "batch4" else "one whisper-cli process")
        rows.append({"call": _cell_row("whisper.cpp", cell, texts, compute_ms, lc,
                    {"wall_ms": round(wall_ms, 3), "model_load_ms": timings["load_time"],
                     "timing_basis": "required native total_time minus required native load_time",
                     "warmup_state": "fresh process, first inference, no inference warmup",
                     "audio_x_realtime_basis": "cold-process first-inference compute time with model load excluded",
                     "batch_semantics": semantics, "native_timings_ms": timings}), "texts": texts,
                     "refs": [call["ref"] for call in cell["calls"]]})
        arm_footprints.append(footprint)
        commands.append(C.public_command(command))
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30)
    footprint = max(arm_footprints,
                    key=lambda value: value.get("process_tree_footprint_peak_bytes", 0),
                    default={})
    return {"provider": "whisper.cpp", "arm": "whisper.cpp", "model": MODEL,
            "binary": C.file_identity(binary), "version": " ".join((version.stdout + version.stderr).split())[:300],
            "commands": commands, "rows": rows, "footprint": footprint,
            "warmup": {"performed": False, "state": "fresh process per cell; first inference is measured"},
            "cell_footprints": arm_footprints}


def _read_wav(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError("expected mono 16-bit 16 kHz WAV")
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype("float32") / 32768.0


def transformers_arm(only: str | None, out: Path, lc: int) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    cells = make_cells(only, out)
    reference = Path(C.reference_paths()[MODEL])
    device = C.ACCEL
    processor = AutoProcessor.from_pretrained(reference, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(reference, local_files_only=True,
                                                       dtype=torch.float32).to(device).eval()
    recognizer = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                          feature_extractor=processor.feature_extractor, device=device,
                          dtype=torch.float32)
    generate = {"language": "en", "task": "transcribe", "do_sample": False, "num_beams": 1}

    prepared = {cell["id"]: [{"raw": _read_wav(Path(call["path"])), "sampling_rate": 16000}
                              for call in cell["calls"]] for cell in cells}

    def infer(inputs: list[dict[str, Any]]):
        # The pipeline pops audio and sampling_rate from each input dictionary.
        # Preserve the prepared arrays for both warmup and measured requests.
        request = [dict(item) for item in inputs]
        argument: Any = request if len(request) > 1 else request[0]
        result = recognizer(argument, batch_size=len(inputs), chunk_length_s=30,
                            return_timestamps=False, generate_kwargs=generate)
        values = result if isinstance(result, list) else [result]
        return [value["text"] for value in values]

    infer(prepared[cells[0]["id"]][:1])
    rows = []
    sync = C.torch_sync(torch)
    for cell in cells:
        sync()
        started = time.perf_counter()
        texts = infer(prepared[cell["id"]])
        sync()
        elapsed_ms = (time.perf_counter() - started) * 1000
        semantics = ("one Transformers pipeline call with batch_size=4"
                     if cell["id"] == "batch4" else "one Transformers pipeline call")
        rows.append({"call": _cell_row("transformers", cell, texts, elapsed_ms, lc,
                    {"batch_semantics": semantics,
                     "warmup_state": "same Python/model process after one inference warmup",
                     "timing_basis": "ASR pipeline feature extraction, model inference, and decode; WAV file read excluded"}),
                    "texts": texts, "refs": [call["ref"] for call in cell["calls"]]})
    versions = {package: importlib.metadata.version(package) for package in ("torch", "transformers", "numpy")}
    return {"provider": "transformers", "arm": "transformers", "model": MODEL,
            "versions": versions, "dtype": "float32", "device": device,
            "warmup": {"performed": True, "state": "one inference before measured cells"},
            "decoding": {"language": "en", "task": "transcribe", "do_sample": False,
                         "num_beams": 1, "return_timestamps": False, "chunk_length_s": 30},
            "device_memory": C.torch_device_bytes(torch), "rows": rows}


def pyarm_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["auto", "cuda", "metal"])
    parser.add_argument("--only-cell", choices=CELLS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lc", type=int, required=True)
    args = parser.parse_args(argv)
    C.configure_backend(args.backend)
    try:
        result = transformers_arm(args.only_cell, args.out, args.lc)
    except Exception as exc:  # child must return a structured terminal record
        result = {"provider": "transformers", "arm": "transformers", "model": MODEL,
                  "error": public_error(exc)}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if "error" not in result else 1


def agreement_rows(lc: int, cells: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for cell in cells:
        by_arm: dict[str, list[str]] = {}
        for arm in PROVIDERS:
            result = results.get(arm) or {}
            match = next((row for row in result.get("rows", []) if row["call"]["cell"] == cell["id"]), None)
            if match is not None:
                by_arm[arm] = match["texts"]
        row: dict[str, Any] = {"workload": WORKLOAD, "model": MODEL, "lc": lc,
                               "cell": cell["id"], "providers": list(by_arm), "complete": len(by_arm) == len(PROVIDERS)}
        if row["complete"]:
            normalized = {arm: [normalize_text(text) for text in texts] for arm, texts in by_arm.items()}
            lengths = {len(texts) for texts in normalized.values()}
            if len(lengths) != 1:
                raise ValueError("inter-engine transcript counts differ")
            count = lengths.pop()
            exact = [len({normalized[arm][index] for arm in PROVIDERS}) == 1 for index in range(count)]
            raw_exact = [len({by_arm[arm][index].strip() for arm in PROVIDERS}) == 1 for index in range(count)]
            pairwise = {}
            for left_index, left in enumerate(PROVIDERS):
                for right in PROVIDERS[left_index + 1:]:
                    edits = words = 0
                    for ltext, rtext in zip(normalized[left], normalized[right], strict=True):
                        lwords, rwords = ltext.split(), rtext.split()
                        edits += edit_distance(lwords, rwords)
                        words += len(rwords)
                    pairwise[f"{left}_vs_{right}_wer"] = edits / max(1, words)
            row.update({"normalized_exact_count": sum(exact), "raw_exact_count": sum(raw_exact),
                        "items": count, "all_normalized_exact": all(exact), "pairwise": pairwise})
        output.append(row)
    return output


def _write_arm(out: Path, result: dict[str, Any], lc: int) -> None:
    calls = C.JsonlWriter(out / "calls.jsonl")
    outputs = C.JsonlWriter(out / "outputs.jsonl")
    try:
        for row in result.get("rows", []):
            calls.write(row["call"])
            for index, text in enumerate(row["texts"]):
                outputs.write({"workload": WORKLOAD, "model": MODEL, "provider": result["provider"],
                               "arm": result["arm"], "lc": lc, "cell": row["call"]["cell"], "k": index,
                               "text": text, "ref": row["refs"][index]})
    finally:
        calls.close()
        outputs.close()


def orchestrate(args: argparse.Namespace) -> int:
    C.configure_backend(args.backend)
    C.exclusive_bench_lock()
    out = Path(args.out)
    C.prepare_output(out)
    cells = make_cells(args.only_cell, out)
    same_weights = verify_same_weights()
    cpp_binary, cpp_model, cpp_provenance_path = _cpp_paths()
    cpp_provenance = validate_cpp_provenance(cpp_model, cpp_provenance_path)
    C.write_json(out / "model_files.json", {"transformers_and_antfly": same_weights,
                 "whisper_cpp": cpp_provenance, "whisper_cpp_binary": C.file_identity(cpp_binary)})
    C.write_json(out / "inputs.json", {"source": {"dataset": "hf-internal-testing/librispeech_asr_dummy",
                 "revision": "5be91486e11a2d616f4ec5db8d3fd248585ac07a"},
                 "cells": [{"id": cell["id"], "items": [{"id": call["id"], "seconds": call["seconds"],
                 "sha256": call["sha256"]} for call in cell["calls"]]} for cell in cells]})
    lifecycle_writer = C.JsonlWriter(out / "lifecycles.jsonl")
    agreement_writer = C.JsonlWriter(out / "agreement.jsonl")
    failed = False
    try:
        for lc in range(args.lifecycles):
            ordered = PROVIDERS[lc % len(PROVIDERS):] + PROVIDERS[:lc % len(PROVIDERS)]
            results: dict[str, dict[str, Any]] = {}
            for arm in ordered:
                started = time.time()
                try:
                    if arm == "antfly":
                        result = run_antfly(args, cells, out, lc)
                    elif arm == "transformers":
                        child_args = ["--backend", args.backend, "--out", str(out), "--lc", str(lc)]
                        if args.only_cell:
                            child_args.extend(["--only-cell", args.only_cell])
                        result = C.run_python_arm(str(Path(__file__)), child_args, out, f"lc{lc}-transformers")
                        sanitize_log(out / f"lc{lc}-transformers.py.log")
                    else:
                        result = run_cpp_arm(cells, out, lc)
                    if result.get("rc", 0) != 0 or "error" in result:
                        failed = True
                    else:
                        _write_arm(out, result, lc)
                except Exception as exc:
                    failed = True
                    result = {"provider": arm, "arm": arm, "model": MODEL, "error": public_error(exc)}
                results[arm] = result
                lifecycle = {key: value for key, value in result.items() if key != "rows"}
                lifecycle.update({"workload": WORKLOAD, "model": MODEL, "provider": arm, "lc": lc,
                                  "status": "failed" if "error" in result or result.get("rc", 0) else "ok",
                                  "wall_s": round(time.time() - started, 3)})
                lifecycle_writer.write(lifecycle)
            for row in agreement_rows(lc, cells, results):
                agreement_writer.write(row)
    finally:
        lifecycle_writer.close()
        agreement_writer.close()
    complete = C.benchmark_complete(out, set(PROVIDERS), args.lifecycles, len(cells))
    return 1 if failed or not complete else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "pyarm":
        return pyarm_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--antfly", type=Path, required=True)
    run.add_argument("--backend", choices=["auto", "cuda", "metal"], default="auto")
    run.add_argument("--lifecycles", type=int, default=3)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--only-cell", choices=CELLS)
    args = parser.parse_args(argv)
    if args.lifecycles < 1:
        parser.error("--lifecycles must be positive")
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
