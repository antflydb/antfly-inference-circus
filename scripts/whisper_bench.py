#!/usr/bin/env python3
"""Paired Whisper-small benchmark: Antfly, Transformers, and whisper.cpp.

All arms use openai/whisper-small, forced English transcription, greedy decoding,
and text-only output. Each invocation transcribes one intact LibriSpeech recording:
short (~6s), medium (~15s), or near-window long (~29s), with no multiwindow/fanout.
"""
from __future__ import annotations

import argparse
import base64
import http.client
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
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
CELLS = ["short", "medium", "long"]
REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
SOURCE_WEIGHT_SHA256 = "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b"
CPP_REVISION = "306c88f4d1286aec1bf96e544632897886af5501"  # whisper.cpp v1.9.2
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
    if len(texts) != len(references) or not references:
        raise ValueError("need equal, nonempty transcript/reference sets")
    rows = []
    for text, reference in zip(texts, references, strict=True):
        observed, expected = normalize_text(text), normalize_text(reference)
        words = expected.split()
        if not words:
            raise ValueError("reference has no normalized words")
        # CER excludes spaces; punctuation/case normalization is shared with WER.
        chars = expected.replace(" ", "")
        word_edits = edit_distance(observed.split(), words)
        character_edits = edit_distance(observed.replace(" ", ""), chars)
        rows.append({"word_edits": word_edits, "reference_words": len(words),
                     "character_edits": character_edits, "reference_characters": len(chars),
                     "wer": word_edits / len(words), "cer": character_edits / len(chars),
                     "exact_normalized": observed == expected})
    word_edits = sum(row["word_edits"] for row in rows)
    words = sum(row["reference_words"] for row in rows)
    character_edits = sum(row["character_edits"] for row in rows)
    characters = sum(row["reference_characters"] for row in rows)
    return {"wer": word_edits / words, "cer": character_edits / characters,
            "word_edits": word_edits, "reference_words": words,
            "character_edits": character_edits, "reference_characters": characters,
            "clips": len(rows),
            "exact_normalized_count": sum(row["exact_normalized"] for row in rows),
            "items": rows}


def _verified_manifest() -> dict[str, Any]:
    manifest = json.loads(MEDIA_MANIFEST.read_text(encoding="utf-8"))
    audio = manifest.get("audio")
    source = manifest.get("librispeech") or {}
    if not isinstance(audio, list) or not audio:
        raise ValueError("media_manifest.json must contain an audio fixture")
    if source.get("revision") != "5be91486e11a2d616f4ec5db8d3fd248585ac07a":
        raise ValueError("unexpected LibriSpeech fixture revision")
    entry = audio[0]
    path = C.MEDIA / entry["name"]
    if C.sha256_file(path) != entry["sha256"]:
        raise ValueError(f"audio fixture checksum differs: {entry['name']}")
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(),
                source.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError(f"unexpected WAV format: {entry['name']}")
        # Timing denominators use actual sample count, not the manifest's
        # two-decimal display duration.
        entry["seconds"] = source.getnframes() / source.getframerate()
    return manifest


def make_cells(only: str | None) -> list[dict[str, Any]]:
    if only not in (None, *CELLS):
        raise ValueError(f"unknown cell: {only}")
    result = []
    if only in (None, "short"):
        entry = _verified_manifest()["audio"][0]
        result.append({"id": "short", "calls": [
            {"id": entry["id"], "path": C.MEDIA / entry["name"],
             "ref": entry["text"], "seconds": float(entry["seconds"]),
             "sha256": entry["sha256"]}]})
    if only != "short":
        from quality_whisper import timing_clips
        result.extend(timing_clips(only))
    return result


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
    if response.status != 200:
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            payload = None
        raise C.HttpResponseError(response.status, payload)
    payload = json.loads(raw) if raw else {}
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise RuntimeError("Antfly dictate returned a malformed successful response")
    return payload["text"], elapsed_ms


def _cell_row(arm: str, cell: dict[str, Any], texts: list[str], elapsed_ms: float,
              lc: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ValueError("elapsed request time must be finite and positive")
    references = [call["ref"] for call in cell["calls"]]
    audio_seconds = sum(float(call["seconds"]) for call in cell["calls"])
    if not math.isfinite(audio_seconds) or audio_seconds <= 0:
        raise ValueError("audio duration must be finite and positive")
    row: dict[str, Any] = {"workload": WORKLOAD, "model": MODEL, "provider": arm, "arm": arm,
        "lc": lc, "cell": cell["id"], "n_inputs": len(texts), "audio_seconds": audio_seconds,
        "words": sum(len(normalize_text(reference).split()) for reference in references),
        "ms": round(elapsed_ms, 3), "cell_wall_s": elapsed_ms / 1000,
        "request_ms": [round(elapsed_ms, 3)], "requests_per_second": 1000 / elapsed_ms,
        "audio_x_realtime": audio_seconds * 1000 / elapsed_ms,
        "audio_real_time_factor": elapsed_ms / (1000 * audio_seconds),
        "quality": quality(texts, references), "ok": True}
    if extra:
        row.update(extra)
    return row


def _record_error(info: dict[str, Any], exc: Exception, stage: str, **identity: Any) -> dict[str, Any]:
    fields = exc.call_fields() if isinstance(exc, C.HttpResponseError) else {
        "ok": False, "error": public_error(exc)}
    refused = C.call_refused(fields)
    detail = {**identity, **fields, "stage": stage, "state": "refused" if refused else "failed"}
    info.setdefault("errors", []).append(detail)
    if not refused:
        info.setdefault("error", fields["error"])
    return detail


def _measure_cells(info: dict[str, Any], cells: list[dict[str, Any]], lc: int,
                   repetitions: int, prepare, extra) -> None:
    """Retain each actual attempt; a failed cell never replaces earlier evidence."""
    warmups = info["warmup"]["cells"]
    for cell in cells:
        try:
            infer = prepare(cell)
        except Exception as exc:
            _record_error(info, exc, "prepare", cell=cell["id"])
            continue
        warm_state = "complete"
        try:
            texts, elapsed_ms, _ = infer()
            warmups.append({"cell": cell["id"], "state": "complete",
                            "ms": round(elapsed_ms, 3),
                            "nonempty": bool(normalize_text(texts[0])), "measured": False})
            info["warmup"]["performed"] = True
            info["warmup"]["inference_count"] += 1
        except Exception as exc:
            detail = _record_error(info, exc, "warmup", cell=cell["id"])
            warmups.append({**detail, "measured": False})
            warm_state = detail["state"]
            if warm_state != "refused":
                continue
        for repetition in range(repetitions):
            try:
                texts, elapsed_ms, details = infer()
                call = _cell_row(info["arm"], cell, texts, elapsed_ms, lc, {
                    **extra(cell), **details, "repetition": repetition,
                    "warmup_state": warm_state})
                row = {"call": call, "texts": texts, "refs": [item["ref"] for item in cell["calls"]]}
            except Exception as exc:
                detail = _record_error(info, exc, "measured", cell=cell["id"], repetition=repetition)
                row = {"call": {"workload": WORKLOAD, "model": MODEL, "provider": info["arm"],
                    "arm": info["arm"], "lc": lc, "n_inputs": len(cell["calls"]),
                    "input_ids": [item["id"] for item in cell["calls"]],
                    "warmup_state": warm_state, **detail}, "texts": [], "refs": []}
            info["rows"].append(row)
            if not row["call"]["ok"] and not C.call_refused(row["call"]):
                break


def _capture_stage(info: dict[str, Any], stage: str, action, key: str | None = None) -> None:
    try:
        value = action()
        if key is not None:
            info[key] = value
    except Exception as exc:
        _record_error(info, exc, stage)


def run_antfly(args: argparse.Namespace, cells: list[dict[str, Any]], out: Path, lc: int) -> dict[str, Any]:
    tag = f"lc{lc}-antfly"
    requests = {cell["id"]: _antfly_request(cell["calls"][0]) for cell in cells}
    server = C.AntflyServer(args.antfly, out, tag, models_dir=C.MODELS_DIR)
    info: dict[str, Any] = {"provider": "antfly", "arm": "antfly", "model": MODEL,
                            "binary": C.file_identity(args.antfly), "host_before": C.host_snapshot(),
                            "decoding": {"language": "en", "strategy": "greedy",
                                         "timestamps": False, "audio_context": "full"},
                            "precision": C.native_precision(args.antfly, MODEL)}
    info["rows"] = []
    info["warmup"] = {"performed": False, "inference_count": 0, "cells": [],
                     "state": "one independent inference per cell before its measured repetitions; same loaded process"}
    try:
        info.update(server.start())

        def prepare(cell):
            def infer():
                text, elapsed_ms = _antfly_one(server.port, requests[cell["id"]])
                return [text], elapsed_ms, {}
            return infer

        _measure_cells(info, cells, lc, args.repetitions, prepare, lambda cell: {
            "timing_basis": "pre-encoded HTTP request dispatch through complete response-body read; WAV/base64/JSON preparation excluded"})
    except Exception as exc:
        _record_error(info, exc, "lifecycle")
    finally:
        _capture_stage(info, "host_after", C.host_snapshot, "host_after")
        _capture_stage(info, "cleanup", server.stop, "footprint")
        _capture_stage(info, "log_cleanup", lambda: sanitize_log(out / f"{tag}.server.log"))
    return info


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
            "comparison": "SHA-256 of model.safetensors in both installed snapshots"}


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
    if converter.get("name") != "whisper.cpp" or converter.get("revision") != CPP_REVISION:
        raise ValueError("whisper.cpp converter must be pinned to v1.9.2")
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


CPP_FIELDS = {"language": "en", "translate": "false", "best_of": "1", "beam_size": "1",
              "temperature": "0", "temperature_inc": "0", "response_format": "json",
              "no_timestamps": "true", "audio_ctx": "0"}


def _cpp_request(path: Path) -> tuple[bytes, str]:
    boundary = "termite-whisper-" + os.urandom(16).hex()
    parts = []
    for name, value in CPP_FIELDS.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
    parts.extend([path.read_bytes(), f"\r\n--{boundary}--\r\n".encode()])
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _cpp_one(port: int, request: tuple[bytes, str]) -> tuple[str, float]:
    body, content_type = request
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    started = time.perf_counter()
    try:
        connection.request("POST", "/inference", body=body, headers={"Content-Type": content_type})
        response = connection.getresponse()
        raw = response.read()
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        connection.close()
    if response.status != 200:
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            payload = None
        raise C.HttpResponseError(response.status, payload)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise RuntimeError("whisper.cpp inference returned a malformed successful response")
    return payload["text"].strip(), elapsed_ms


def _cpp_precision(model: Path, log: str) -> dict[str, Any]:
    import struct
    # ggml header: magic, n_vocab, n_audio_ctx/state/head/layer,
    # n_text_ctx/state/head/layer, n_mels, ftype (12 little-endian int32s).
    with model.open("rb") as stream:
        header = stream.read(48)
    if len(header) != 48:
        raise ValueError("truncated whisper.cpp GGML model header")
    fields = struct.unpack("<12i", header)
    if fields[0] != 0x67676D6C:
        raise ValueError("expected whisper.cpp GGML model")
    observed = [line.strip() for line in log.splitlines()
                if re.search(r"(ftype|type\s*=|backend|device|buffer|use_gpu|flash_attn)", line, re.I)]
    return {"converted_model": C.file_identity(model), "ggml_header_ftype": fields[-1],
            "weight_type": {0: "F32", 1: "mostly F16 (mixed tensor types)"}.get(fields[-1], "quantized/mixed; see ftype"),
            "observed_runtime_log": observed[:100],
            "compute_dtype": "not exposed as a complete per-operation runtime dtype audit",
            "proof": "GGML header observed from loaded file plus server launch log; not inferred from scores"}


def run_cpp_arm(cells: list[dict[str, Any]], out: Path, lc: int,
                repetitions: int = 10) -> dict[str, Any]:
    binary, model, provenance = _cpp_paths()
    receipt = validate_cpp_provenance(model, provenance)
    port = C.free_port()
    tag = f"lc{lc}-whisper-cpp"
    log_path = out / f"{tag}.log"
    command = [str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
               "--language", "en", "--no-timestamps", "--best-of", "1", "--beam-size", "1",
               "--no-fallback"]
    prepared = {cell["id"]: _cpp_request(Path(cell["calls"][0]["path"])) for cell in cells}
    info: dict[str, Any] = {"provider": "whisper.cpp", "arm": "whisper.cpp", "model": MODEL,
        "binary": C.file_identity(binary), "command": C.public_command(command),
        "provenance": receipt, "decoding": CPP_FIELDS,
        "upstream": "https://github.com/ggml-org/whisper.cpp/blob/v1.9.2/examples/server/server.cpp",
        "context_policy": "upstream server no_context=true; independent requests, no carried text history",
        "rows": [], "warmup": {"performed": False, "inference_count": 0, "cells": [],
            "state": "one independent inference per cell before its measured repetitions; same loaded process"}}
    process = sampler = None
    started = time.perf_counter()
    log = log_path.open("w", encoding="utf-8")
    try:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
            info["server_pid"] = process.pid
            if C.memory_sampling_enabled():
                sampler = C.Sampler(process.pid, out, tag)
            info["ready_s"] = C.wait_http(process, port, "/health", 900, log_path.name, started)

            def prepare(cell):
                def infer():
                    text, elapsed_ms = _cpp_one(port, prepared[cell["id"]])
                    return [text], elapsed_ms, {}
                return infer

            _measure_cells(info, cells, lc, repetitions, prepare, lambda cell: {
                "timing_basis": "pre-encoded multipart HTTP dispatch through complete response-body read; WAV read and multipart encoding excluded"})
        except Exception as exc:
            _record_error(info, exc, "lifecycle")
        finally:
            if sampler is not None:
                _capture_stage(info, "sampler_cleanup", sampler.stop, "footprint")
            _capture_stage(info, "process_cleanup", lambda: C.stop_proc(process))
            _capture_stage(info, "log_flush", log.flush)
            _capture_stage(info, "log_cleanup", lambda: sanitize_log(log_path))
    finally:
        _capture_stage(info, "log_close", log.close)
    _capture_stage(info, "precision", lambda: _cpp_precision(model, log_path.read_text(encoding="utf-8")), "precision")
    return info


def _read_wav(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError("expected mono 16-bit 16 kHz WAV")
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype("float32") / 32768.0


def transformers_arm(only: str | None, lc: int, repetitions: int = 10,
                     cells: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    cells = make_cells(only) if cells is None else cells
    reference = Path(C.reference_paths()[MODEL])
    device = C.ACCEL
    processor = AutoProcessor.from_pretrained(reference, local_files_only=True)
    # Audit the library's load policy instead of forcing FP32 as a benchmark override.
    model = AutoModelForSpeechSeq2Seq.from_pretrained(reference, local_files_only=True).to(device).eval()
    precision = C.reference_precision(model)
    recognizer = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                          feature_extractor=processor.feature_extractor, device=device)
    generate = {"language": "en", "task": "transcribe", "do_sample": False, "num_beams": 1}

    prepared = {cell["id"]: {"raw": _read_wav(Path(cell["calls"][0]["path"])),
                              "sampling_rate": 16000} for cell in cells}

    def infer(inputs: dict[str, Any]) -> list[str]:
        # The pipeline pops audio and sampling_rate; retain the prepared audio.
        result = recognizer(dict(inputs), return_timestamps=False, generate_kwargs=generate)
        return [result["text"]]

    sync = C.torch_sync(torch)
    info = {"provider": "transformers", "arm": "transformers", "model": MODEL,
            "precision": precision, "device": device, "rows": [],
            "warmup": {"performed": False, "inference_count": 0, "cells": [],
                       "state": "one independent inference per cell before its measured repetitions; same loaded process"},
            "decoding": {"language": "en", "task": "transcribe", "do_sample": False,
                         "num_beams": 1, "return_timestamps": False}}

    def prepare(cell):
        def measured():
            sync()
            started = time.perf_counter()
            texts = infer(prepared[cell["id"]])
            sync()
            return texts, (time.perf_counter() - started) * 1000, {}
        return measured

    _capture_stage(info, "measurement", lambda: _measure_cells(
        info, cells, lc, repetitions, prepare, lambda cell: {
            "timing_basis": "ASR pipeline feature extraction, model inference, and decode; WAV file read excluded"}))
    _capture_stage(info, "versions", lambda: {
        package: importlib.metadata.version(package) for package in ("torch", "transformers", "numpy")}, "versions")
    _capture_stage(info, "device_memory", lambda: C.torch_device_bytes(torch), "device_memory")
    return info


def pyarm_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["auto", "cuda", "metal"])
    parser.add_argument("--only-cell", choices=CELLS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lc", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args(argv)
    C.configure_backend(args.backend)
    try:
        result = transformers_arm(args.only_cell, args.lc, args.repetitions)
    except Exception as exc:  # child must return a structured terminal record
        result = {"provider": "transformers", "arm": "transformers", "model": MODEL,
                  "error": public_error(exc)}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if "error" not in result else 1


def agreement_rows(lc: int, cells: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for cell in cells:
        by_arm: dict[str, list[str]] = {}
        repetitions: dict[str, list[int]] = {}
        for arm in PROVIDERS:
            result = results.get(arm) or {}
            matches = [item for item in result.get("rows", []) if item["call"]["cell"] == cell["id"]
                       and item["call"].get("ok", True)]
            if matches:
                by_arm[arm] = [text for item in matches for text in item["texts"]]
                repetitions[arm] = [item["call"].get("repetition", 0) for item in matches]
        row: dict[str, Any] = {"workload": WORKLOAD, "model": MODEL, "lc": lc,
                               "cell": cell["id"], "providers": list(by_arm),
                               "complete": len(by_arm) == len(PROVIDERS)
                               and len({tuple(values) for values in repetitions.values()}) == 1}
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
                               "arm": result["arm"], "lc": lc, "cell": row["call"]["cell"],
                               "repetition": row["call"]["repetition"], "k": index,
                               "text": text, "ref": row["refs"][index]})
    finally:
        calls.close()
        outputs.close()


def orchestrate(args: argparse.Namespace) -> int:
    C.configure_backend(args.backend)
    C.exclusive_bench_lock()
    out = Path(args.out)
    C.prepare_output(out)
    cells = make_cells(args.only_cell)
    same_weights = verify_same_weights()
    cpp_binary, cpp_model, cpp_provenance_path = _cpp_paths()
    cpp_provenance = validate_cpp_provenance(cpp_model, cpp_provenance_path)
    C.write_json(out / "model_files.json", {"transformers_and_antfly": same_weights,
                 "whisper_cpp": cpp_provenance, "whisper_cpp_binary": C.file_identity(cpp_binary)})
    C.write_json(out / "inputs.json", {"source": {"dataset": "hf-internal-testing/librispeech_asr_dummy",
                 "revision": "5be91486e11a2d616f4ec5db8d3fd248585ac07a",
                 "source_file": "clean/validation-00000-of-00001.parquet",
                 "source_sha256": "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603",
                 "license": "CC-BY-4.0", "license_url": "https://www.openslr.org/12",
                 "construction": "one intact natural source recording per cell; no concatenation, padding, repetition or fanout"},
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
                        child_args = ["--backend", args.backend, "--out", str(out), "--lc", str(lc),
                                      "--repetitions", str(args.repetitions)]
                        if args.only_cell:
                            child_args.extend(["--only-cell", args.only_cell])
                        result = C.run_python_arm(str(Path(__file__)), child_args, out, f"lc{lc}-transformers")
                        _capture_stage(result, "log_cleanup", lambda: sanitize_log(out / f"lc{lc}-transformers.py.log"))
                    else:
                        result = run_cpp_arm(cells, out, lc, args.repetitions)
                except Exception as exc:
                    failed = True
                    result = {"provider": arm, "arm": arm, "model": MODEL, "error": public_error(exc)}
                if result.get("rc", 0) != 0 or "error" in result:
                    failed = True
                _write_arm(out, result, lc)
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
    complete = C.benchmark_complete(out, set(PROVIDERS), args.lifecycles, len(cells) * args.repetitions)
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
    run.add_argument("--repetitions", type=int, default=10)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--only-cell", choices=CELLS)
    args = parser.parse_args(argv)
    if args.lifecycles < 1 or args.repetitions < 1:
        parser.error("--lifecycles and --repetitions must be positive")
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
