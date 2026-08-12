#!/usr/bin/env python3
"""Local reference-provider adapter for the termite-circus JSON protocol."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def timed(iterations: int, operation: Callable[[], Any]) -> tuple[list[float], Any]:
    operation()  # warmup
    latencies: list[float] = []
    value: Any = None
    for _ in range(iterations):
        started = time.perf_counter()
        value = operation()
        latencies.append((time.perf_counter() - started) * 1000)
    return latencies, value


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def normalized_vectors(value: Any) -> list[list[float]]:
    if hasattr(value, "toarray"):
        value = value.toarray()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [[float(item) for item in row] for row in value]


def result(latencies: list[float], quality_gate: str, quality_value: float, quality_unit: str, response: Any) -> dict[str, Any]:
    return {
        "status": "ok",
        "latencies_ms": latencies,
        "throughput_per_second": 1000 * len(latencies) / sum(latencies) if sum(latencies) else None,
        "quality_gate": quality_gate,
        "quality_value": quality_value,
        "quality_unit": quality_unit,
        "response": response,
    }
def quality_failure(
    latencies: list[float],
    quality_gate: str,
    reason: str,
    quality_value: float | None,
    quality_unit: str | None,
    response: Any,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": f"quality gate failed: {reason}",
        "latencies_ms": latencies,
        "throughput_per_second": 1000 * len(latencies) / sum(latencies) if sum(latencies) else None,
        "quality_gate": quality_gate,
        "quality_value": quality_value,
        "quality_unit": quality_unit,
        "response": response,
        "terminal_negative": {
            "kind": "quality-gate",
            "stage": "quality-gate",
            "reason": reason,
            "observed": {
                "quality_value": quality_value,
                "quality_unit": quality_unit,
                "passed": False,
            },
        },
    }




def sentence_transformers(request: dict[str, Any], onnx: bool) -> dict[str, Any]:
    bench = request["benchmark"]
    fixture = request["fixture"]
    model_name = bench.get("reference_model") or bench["model"]
    category = bench["category"]
    backend = "onnx" if onnx else "torch"
    if category == "dense embedding":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, backend=backend)
        latencies, embeddings = timed(request["iterations"], lambda: model.encode(fixture["texts"], normalize_embeddings=True))
        vectors = normalized_vectors(embeddings)
        margin = cosine(vectors[0], vectors[1]) - cosine(vectors[0], vectors[2])
        if margin <= 0:
            return quality_failure(
                latencies, "relevant cosine margin", "relevant cosine margin",
                margin, "cosine", {"dimensions": len(vectors[0])},
            )
        return result(latencies, "relevant cosine margin", margin, "cosine", {"dimensions": len(vectors[0])})
    if category == "sparse embedding":
        from sentence_transformers import SparseEncoder

        model = SparseEncoder(model_name, backend=backend)
        latencies, embeddings = timed(request["iterations"], lambda: model.encode(fixture["texts"]))
        if hasattr(embeddings, "to_dense"):
            embeddings = embeddings.to_dense()
        vectors = normalized_vectors(embeddings)
        nonzero = sum(sum(value != 0 for value in vector) for vector in vectors)
        if nonzero < fixture["minimum_nonzero"]:
            return quality_failure(
                latencies, "nonzero sparse entries", "no sparse entries",
                float(nonzero), "entries", {"vectors": len(vectors)},
            )
        return result(latencies, "nonzero sparse entries", float(nonzero), "entries", {"vectors": len(vectors)})
    if category == "reranking":
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name, backend=backend)
        pairs = [(fixture["query"], document) for document in fixture["documents"]]
        latencies, scores = timed(request["iterations"], lambda: model.predict(pairs))
        values = [float(value) for value in scores]
        top = max(range(len(values)), key=values.__getitem__)
        if top != fixture["expected_top_index"]:
            return quality_failure(
                latencies, "expected document ranked first", f"document {top} ranked first",
                0.0, "pass", {"scores": values},
            )
        return result(latencies, "expected document ranked first", 1.0, "pass", {"scores": values})
    if category == "multimodal embedding" and onnx:
        fail("the ClipClap exact-artifact ONNX adapter requires its model_manifest input map; no safe generic fallback exists")
    fail(f"sentence-transformers adapter does not support {category}")


def gliner2_provider(request: dict[str, Any]) -> dict[str, Any]:
    from gliner2 import GLiNER2

    fixture = request["fixture"]
    model_name = request["benchmark"].get("reference_model") or request["benchmark"]["model"]
    model = GLiNER2.from_pretrained(model_name)
    operation = lambda: model.extract_json(fixture["texts"][0], fixture["schema"])
    latencies, response = timed(request["iterations"], operation)
    raw = json.dumps(response, default=str).lower()
    found = sum(value.lower() in raw for value in fixture["expected"])
    recall = found / len(fixture["expected"])
    if recall < 1:
        return quality_failure(
            latencies, "expected extraction values", f"expected extraction recall {recall}",
            recall, "recall", response,
        )
    return result(latencies, "expected extraction values", recall, "recall", response)


def tokenizer_provider(request: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    bench = request["benchmark"]
    fixture = request["fixture"]
    if bench["model"] == "fixed_bert":
        model_name = "google-bert/bert-base-uncased"
    elif bench["model"] == "fixed_bpe":
        model_name = "openai-community/gpt2"
    else:
        model_name = bench["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def operation() -> list[list[int]]:
        tokens = tokenizer.encode(fixture["text"], add_special_tokens=False)
        target = fixture["target_tokens"]
        overlap = fixture["overlap_tokens"]
        step = max(1, target - overlap)
        return [tokens[start : start + target] for start in range(0, len(tokens), step) if tokens[start : start + target]]

    latencies, chunks = timed(request["iterations"], operation)
    if not chunks:
        return quality_failure(
            latencies, "nonempty ordered chunks", "no chunks",
            0.0, "chunks", {"chunk_lengths": []},
        )
    return result(latencies, "nonempty ordered chunks", float(len(chunks)), "chunks", {"chunk_lengths": [len(chunk) for chunk in chunks]})


def transformers_ocr(request: dict[str, Any]) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    bench = request["benchmark"]
    fixture = request["fixture"]
    model_name = bench.get("reference_model") or bench["model"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    image = Image.open(Path(request["root"]) / fixture["image"]).convert("RGB")

    def operation() -> str:
        inputs = processor(text="<OCR>", images=image, return_tensors="pt")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=128)
        return processor.batch_decode(generated, skip_special_tokens=True)[0]

    latencies, text = timed(request["iterations"], operation)
    expected = fixture["expected_text"].lower().split()
    found = sum(word in text.lower() for word in expected)
    recall = found / len(expected)
    if recall < 1:
        return quality_failure(
            latencies, "OCR expected words", f"OCR word recall {recall}",
            recall, "word recall", {"text": text},
        )
    return result(latencies, "OCR expected words", recall, "word recall", {"text": text})


def faster_whisper_provider(request: dict[str, Any]) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    fixture = request["fixture"]
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    audio = str(Path(request["root"]) / fixture["audio"])

    def operation() -> str:
        segments, _ = model.transcribe(audio, language="en", beam_size=1)
        return " ".join(segment.text.strip() for segment in segments)

    latencies, text = timed(request["iterations"], operation)
    expected = fixture["expected_text"].lower().split()
    found = sum(word in text.lower() for word in expected)
    recall = found / len(expected)
    if recall < 0.8:
        return quality_failure(
            latencies, "transcription expected words", f"transcription word recall {recall}",
            recall, "word recall", {"text": text},
        )
    return result(latencies, "transcription expected words", recall, "word recall", {"text": text})


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, body: dict[str, Any] | None = None, timeout: float = 600) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        fail(f"expected object from {url}")
    return value


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def wait_http(url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            fail(f"provider server exited with status {process.returncode}")
        try:
            http_json(url, timeout=2)
            return
        except Exception:
            time.sleep(0.25)
    fail(f"provider server did not become ready: {url}")


def generation_provider(request: dict[str, Any]) -> dict[str, Any]:
    provider = request["provider"]
    bench = request["benchmark"]
    fixture = request["fixture"]
    iterations = request["iterations"]
    model_path = Path(request["model_path"])
    if model_path.is_dir():
        candidates = sorted(model_path.rglob("*.gguf"))
        if not candidates:
            fail(f"no GGUF artifact found under {model_path}")
        model_path = candidates[0]
    env_key = re.sub(r"[^A-Za-z0-9]", "_", bench["id"]).upper()
    if provider == "llama.cpp":
        binary = os.environ.get("LLAMA_CLI", "llama-cli")

        def operation() -> str:
            proc = subprocess.run(
                [binary, "-m", str(model_path), "-p", fixture["prompt"], "-n", str(fixture["max_tokens"]), "--ignore-eos", "--no-display-prompt"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=900,
                check=False,
            )
            if proc.returncode:
                fail(proc.stdout[-2000:])
            return proc.stdout

        latencies, output = timed(iterations, operation)
        if not output.strip():
            return quality_failure(
                latencies, "nonempty completion", "llama.cpp produced no output",
                0.0, "pass", {"output_tail": output[-1000:]},
            )
        return result(latencies, "nonempty completion", 1.0, "pass", {"output_tail": output[-1000:]})
    if provider == "ollama":
        binary = os.environ.get("OLLAMA_BIN", "ollama")
        model = os.environ.get(f"OLLAMA_MODEL_{env_key}") or os.environ.get("OLLAMA_MODEL")
        if not model:
            fail(f"set OLLAMA_MODEL_{env_key} to the imported provider-native model")
        port = free_port()
        host = f"127.0.0.1:{port}"
        env = os.environ.copy()
        env["OLLAMA_HOST"] = host
        process = subprocess.Popen([binary, "serve"], stdout=sys.stderr, stderr=sys.stderr, env=env, start_new_session=True, text=True)
        try:
            wait_http(f"http://{host}/api/tags", process)

            def operation() -> dict[str, Any]:
                return http_json(f"http://{host}/api/generate", {"model": model, "prompt": fixture["prompt"], "stream": False, "options": {"num_predict": fixture["max_tokens"], "temperature": 0}})

            latencies, response = timed(iterations, operation)
        finally:
            stop_process(process)
        if not str(response.get("response", "")).strip():
            return quality_failure(
                latencies, "nonempty completion", "Ollama produced no completion",
                0.0, "pass", response,
            )
        return result(latencies, "nonempty completion", 1.0, "pass", response)
    if provider == "vllm-metal":
        binary = os.environ.get("VLLM_METAL_BIN", "vllm-metal")
        model = os.environ.get(f"VLLM_MODEL_{env_key}") or os.environ.get("VLLM_MODEL")
        if not model:
            fail(f"set VLLM_MODEL_{env_key} to a local provider-native model or cache entry")
        port = free_port()
        process = subprocess.Popen([binary, "serve", model, "--port", str(port), "--host", "127.0.0.1"], stdout=sys.stderr, stderr=sys.stderr, start_new_session=True, text=True)
        try:
            wait_http(f"http://127.0.0.1:{port}/v1/models", process)

            def operation() -> dict[str, Any]:
                return http_json(f"http://127.0.0.1:{port}/v1/completions", {"model": model, "prompt": fixture["prompt"], "max_tokens": fixture["max_tokens"], "temperature": 0})

            latencies, response = timed(iterations, operation)
        finally:
            stop_process(process)
        choices = response.get("choices") or []
        if not choices or not str(choices[0].get("text", "")).strip():
            return quality_failure(
                latencies, "nonempty completion", "vLLM-Metal produced no completion",
                0.0, "pass", response,
            )
        return result(latencies, "nonempty completion", 1.0, "pass", response)
    fail(f"unsupported generation provider {provider}")


def whisper_cpp_provider(request: dict[str, Any]) -> dict[str, Any]:
    binary = os.environ.get("WHISPER_CLI", "whisper-cli")
    model = os.environ.get("WHISPER_CPP_MODEL")
    if not model:
        fail("set WHISPER_CPP_MODEL to a whisper.cpp tiny model")
    audio = str(Path(request["root"]) / request["fixture"]["audio"])

    def operation() -> str:
        proc = subprocess.run([binary, "-m", model, "-f", audio, "-l", "en", "-nt"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900, check=False)
        if proc.returncode:
            fail(proc.stdout[-2000:])
        return proc.stdout

    latencies, text = timed(request["iterations"], operation)
    expected = request["fixture"]["expected_text"].lower().split()
    found = sum(word in text.lower() for word in expected)
    recall = found / len(expected)
    if recall < 0.8:
        return quality_failure(
            latencies, "transcription expected words", f"transcription word recall {recall}",
            recall, "word recall", {"output_tail": text[-1000:]},
        )
    return result(latencies, "transcription expected words", recall, "word recall", {"output_tail": text[-1000:]})


def main() -> int:
    request = json.load(sys.stdin)
    provider = request["provider"]
    category = request["benchmark"]["category"]
    if provider in {"llama.cpp", "ollama", "vllm-metal"}:
        response = generation_provider(request)
    elif provider == "onnxruntime":
        response = sentence_transformers(request, onnx=True)
    elif provider == "sentence-transformers":
        response = sentence_transformers(request, onnx=False)
    elif provider == "gliner2":
        response = gliner2_provider(request)
    elif provider == "tokenizers":
        response = tokenizer_provider(request)
    elif provider == "transformers" and category == "reading / OCR":
        response = transformers_ocr(request)
    elif provider == "faster-whisper":
        response = faster_whisper_provider(request)
    elif provider == "whisper.cpp":
        response = whisper_cpp_provider(request)
    else:
        fail(f"unsupported provider/category: {provider}/{category}")
    json.dump(response, sys.stdout, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
