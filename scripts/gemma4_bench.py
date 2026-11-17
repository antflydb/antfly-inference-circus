#!/usr/bin/env python3
"""Benchmark Gemma 4 E4B through Antfly and llama.cpp using identical GGUF files."""
from __future__ import annotations

import argparse
import base64
import http.client
import hashlib
import json
import math
import os
import shutil
import re
import subprocess
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODEL = "google/gemma-4-E4B-it-qat-q4_0-gguf"
MODEL_KEY = "gemma4-e4b"
PROVIDERS = ["antfly", "llama.cpp"]
HF_REVISION = "4b4a2c1d584be7264f87aac328a1bc739ce81b6c"
FILES = {
    "gguf": ("gemma-4-E4B_q4_0-it.gguf", "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee"),
    "mmproj": ("gemma-4-E4B-it-mmproj.gguf", "7498a37cb619e55f2fcf87eb931f56e99389ed6d432e4c5c66110694c0d65578"),
}
GEN_MODELS_DIR = Path(os.environ.get("TERMITE_GEN_MODELS_DIR", C.MODELS_DIR)).expanduser()
MEDIA_MANIFEST = C.ROOT / "media_manifest.json"
CONTEXT_SIZE = 8192
HOST_BUDGET_MB = 8192
IMAGE_MAX_TOKENS = 280
CELLS = ["text_short_256", "page_summary_128", "long2k_summary_64", "image_describe_64", "audio_transcribe_64"]


def model_dir() -> Path:
    owner, name = MODEL.split("/")
    owner_dir = GEN_MODELS_DIR / owner
    candidates = sorted(path for path in owner_dir.glob(f"{name}--antfly-*") if path.is_dir())
    flat = GEN_MODELS_DIR / MODEL
    if flat.is_dir():
        candidates.append(flat)
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise RuntimeError(f"expected one pulled {MODEL} model, found {len(unique)}")
    return unique[0]


def verify_files() -> dict[str, dict | str]:
    directory = model_dir()
    verified: dict[str, dict | str] = {"model": MODEL, "hf_revision": HF_REVISION}
    for key, (filename, expected) in FILES.items():
        path = directory / filename
        actual = C.sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{filename}: sha256 {actual} does not match expected {expected}")
        verified[key] = {
            "name": filename,
            "sha256": actual,
            "size_bytes": path.stat().st_size,
            "path": path,
        }
    return verified


def public_verified_files(files: dict) -> dict:
    return {
        "model": files["model"],
        "hf_revision": files["hf_revision"],
        **{
            key: {field: value for field, value in files[key].items() if field != "path"}
            for key in FILES
        },
    }


def gguf_inventory(path: Path) -> dict:
    """Read tensor descriptors, not tensor payloads or the tokenizer vocabulary."""
    types = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
             8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
             13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with path.open("rb") as handle:
        def unpack(fmt):
            size = struct.calcsize("<" + fmt)
            data = handle.read(size)
            if len(data) != size:
                raise ValueError("truncated GGUF header")
            return struct.unpack("<" + fmt, data)[0]

        def string(keep=False):
            length = unpack("Q")
            if length > path.stat().st_size - handle.tell():
                raise ValueError("invalid GGUF string length")
            if keep:
                return handle.read(length).decode("utf-8")
            handle.seek(length, 1)

        def skip(kind):
            if kind == 8:
                string()
            elif kind == 9:
                element, count = unpack("I"), unpack("Q")
                if element in sizes:
                    handle.seek(sizes[element] * count, 1)
                else:
                    for _ in range(count):
                        skip(element)
            elif kind in sizes:
                handle.seek(sizes[kind], 1)
            else:
                raise ValueError(f"unsupported GGUF metadata type {kind}")

        if handle.read(4) != b"GGUF":
            raise ValueError("not a GGUF checkpoint")
        version = unpack("I")
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        tensors, metadata = unpack("Q"), unpack("Q")
        for _ in range(metadata):
            string()
            skip(unpack("I"))
        counts, heads = {}, {}
        for _ in range(tensors):
            name = string(True)
            dimensions = unpack("I")
            shape = [unpack("Q") for _ in range(dimensions)]
            kind = unpack("I")
            unpack("Q")  # Data offset; payloads are intentionally not read.
            label = types.get(kind, f"GGML_TYPE_{kind}")
            row = counts.setdefault(label, {"tensors": 0, "parameters": 0})
            row["tensors"] += 1
            row["parameters"] += math.prod(shape)
            if name in ("token_embd.weight", "output.weight", "output_norm.weight"):
                heads[name] = {"storage_type": label, "shape": shape}
        if handle.tell() > path.stat().st_size:
            raise ValueError("truncated GGUF tensor descriptors")
    return {"gguf_version": version, "tensor_count": tensors, "storage_types": counts,
            "embedding_and_output_tensors": heads,
            "scope": "checkpoint storage, not runtime arithmetic or KV dtype"}


def loading_evidence(server, files: dict, arm: str) -> dict:
    log_path = Path(server.log.name) if arm == "llama" else server.out_dir / f"{server.tag}.server.log"
    excerpts = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if re.search(r"(type\s+\w+:|KV.*(type|size)|cache.*(type|dtype)|mmap|model buffer size|mmproj|projector)", line, re.I):
                excerpts.append(C.public_error(line)[:400])
    return {
        "checkpoints": public_verified_files(files),
        "tensor_inventory": {key: gguf_inventory(files[key]["path"]) for key in FILES},
        "requested_kv": {"key": "f16", "value": "f16", "prompt_cache": False},
        "native_source_policy": C.native_precision(server.binary, MODEL, models_dir=GEN_MODELS_DIR)["native_policy"] if arm == "antfly" else None,
        "mmap": {"launch_override": None, "observed_log_lines": excerpts[:60],
                 "note": "No mmap override is passed; source policy and observed logs are distinct from resident bytes."},
        "projector": {"file": files["mmproj"]["name"], "sha256": files["mmproj"]["sha256"],
                      "llama_flag": "--mmproj", "native_selection": "model manifest"},
    }


def prompt_fingerprints(prompt: str, tokens: list[int] | None = None) -> dict:
    """Retain identity evidence, never the diagnostic prompt or token sequence."""
    data = prompt.encode("utf-8")
    evidence = {"rendered_prompt_sha256": hashlib.sha256(data).hexdigest(),
                "rendered_prompt_bytes": len(data)}
    if tokens is not None:
        encoded = json.dumps(tokens, separators=(",", ":")).encode("ascii")
        evidence.update(prompt_token_ids_sha256=hashlib.sha256(encoded).hexdigest(),
                        prompt_token_count=len(tokens))
    return evidence


def llama_prompt_evidence(port: int, payload: dict) -> dict:
    """Observe llama's formatting/tokenizer endpoints, outside timed operations."""
    client = C.Client(port)
    try:
        status, rendered, _ = client.post("/apply-template", {
            "messages": payload["messages"], "add_generation_prompt": True,
            "chat_template_kwargs": payload["chat_template_kwargs"],
        })
        if status != 200 or not isinstance(rendered.get("prompt"), str):
            return {"status": "unavailable", "apply_template_http_status": status}
        prompt = rendered["prompt"]
        status, encoded, _ = client.post("/tokenize", {
            "content": prompt, "add_special": True, "parse_special": True,
        })
        ids = encoded.get("tokens")
        if status != 200 or not isinstance(ids, list) or any(type(x) is not int for x in ids):
            return {"status": "unavailable", **prompt_fingerprints(prompt), "tokenize_http_status": status,
                    "scope": "diagnostic endpoints, not an HTTP generation input trace"}
        return {"status": "observed", "source": "llama.cpp /apply-template + /tokenize",
                **prompt_fingerprints(prompt, ids),
                "scope": "diagnostic endpoints, not an HTTP generation input trace"}
    except Exception as exc:
        return {"status": "unavailable", "error": C.public_error(exc)}
    finally:
        client.conn.close()


def parse_native_prompt(text: str) -> dict:
    ids = re.search(r"(?m)^prompt_token_ids:([^\n]*)", text)
    rendered = re.search(r"(?ms)^prompt:\n(.*?)\nprompt_token_ids:", text)
    if ids is None or rendered is None:
        return {"status": "unavailable", "reason": "CLI did not emit rendering and prompt IDs"}
    try:
        tokens = [int(value) for value in ids.group(1).split()]
    except ValueError:
        return {"status": "unavailable", "reason": "invalid CLI prompt token-ID output"}
    return {"status": "observed", "source": "antfly inference generate CLI diagnostic",
            **prompt_fingerprints(rendered.group(1), tokens),
            "scope": "CLI pre-media-expansion encoding, NOT observed HTTP generation token IDs"}


def native_prompt_evidence(binary: Path, call: dict) -> dict:
    argv = [str(binary), "inference", "generate", str(model_dir()), call["text"],
            "--backend", C.ANTFLY_BACKEND, "--max-tokens", "1", "--temperature", "0",
            "--cache-dtype", "f16", "--disable-thinking", "--ignore-eos",
            "--print-prompt", "--print-prompt-token-ids"]
    for key in ("image", "audio"):
        if call.get(key):
            argv.extend(["--" + key, str(C.MEDIA / call[key])])
    env = dict(os.environ)
    env.pop("ANTFLY_INFERENCE_SERVER_URL", None)
    env.pop("ANTFLY_INFERENCE_REQUIRE_WARM_SERVER", None)
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env)
        observed = parse_native_prompt(result.stdout + "\n" + result.stderr)
        observed["exit_code"] = result.returncode
        observed["diagnostic_settings"] = {"backend": C.ANTFLY_BACKEND, "max_tokens": 1,
                                           "temperature": 0, "enable_thinking": False, "cache_dtype": "f16",
                                           "scope": "separate CLI operation; timed HTTP calls retain their original token caps"}
        observed["binary"] = C.file_identity(binary)
        return observed
    except Exception as exc:
        return {"status": "unavailable", "error": type(exc).__name__}


def compare_prompts(native: dict, reference: dict) -> dict:
    observed = native.get("status") == reference.get("status") == "observed"
    return {"diagnostic_token_ids_equal": (
                native["prompt_token_ids_sha256"] == reference["prompt_token_ids_sha256"]
                and native["prompt_token_count"] == reference["prompt_token_count"]) if observed else None,
            "diagnostic_rendering_equal": (
                native["rendered_prompt_sha256"] == reference["rendered_prompt_sha256"]
                and native["rendered_prompt_bytes"] == reference["rendered_prompt_bytes"]) if observed else None,
            "http_generation_prompt_parity": "unproven",
            "note": "Counts do not establish prompt parity; CLI and tokenizer diagnostics are not HTTP input traces."}


def make_cells(only: str | None = None) -> list[dict]:
    corpus = C.load_corpus()
    words = " ".join(chapter["text"] for chapter in corpus["chapters"]).split()
    long_texts = [" ".join(words[index * 1500 : (index + 1) * 1500]) for index in range(6)]
    media = json.loads(MEDIA_MANIFEST.read_text(encoding="utf-8"))
    cells = [
        {
            "id": "text_short_256",
            "max_tokens": 256,
            "warmup": 2,
            "calls": [{"text": f"Explain this sentence to a curious twelve-year-old:\n\n{text}"} for text in corpus["sentences"][:12]],
        },
        {
            "id": "page_summary_128",
            "max_tokens": 128,
            "warmup": 1,
            "calls": [{"text": f"Summarise this passage in one paragraph.\n\n{text}"} for text in corpus["pages"][:12]],
        },
        {
            "id": "long2k_summary_64",
            "max_tokens": 64,
            "warmup": 1,
            "calls": [{"text": f"Summarise the main argument of this passage in two sentences.\n\n{text}"} for text in long_texts],
        },
        {
            "id": "image_describe_64",
            "max_tokens": 64,
            "warmup": 1,
            "calls": [{"text": "Describe this image in one paragraph.", "image": item["name"]} for item in media["images"]],
        },
        {
            "id": "audio_transcribe_64",
            "max_tokens": 64,
            "warmup": 1,
            "calls": [
                {"text": "Transcribe this audio clip.", "audio": item["name"], "ref": item["text"]}
                for item in media["audio"]
            ],
        },
    ]
    return [cell for cell in cells if only in (None, cell["id"])]


def media_base64(name: str) -> str:
    return base64.b64encode((C.MEDIA / name).read_bytes()).decode()


def content(call: dict, engine: str) -> list[dict]:
    parts = []
    if call.get("image"):
        parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + media_base64(call["image"])}})
    if call.get("audio"):
        audio = media_base64(call["audio"])
        if engine == "antfly":
            parts.append({"type": "media", "mime_type": "audio/wav", "data": audio})
        else:
            parts.append({"type": "input_audio", "input_audio": {"format": "wav", "data": audio}})
    parts.append({"type": "text", "text": call["text"]})
    return parts


def request_body(call: dict, cell: dict, engine: str) -> dict:
    value = {
        "messages": [{"role": "user", "content": content(call, engine)}],
        "max_tokens": cell["max_tokens"],
        "temperature": 0,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if engine == "antfly":
        value.update(
            {
                "model": MODEL,
                "backend": C.ANTFLY_BACKEND,
                "cache_dtype": "f16",
                "prompt_cache": False,
            }
        )
    else:
        value["cache_prompt"] = False
    return value


def stream(port: int, path: str, payload: dict, timeout: float = 900) -> dict:
    started = time.perf_counter()
    try:
        return _stream(port, path, payload, timeout)
    except C.HttpResponseError as exc:
        return {**exc.call_fields(), "ms": (time.perf_counter() - started) * 1000}
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        return {"ok": False, "status": -1, "error": C.public_error(exc), "ms": (time.perf_counter() - started) * 1000}


def _stream(port: int, path: str, payload: dict, timeout: float) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    data = json.dumps(payload).encode()
    started = time.perf_counter()
    connection.request("POST", path, body=data, headers={"content-type": "application/json", "accept": "text/event-stream"})
    response = connection.getresponse()
    if response.status != 200:
        body = response.read()
        connection.close()
        try:
            body = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            body = None
        raise C.HttpResponseError(response.status, body)
    text, reasoning = [], []
    usage = timings = first_token = finish = None
    deltas = 0
    while True:
        line = response.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        chunk = line[5:].strip()
        if chunk == b"[DONE]":
            break
        event = json.loads(chunk)
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            piece, thought = delta.get("content") or "", delta.get("reasoning_content") or ""
            if piece or thought:
                deltas += 1
                first_token = first_token or time.perf_counter()
            text.append(piece)
            reasoning.append(thought)
            finish = choice.get("finish_reason") or finish
        usage = event.get("usage") or usage
        timings = event.get("timings") or timings
    ended = time.perf_counter()
    connection.close()
    return {
        "status": 200,
        "ms": (ended - started) * 1000,
        "ttft_ms": None if first_token is None else (first_token - started) * 1000,
        "decode_s": None if first_token is None else ended - first_token,
        "text": "".join(text),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "engine_timings": timings,
        "finish": finish,
        "deltas": deltas,
    }


class LlamaServer:
    def __init__(self, binary: str, files: dict, out: Path, tag: str):
        self.port = C.free_port()
        self.argv = [
            binary,
            "-m",
            str(files["gguf"]["path"]),
            "--mmproj",
            str(files["mmproj"]["path"]),
            "-ngl",
            "999",
            "-c",
            str(CONTEXT_SIZE),
            "-np",
            "1",
            "--image-max-tokens",
            str(IMAGE_MAX_TOKENS),
            "--jinja",
            "-ctk",
            "f16",
            "-ctv",
            "f16",
            "--no-cache-prompt",
            "-cram",
            "0",
            "--no-webui",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        self.log = (out / f"{tag}.server.log").open("w", encoding="utf-8")
        self.out, self.tag = out, tag
        self.process = None
        self.sampler = None

    def start(self) -> dict:
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            if C.memory_sampling_enabled():
                self.sampler = C.Sampler(self.process.pid, self.out, self.tag)
            ready = C.wait_http(self.process, self.port, "/health", 300, self.log.name, started)
        except Exception:
            self.stop()
            raise
        return {"ready_s": ready, "command": C.public_command(self.argv)}

    def stop(self) -> dict:
        try:
            return self.sampler.stop() if self.sampler else {}
        finally:
            self.sampler = None
            C.stop_proc(self.process)
            if not self.log.closed:
                self.log.close()


def llama_version(binary: str) -> str:
    process = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30, check=False)
    return " ".join((process.stdout + process.stderr).split())[:300]


def warmup_observation(response: dict) -> dict:
    return {"state": "complete" if response["status"] == 200 else "refused" if C.call_refused(response) else "failed",
            "status": response["status"], "ms": response["ms"],
            **{key: response[key] for key in ("error", "response_error") if key in response}}


def run_arm(args, arm: str, files: dict, cells: list[dict], out: Path, lifecycle: int) -> dict:
    tag = f"lc{lifecycle}-{arm}"
    if arm == "antfly":
        server = C.AntflyServer(
            args.antfly,
            out,
            tag,
            models_dir=GEN_MODELS_DIR,
            extra_args=[
                "--preload-model",
                f"generator:{C.ANTFLY_BACKEND}:{MODEL}",
                "--host-budget-mb",
                str(HOST_BUDGET_MB),
            ],
        )
        endpoint = "/ai/v1/generate"
        info = {"arm": arm, "binary": C.public_path(args.antfly)}
        info["precision"] = C.native_precision(args.antfly, MODEL, models_dir=GEN_MODELS_DIR)
    else:
        server = LlamaServer(args.llama_server, files, out, tag)
        endpoint = "/v1/chat/completions"
        info = {"arm": arm, "provider": "llama.cpp", "binary": Path(args.llama_server).name, "version": llama_version(args.llama_server)}
        info["precision"] = {"implementation": "llama.cpp", "binary": C.file_identity(Path(shutil.which(args.llama_server) or args.llama_server)),
                             "policy": "GGUF mixed tensor storage; no single floating-point parameter dtype",
                             "requested_kv": {"key": "f16", "value": "f16"},
                             "compute_dtype": "not directly observed; do not infer from checkpoint type"}
    info["host_before"] = C.host_snapshot()
    info.update(server.start())
    calls = C.JsonlWriter(out / "calls.jsonl")
    outputs = C.JsonlWriter(out / "outputs.jsonl") if lifecycle == 0 else None
    try:
        if arm == "llama" and lifecycle == 0:
            evidence = []
            for cell in cells:
                call = cell["calls"][0]
                reference = llama_prompt_evidence(server.port, request_body(call, cell, arm))
                evidence.append({"cell": cell["id"], "k": 0, "reference": reference})
            C.write_json(out / "llama_prompt_evidence.json", evidence)
        first = stream(server.port, endpoint, request_body(cells[0]["calls"][0], cells[0], arm))
        info["first_call_ms"] = round(first["ms"], 2)
        info["first_call_status"] = first["status"]
        if first["status"] != 200:
            info["first_call_error"] = first.get("error")
        info["warmups"] = {}
        info["cell_wall_s"] = {}
        for cell in cells:
            warmup = {"state": "complete", "calls": []}
            info["warmups"][cell["id"]] = warmup
            if cell is cells[0]:
                warmup["calls"].append({"role": "first_call", **warmup_observation(first)})
                warmup["state"] = warmup["calls"][-1]["state"]
            measuring = False
            try:
                if warmup["state"] != "failed":
                    for index in range(cell["warmup"]):
                        response = stream(server.port, endpoint, request_body(cell["calls"][index % len(cell["calls"])], cell, arm))
                        observation = warmup_observation(response)
                        warmup["calls"].append(observation)
                        if observation["state"] != "complete":
                            warmup["state"] = observation["state"]
                        if observation["state"] == "failed":
                            break
                if warmup["state"] == "failed":
                    continue
                # Refusal is retained as such; measured requests are independent attempts.
                measuring = True
                cell_started = time.perf_counter()
                for index, call in enumerate(cell["calls"]):
                    result = stream(server.port, endpoint, request_body(call, cell, arm))
                    usage = result.get("usage") or {}
                    output_tokens = usage.get("completion_tokens")
                    ok = result["status"] == 200 and output_tokens == cell["max_tokens"]
                    error = result.get("error") or (None if ok else f"completion_tokens={output_tokens} != {cell['max_tokens']}")
                    decode_seconds = result.get("decode_s")
                    calls.write(
                        {
                            "arm": arm,
                            "model": MODEL_KEY,
                            "lc": lifecycle,
                            "cell": cell["id"],
                            "k": index,
                            "n_inputs": 1,
                            "words": len(call["text"].split()),
                            "ms": round(result["ms"], 3),
                            "ok": ok,
                            "status": result["status"],
                            "err": error,
                            "response_error": result.get("response_error"),
                            "warmup_state": warmup["state"],
                            "ttft_ms": None if result.get("ttft_ms") is None else round(result["ttft_ms"], 3),
                            "decode_tok_s": round((output_tokens - 1) / decode_seconds, 2) if ok and decode_seconds else None,
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": output_tokens,
                            "engine_timings": result.get("engine_timings"),
                        }
                    )
                    if outputs is not None and result["status"] == 200:
                        outputs.write(
                            {
                                "arm": arm,
                                "lc": lifecycle,
                                "cell": cell["id"],
                                "k": index,
                                "text": result["text"],
                                "ref": call.get("ref"),
                                "reasoning": result["reasoning"][:2000],
                                "prompt_tokens": usage.get("prompt_tokens"),
                            }
                        )
                info["cell_wall_s"][cell["id"]] = round(time.perf_counter() - cell_started, 4)
            except Exception as exc:
                if not measuring:
                    warmup.update(state="failed", error=C.public_error(exc))
                info.setdefault("cell_errors", {})[cell["id"]] = C.public_error(exc)
                info["error"] = "; ".join(info["cell_errors"].values())
    finally:
        calls.close()
        if outputs is not None:
            outputs.close()
        for field, observe in (
            ("host_after", C.host_snapshot),
            ("footprint", server.stop),
            ("weight_loading", lambda: loading_evidence(server, files, arm)),
        ):
            try:
                info[field] = observe()
            except Exception as exc:
                info.setdefault("teardown_errors", {})[field] = C.public_error(exc)
        if info.get("teardown_errors"):
            info["error"] = "; ".join(filter(None, [info.get("error"), *info["teardown_errors"].values()]))
        if arm == "llama" and "weight_loading" in info:
            info["precision"]["observed_storage"] = info["weight_loading"]["tensor_inventory"]
            info["precision"]["launch_flags"] = info.get("command")
            info["precision"]["observed_loading_logs"] = info["weight_loading"]["mmap"]["observed_log_lines"]
    return info


def orchestrate(args) -> bool:
    C.configure_backend(args.backend)
    files = verify_files()
    print(f"benchmark lock acquired after {C.exclusive_bench_lock()}s", flush=True)
    out = Path(args.out)
    C.prepare_output(out)
    C.write_json(out / "model_files.json", public_verified_files(files))
    cells = make_cells(args.only_cell)
    native_prompts = [
        {"cell": cell["id"], "k": 0, "native": native_prompt_evidence(args.antfly, cell["calls"][0])}
        for cell in cells
    ]
    C.write_json(out / "native_prompt_evidence.json", native_prompts)
    arms = ["antfly", "llama"]
    metadata = C.JsonlWriter(out / "lifecycles.jsonl")
    try:
        for lifecycle in range(args.lifecycles):
            for arm in arms if lifecycle % 2 == 0 else reversed(arms):
                quiet = C.wait_quiet(args.max_load, log=lambda message: print(message, flush=True))
                started = time.time()
                try:
                    info = run_arm(args, arm, files, cells, out, lifecycle)
                except Exception as exc:
                    info = {"arm": arm, "error": C.public_error(exc)}
                info.update(
                    {
                        "lc": lifecycle,
                        "model": MODEL_KEY,
                        "only_cell": args.only_cell,
                        "wall_s": round(time.time() - started, 2),
                        "quiet": quiet,
                    }
                )
                metadata.write(info)
                print(f"done {arm} in {info['wall_s']}s" + (f" ERROR {info['error']}" if "error" in info else ""), flush=True)
    finally:
        metadata.close()
    reference_path = out / "llama_prompt_evidence.json"
    if reference_path.exists():
        references = {row["cell"]: row["reference"] for row in json.loads(reference_path.read_text())}
        C.write_json(out / "prompt_comparisons.json", [
            {"cell": row["cell"], **compare_prompts(row["native"], references.get(row["cell"], {}))}
            for row in native_prompts
        ])
    expected_calls = sum(len(cell["calls"]) for cell in cells)
    return C.benchmark_complete(out, set(arms), args.lifecycles, expected_calls)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--antfly", type=Path, required=True)
    run.add_argument("--backend", choices=("auto", "metal", "cuda"), default=os.environ.get("TERMITE_BACKEND", "auto"))
    run.add_argument("--llama-server", default=os.environ.get("TERMITE_LLAMA_SERVER", shutil.which("llama-server") or "llama-server"))
    run.add_argument("--only-cell", choices=CELLS)
    run.add_argument("--lifecycles", type=int, default=3)
    run.add_argument("--max-load", type=float)
    run.add_argument("--out", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--backend", choices=("auto", "metal", "cuda"), default=os.environ.get("TERMITE_BACKEND", "auto"))
    return result


def main() -> int:
    args = parser().parse_args()
    C.configure_backend(args.backend)
    if args.command == "verify":
        print(json.dumps(public_verified_files(verify_files()), separators=(",", ":")))
        return 0
    return 0 if orchestrate(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
