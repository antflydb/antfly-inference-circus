#!/usr/bin/env python3
"""Benchmark Gemma 4 E4B through Antfly and llama.cpp using identical GGUF files."""
from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import shutil
import subprocess
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
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        return {"status": -1, "err": C.public_error(exc), "ms": (time.perf_counter() - started) * 1000}


def _stream(port: int, path: str, payload: dict, timeout: float) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    data = json.dumps(payload).encode()
    started = time.perf_counter()
    connection.request("POST", path, body=data, headers={"content-type": "application/json", "accept": "text/event-stream"})
    response = connection.getresponse()
    if response.status != 200:
        error = response.read()[:400].decode(errors="replace")
        connection.close()
        return {"status": response.status, "err": error, "ms": (time.perf_counter() - started) * 1000}
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
            self.sampler = C.Sampler(self.process.pid, self.out, self.tag)
            ready = C.wait_http(self.process, self.port, "/health", 300, self.log.name, started)
        except Exception:
            self.stop()
            raise
        return {"ready_s": ready, "command": C.public_command(self.argv)}

    def stop(self) -> dict:
        C.stop_proc(self.process)
        try:
            return self.sampler.stop() if self.sampler else {}
        finally:
            self.sampler = None
            if not self.log.closed:
                self.log.close()


def llama_version(binary: str) -> str:
    process = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30, check=False)
    return " ".join((process.stdout + process.stderr).split())[:300]


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
    else:
        server = LlamaServer(args.llama_server, files, out, tag)
        endpoint = "/v1/chat/completions"
        info = {"arm": arm, "provider": "llama.cpp", "binary": Path(args.llama_server).name, "version": llama_version(args.llama_server)}
    info["host_before"] = C.host_snapshot()
    info.update(server.start())
    calls = C.JsonlWriter(out / "calls.jsonl")
    outputs = C.JsonlWriter(out / "outputs.jsonl") if lifecycle == 0 else None
    try:
        first = stream(server.port, endpoint, request_body(cells[0]["calls"][0], cells[0], arm))
        info["first_call_ms"] = round(first["ms"], 2)
        info["first_call_status"] = first["status"]
        if first["status"] != 200:
            info["first_call_error"] = first.get("err")
        for cell in cells:
            for index in range(cell["warmup"]):
                stream(server.port, endpoint, request_body(cell["calls"][index % len(cell["calls"])], cell, arm))
            cell_started = time.perf_counter()
            rows = [stream(server.port, endpoint, request_body(call, cell, arm)) for call in cell["calls"]]
            cell_wall = time.perf_counter() - cell_started
            for index, result in enumerate(rows):
                usage = result.get("usage") or {}
                output_tokens = usage.get("completion_tokens")
                ok = result["status"] == 200 and output_tokens == cell["max_tokens"]
                error = result.get("err") or (None if ok else f"completion_tokens={output_tokens} != {cell['max_tokens']}")
                decode_seconds = result.get("decode_s")
                calls.write(
                    {
                        "arm": arm,
                        "model": MODEL_KEY,
                        "lc": lifecycle,
                        "cell": cell["id"],
                        "k": index,
                        "n_inputs": 1,
                        "words": len(cell["calls"][index]["text"].split()),
                        "ms": round(result["ms"], 3),
                        "ok": ok,
                        "status": result["status"],
                        "err": error,
                        "cell_wall_s": round(cell_wall, 4),
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
                            "cell": cell["id"],
                            "k": index,
                            "text": result["text"],
                            "ref": cell["calls"][index].get("ref"),
                            "reasoning": result["reasoning"][:2000],
                            "prompt_tokens": usage.get("prompt_tokens"),
                        }
                    )
    finally:
        calls.close()
        if outputs is not None:
            outputs.close()
        info["host_after"] = C.host_snapshot()
        info["footprint"] = server.stop()
    return info


def orchestrate(args) -> bool:
    C.configure_backend(args.backend)
    files = verify_files()
    print(f"benchmark lock acquired after {C.exclusive_bench_lock()}s", flush=True)
    out = Path(args.out)
    C.prepare_output(out)
    C.write_json(out / "model_files.json", public_verified_files(files))
    cells = make_cells(args.only_cell)
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
