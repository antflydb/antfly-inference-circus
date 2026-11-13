#!/usr/bin/env python3
"""Benchmark EmbeddingGemma 2 through Antfly and sentence-transformers."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODEL = "google/embeddinggemma-2"
MODEL_KEY = "embeddinggemma2"
PROVIDERS = ["antfly", "sentence-transformers"]
CELLS = ["sentence_x1", "sentence_b32", "paragraph_x1", "paragraph_b32", "page_b8", "long2k_x1", "long8k_x1"]


def make_cells(only: str | None = None) -> list[dict]:
    corpus = C.load_corpus()
    words = " ".join(chapter["text"] for chapter in corpus["chapters"]).split()

    def long_texts(size: int, count: int) -> list[str]:
        return [" ".join(words[index * size : (index + 1) * size]) for index in range(count)]

    def chunks(values: list[str], size: int) -> list[list[str]]:
        return [values[index : index + size] for index in range(0, len(values), size)]

    cells = [
        {"id": "sentence_x1", "calls": [[value] for value in corpus["sentences"][:64]], "warmup": 2},
        {"id": "sentence_b32", "calls": chunks(corpus["sentences"][:256], 32), "warmup": 1},
        {"id": "paragraph_x1", "calls": [[value] for value in corpus["paragraphs"][:64]], "warmup": 2},
        {"id": "paragraph_b32", "calls": chunks(corpus["paragraphs"][:256], 32), "warmup": 1},
        {"id": "page_b8", "calls": chunks(corpus["pages"][:64], 8), "warmup": 1},
        {"id": "long2k_x1", "calls": [[value] for value in long_texts(1500, 8)], "warmup": 1},
        {"id": "long8k_x1", "calls": [[value] for value in long_texts(5500, 6)], "warmup": 1},
    ]
    return [cell for cell in cells if only in (None, cell["id"])]


def word_count(texts: list[str]) -> int:
    return sum(len(text.split()) for text in texts)


def run_antfly(args, cells: list[dict], out: Path, lifecycle: int) -> dict:
    server = C.AntflyServer(args.antfly, out, f"lc{lifecycle}-antfly", extra_args=["--scratch-budget-mb", "8192"])
    info = {"arm": "antfly", "binary": C.public_path(args.antfly), "host_before": C.host_snapshot()}
    info.update(server.start())
    calls = C.JsonlWriter(out / "calls.jsonl")
    vectors = C.JsonlWriter(out / "vectors.jsonl") if lifecycle == 0 else None
    try:
        client = C.Client(server.port)
        started = time.perf_counter()
        _, first, _ = client.post("/ai/v1/embed", {"model": MODEL, "input": cells[0]["calls"][0]})
        info["first_call_ms"] = round((time.perf_counter() - started) * 1000, 2)
        info["backend"] = first.get("backend")
        for cell in cells:
            for index in range(cell["warmup"]):
                client.post("/ai/v1/embed", {"model": MODEL, "input": cell["calls"][index % len(cell["calls"])]})
            cell_started = time.perf_counter()
            rows = []
            for index, texts in enumerate(cell["calls"]):
                started = time.perf_counter()
                status, body, _ = client.post("/ai/v1/embed", {"model": MODEL, "input": texts})
                rows.append((index, texts, (time.perf_counter() - started) * 1000, status, body))
            cell_wall = time.perf_counter() - cell_started
            for index, texts, elapsed_ms, status, body in rows:
                data = body.get("data") or []
                ok = status == 200 and len(data) == len(texts)
                calls.write(
                    {
                        "arm": "antfly",
                        "model": MODEL_KEY,
                        "lc": lifecycle,
                        "cell": cell["id"],
                        "k": index,
                        "n_inputs": len(texts),
                        "words": word_count(texts),
                        "ms": round(elapsed_ms, 3),
                        "ok": ok,
                        "status": status,
                        "tokens": (body.get("usage") or {}).get("prompt_tokens"),
                        "backend": body.get("backend"),
                        "err": None if ok else json.dumps(body, ensure_ascii=False)[:300],
                        "cell_wall_s": round(cell_wall, 4),
                    }
                )
                if vectors is not None and ok and index < 2:
                    for vector_index, item in enumerate(data[:4]):
                        vectors.write(
                            {
                                "arm": "antfly",
                                "cell": cell["id"],
                                "k": index,
                                "j": vector_index,
                                "v": [round(value, 6) for value in item["embedding"]],
                            }
                        )
    finally:
        calls.close()
        if vectors is not None:
            vectors.close()
        info["host_after"] = C.host_snapshot()
        info["footprint"] = server.stop()
    return info


def run_python(args, cells: list[dict], out: Path, lifecycle: int) -> dict:
    started = time.perf_counter()
    import torch
    from sentence_transformers import SentenceTransformer

    import_s = time.perf_counter() - started
    started = time.perf_counter()
    model = SentenceTransformer(C.reference_paths()[MODEL], device=C.ACCEL, model_kwargs={"dtype": "float32"})
    load_s = time.perf_counter() - started
    parameter = next(model.parameters())
    info = {
        "arm": "py",
        "provider": "sentence-transformers",
        "import_s": round(import_s, 3),
        "load_s": round(load_s, 3),
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "sentence_transformers": __import__("sentence_transformers").__version__,
        "host_before": C.host_snapshot(),
    }
    sync = C.torch_sync(torch)
    tokenizer = model.tokenizer

    def encode(texts: list[str]):
        return model.encode(texts, prompt_name="document", batch_size=len(texts), convert_to_numpy=True)

    started = time.perf_counter()
    encode(cells[0]["calls"][0])
    sync()
    info["first_call_ms"] = round((time.perf_counter() - started) * 1000, 2)
    calls = C.JsonlWriter(out / "calls.jsonl")
    vectors = C.JsonlWriter(out / "vectors.jsonl") if lifecycle == 0 else None
    try:
        for cell in cells:
            for index in range(cell["warmup"]):
                encode(cell["calls"][index % len(cell["calls"])])
            sync()
            cell_started = time.perf_counter()
            rows = []
            for index, texts in enumerate(cell["calls"]):
                started = time.perf_counter()
                error, values = None, None
                try:
                    values = encode(texts)
                    sync()
                except Exception as exc:
                    error = C.public_error(exc)
                rows.append((index, texts, (time.perf_counter() - started) * 1000, error, values))
            cell_wall = time.perf_counter() - cell_started
            for index, texts, elapsed_ms, error, values in rows:
                ok = error is None and values is not None and len(values) == len(texts)
                tokens = sum(len(tokenizer("title: none | text: " + text)["input_ids"]) for text in texts)
                calls.write(
                    {
                        "arm": "py",
                        "model": MODEL_KEY,
                        "lc": lifecycle,
                        "cell": cell["id"],
                        "k": index,
                        "n_inputs": len(texts),
                        "words": word_count(texts),
                        "ms": round(elapsed_ms, 3),
                        "ok": ok,
                        "status": 200 if ok else -1,
                        "tokens": tokens,
                        "err": error,
                        "cell_wall_s": round(cell_wall, 4),
                    }
                )
                if vectors is not None and ok and index < 2:
                    for vector_index in range(min(4, len(values))):
                        vectors.write(
                            {
                                "arm": "py",
                                "cell": cell["id"],
                                "k": index,
                                "j": vector_index,
                                "v": [round(float(value), 6) for value in values[vector_index]],
                            }
                        )
    finally:
        calls.close()
        if vectors is not None:
            vectors.close()
    info["host_after"] = C.host_snapshot()
    info.update(C.torch_device_bytes(torch))
    return info


def orchestrate(args) -> bool:
    C.configure_backend(args.backend)
    print(f"benchmark lock acquired after {C.exclusive_bench_lock()}s", flush=True)
    out = Path(args.out)
    C.prepare_output(out)
    cells = make_cells(args.only_cell)
    arms = ["antfly", "py"]
    metadata = C.JsonlWriter(out / "lifecycles.jsonl")
    try:
        for lifecycle in range(args.lifecycles):
            for arm in arms if lifecycle % 2 == 0 else reversed(arms):
                quiet = C.wait_quiet(args.max_load, log=lambda message: print(message, flush=True))
                started = time.time()
                if arm == "antfly":
                    try:
                        info = run_antfly(args, cells, out, lifecycle)
                    except Exception as exc:
                        info = {"arm": arm, "error": C.public_error(exc)}
                else:
                    child_args = ["--lc", str(lifecycle), "--out", str(out), "--backend", C.ANTFLY_BACKEND]
                    if args.only_cell:
                        child_args.extend(["--only-cell", args.only_cell])
                    info = C.run_python_arm(__file__, child_args, out, f"lc{lifecycle}-py")
                    info.setdefault("arm", arm)
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
    run.add_argument("--lifecycles", type=int, default=3)
    run.add_argument("--only-cell", choices=CELLS)
    run.add_argument("--max-load", type=float)
    run.add_argument("--out", required=True)
    child = subparsers.add_parser("pyarm")
    child.add_argument("--lc", type=int, required=True)
    child.add_argument("--out", required=True)
    child.add_argument("--only-cell", choices=CELLS)
    child.add_argument("--backend", choices=("metal", "cuda"), required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    C.configure_backend(args.backend)
    if args.command == "run":
        return 0 if orchestrate(args) else 1
    print(json.dumps(run_python(args, make_cells(args.only_cell), Path(args.out), args.lc), separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
