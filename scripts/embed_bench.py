#!/usr/bin/env python3
"""Benchmark EmbeddingGemma 2 through Antfly and sentence-transformers."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
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


def input_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def vector_spool(out: Path, arm: str, lifecycle: int) -> Path:
    return out / f".vectors-{arm}-lc{lifecycle}.jsonl.gz"


def capture_vectors(stream, arm: str, lifecycle: int, cell: str, call: int,
                    texts: list[str], values, error: str | None = None) -> None:
    """Capture every expected timed input, including unavailable outputs, losslessly."""
    count = 0 if values is None else len(values)
    for index, text in enumerate(texts):
        vector = None if values is None or index >= count else [float(value) for value in values[index]]
        row = {"arm": arm, "lc": lifecycle, "cell": cell, "k": call, "j": index,
               "input_sha256": input_sha256(text), "v": vector, "output_count": count, "err": error}
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def valid_vector(value) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(x, (float, int)) and math.isfinite(x) for x in value)
            and any(x != 0 for x in value))


def paired_agreement(cells: list[dict], lifecycle: int, records: dict[str, dict]) -> list[dict]:
    """Summarize complete paired denominators; diagnostics never imply task accuracy."""
    rows = []
    for cell in cells:
        row = {"schema_version": 1, "model": MODEL_KEY, "lc": lifecycle, "cell": cell["id"],
               "expected_vectors": sum(len(texts) for texts in cell["calls"]),
               "antfly_vectors": 0, "py_vectors": 0, "paired_vectors": 0, "exact_vectors": 0,
               "missing_antfly": 0, "missing_py": 0, "invalid_antfly": 0, "invalid_py": 0,
               "dimension_mismatches": 0, "input_mismatches": 0, "output_count_mismatches": 0,
               "cosine_below_0_99": 0, "cosine_mean": None, "cosine_min": None,
               "max_abs_difference": None, "diagnostics": []}
        cosine_sum = 0.0
        for call, texts in enumerate(cell["calls"]):
            for index, text in enumerate(texts):
                key = (cell["id"], call, index)
                pair = [records[arm].get(key) for arm in ("antfly", "py")]
                reasons = []
                vectors = []
                for arm, record in zip(("antfly", "py"), pair):
                    vector = None if record is None else record.get("v")
                    if vector is None:
                        row[f"missing_{arm}"] += 1
                        reasons.append(f"missing_{arm}")
                    elif not valid_vector(vector):
                        row[f"invalid_{arm}"] += 1
                        reasons.append(f"invalid_{arm}")
                    else:
                        row[f"{arm}_vectors"] += 1
                    vectors.append(vector)
                if any(record and record["input_sha256"] != input_sha256(text) for record in pair):
                    row["input_mismatches"] += 1
                    reasons.append("input_mismatch")
                if any(record and record["output_count"] != len(texts) for record in pair):
                    row["output_count_mismatches"] += 1
                    reasons.append("output_count_mismatch")
                if not reasons:
                    left, right = vectors
                    if len(left) != len(right):
                        row["dimension_mismatches"] += 1
                        reasons.append("dimension_mismatch")
                    else:
                        cosine = sum(x * y for x, y in zip(left, right)) / (
                            math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right)))
                        cosine = max(-1.0, min(1.0, cosine))
                        difference = max(abs(x - y) for x, y in zip(left, right))
                        row["paired_vectors"] += 1
                        row["exact_vectors"] += int(left == right)
                        row["cosine_below_0_99"] += int(cosine < 0.99)
                        cosine_sum += cosine
                        row["cosine_min"] = cosine if row["cosine_min"] is None else min(row["cosine_min"], cosine)
                        row["max_abs_difference"] = max(row["max_abs_difference"] or 0, difference)
                        if cosine < 0.99:
                            reasons.append("cosine_below_0_99")
                if reasons and len(row["diagnostics"]) < 8:
                    row["diagnostics"].append({"k": call, "j": index,
                                               "input_sha256": input_sha256(text), "reasons": reasons})
        if row["paired_vectors"]:
            row["cosine_mean"] = cosine_sum / row["paired_vectors"]
        row["complete"] = row["paired_vectors"] == row["expected_vectors"]
        rows.append(row)
    return rows


def write_agreement(out: Path, cells: list[dict], lifecycles: int) -> None:
    """Read at most one cell of full vectors at a time; retain compact aggregates only."""
    with (out / "agreement.jsonl").open("w", encoding="utf-8") as destination:
        for lifecycle in range(lifecycles):
            streams = {}
            try:
                streams = {arm: gzip.open(vector_spool(out, arm, lifecycle), "rt", encoding="utf-8")
                           for arm in ("antfly", "py") if vector_spool(out, arm, lifecycle).exists()}
                pending = {arm: next(stream, None) for arm, stream in streams.items()}
                for cell in cells:
                    records = {"antfly": {}, "py": {}}
                    for arm, stream in streams.items():
                        while pending[arm] is not None:
                            record = json.loads(pending[arm])
                            if record["cell"] != cell["id"]:
                                break
                            key = (record["cell"], record["k"], record["j"])
                            if key in records[arm]:
                                raise ValueError(f"duplicate embedding output {arm} lc{lifecycle} {key}")
                            records[arm][key] = record
                            pending[arm] = next(stream, None)
                    for row in paired_agreement([cell], lifecycle, records):
                        destination.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            finally:
                for stream in streams.values():
                    stream.close()
    for lifecycle in range(lifecycles):
        for arm in ("antfly", "py"):
            vector_spool(out, arm, lifecycle).unlink(missing_ok=True)


def run_antfly(args, cells: list[dict], out: Path, lifecycle: int) -> dict:
    server = C.AntflyServer(args.antfly, out, f"lc{lifecycle}-antfly", extra_args=["--scratch-budget-mb", "8192"])
    info = {"arm": "antfly", "binary": C.public_path(args.antfly), "host_before": C.host_snapshot()}
    info.update(server.start())
    calls = C.JsonlWriter(out / "calls.jsonl")
    vectors = gzip.open(vector_spool(out, "antfly", lifecycle), "wt", encoding="utf-8")
    try:
        client = C.Client(server.port)
        started = time.perf_counter()
        first = {}
        try:
            status, first, _ = client.post("/ai/v1/embed", {"model": MODEL, "input": cells[0]["calls"][0]})
            info["first_call_status"] = status
            if status != 200:
                info["first_call_error"] = C.api_error(first)
        except Exception as error:
            info["first_call_error"] = C.public_error(error)
        info["first_call_ms"] = round((time.perf_counter() - started) * 1000, 2)
        info["backend"] = first.get("backend") if isinstance(first, dict) else None
        info["precision"] = C.native_precision(args.antfly, MODEL)
        for cell in cells:
            warmup_state = C.http_warmup(client, "/ai/v1/embed",
                ({"model": MODEL, "input": cell["calls"][index % len(cell["calls"])]}
                 for index in range(cell["warmup"])), info, cell["id"])
            if warmup_state == "failed":
                continue
            cell_started = time.perf_counter()
            rows = []
            for index, texts in enumerate(cell["calls"]):
                started = time.perf_counter()
                error = None
                try:
                    status, body, _ = client.post("/ai/v1/embed", {"model": MODEL, "input": texts})
                except Exception as exc:
                    status, body, error = -1, {}, C.public_error(exc)
                rows.append((index, texts, (time.perf_counter() - started) * 1000, status, body, error))
                if error:
                    break
            cell_wall = time.perf_counter() - cell_started
            for index, texts, elapsed_ms, status, body, error in rows:
                data = body.get("data") or []
                ok = status == 200 and len(data) == len(texts)
                calls.write(
                    {
                        "arm": "antfly",
                        "model": MODEL_KEY,
                        "lc": lifecycle,
                        "cell": cell["id"],
                        "k": index,
                        "warmup_state": warmup_state,
                        "n_inputs": len(texts),
                        "words": word_count(texts),
                        "ms": round(elapsed_ms, 3),
                        "ok": ok,
                        "status": status,
                        "tokens": (body.get("usage") or {}).get("prompt_tokens"),
                        "backend": body.get("backend"),
                        "response_error": {} if ok else C.api_error(body),
                        "err": error or (None if ok else C.public_error(body.get("error") or f"HTTP {status}")),
                        "cell_wall_s": round(cell_wall, 4),
                    }
                )
                capture_vectors(vectors, "antfly", lifecycle, cell["id"], index, texts,
                                [item["embedding"] for item in data], None if ok else str(status))
    finally:
        calls.close()
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
    model = SentenceTransformer(C.reference_paths()[MODEL], device=C.ACCEL)
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
    info["precision"] = C.reference_precision(model, default_policy="library_default")
    info["precision"]["default_policy_source"] = "https://sbert.net/docs/sentence_transformer/usage/efficiency.html#pytorch"
    sync = C.torch_sync(torch)
    tokenizer = model.tokenizer

    def encode(texts: list[str]):
        return model.encode(texts, prompt_name="document", batch_size=len(texts), convert_to_numpy=True)

    started = time.perf_counter()
    encode(cells[0]["calls"][0])
    sync()
    info["first_call_ms"] = round((time.perf_counter() - started) * 1000, 2)
    calls = C.JsonlWriter(out / "calls.jsonl")
    vectors = gzip.open(vector_spool(out, "py", lifecycle), "wt", encoding="utf-8")
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
                capture_vectors(vectors, "py", lifecycle, cell["id"], index, texts, values, error)
    finally:
        calls.close()
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
    write_agreement(out, cells, args.lifecycles)
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
