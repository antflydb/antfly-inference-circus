#!/usr/bin/env python3
"""Benchmark GLiNER2.5 extraction through Antfly and the upstream gliner2 package."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODELS = {"base": "fastino/gliner2.5-base-v1", "decide": "fastino/GLiNER2.5-Decide"}
MODEL_KEYS = {"base": "gliner25-base", "decide": "gliner25-decide"}
PROVIDERS = ["antfly", "gliner2"]
ENTITY_LABELS = ["person", "location", "organization", "species", "date"]
MIXED_SCHEMA = {
    "entities": ["person", "location", "organization", "species"],
    "classifications": [
        {"name": "topic", "labels": ["biology", "geology", "history", "business", "politics", "culture"]}
    ],
    "relations": [{"type": "located_in"}, {"type": "works_for"}],
}
DECIDE_SCHEMA = {
    "classifications": [
        {"name": "intent", "labels": ["maintenance", "room_change", "checkout", "billing", "complaint", "information"]},
        {
            "name": "topics",
            "labels": ["hvac", "billing", "noise", "cleanliness", "wifi", "staff"],
            "multi_label": True,
            "threshold": 0.4,
        },
        {"name": "urgent", "labels": ["yes", "no"]},
    ]
}
CELLS = {
    "base": [
        "ent_sentence_x1",
        "ent_paragraph_x1",
        "ent_page_x1",
        "ent_wiki_x1",
        "mixed_paragraph_x1",
        "ent_paragraph_b8",
        "ent_paragraph_b32",
        "long_book_w384",
    ],
    "decide": ["decide_x1", "decide_b8", "decide_b32"],
}


def hotel_texts(count: int, seed: int = 11) -> list[str]:
    import random

    randomizer = random.Random(seed)
    parts = [
        "Guest in room {r} says the AC has been out since yesterday and they want to move tonight.",
        "The wifi in {r} keeps dropping during video calls; can someone take a look before 9am?",
        "I was charged twice for the minibar on my final bill for room {r}. Please refund one charge.",
        "Loud music from the room next to {r} until 2am again. This is the third night.",
        "What time is checkout? We'd like a late checkout for room {r} if possible.",
        "The bathroom in {r} wasn't cleaned today and there are no fresh towels.",
        "Front desk staff were incredibly kind when our flight was cancelled. Thank you!",
        "Water is leaking from the ceiling in {r} onto the bed. Please send someone immediately.",
    ]
    return [
        " ".join(
            randomizer.choice(parts).format(r=randomizer.randint(100, 1899))
            for _ in range(randomizer.randint(1, 3))
        )
        for _ in range(count)
    ]


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def make_cells(model: str, only: str | None = None) -> list[dict]:
    corpus = C.load_corpus()
    cells: list[dict] = []

    def add(cell_id, calls, schema, warmup=3, long=None, python_batch_size=None):
        cells.append(
            {
                "id": cell_id,
                "calls": calls,
                "schema": schema,
                "warmup": warmup,
                "long": long,
                "python_batch_size": python_batch_size,
            }
        )

    if model == "base":
        paragraphs = corpus["paragraphs"]
        add("ent_sentence_x1", [[text] for text in corpus["sentences"][:64]], {"entities": ENTITY_LABELS})
        add("ent_paragraph_x1", [[text] for text in paragraphs[:64]], {"entities": ENTITY_LABELS})
        add("ent_page_x1", [[text] for text in corpus["pages"][:32]], {"entities": ENTITY_LABELS})
        add("ent_wiki_x1", [[text] for text in corpus["wiki_en"][:64]], {"entities": ENTITY_LABELS})
        add("mixed_paragraph_x1", [[text] for text in paragraphs[64:96]], MIXED_SCHEMA)
        add("ent_paragraph_b8", chunked(paragraphs[:64], 8), {"entities": ENTITY_LABELS}, warmup=1, python_batch_size=8)
        add("ent_paragraph_b32", chunked(paragraphs[:128], 32), {"entities": ENTITY_LABELS}, warmup=1, python_batch_size=32)
        add(
            "long_book_w384",
            [[chapter["text"]] for chapter in corpus["chapters"]],
            {"entities": ENTITY_LABELS},
            warmup=1,
            long={"window_words": 384, "overlap_words": 64},
            python_batch_size=8,
        )
    elif model == "decide":
        texts = hotel_texts(256)
        add("decide_x1", [[text] for text in texts[:64]], DECIDE_SCHEMA)
        add("decide_b8", chunked(texts[:128], 8), DECIDE_SCHEMA, warmup=2, python_batch_size=8)
        add("decide_b32", chunked(texts[:128], 32), DECIDE_SCHEMA, warmup=1, python_batch_size=32)
    return [cell for cell in cells if only in (None, cell["id"])]


def byte_to_codepoint(text: str):
    table, offset = {}, 0
    for index, character in enumerate(text):
        table[offset] = index
        offset += len(character.encode())
    table[offset] = len(text)
    return lambda value: table.get(value, value)


def normalize_antfly(item: dict, text: str | None = None) -> dict:
    convert = (
        byte_to_codepoint(text)
        if text is not None and item.get("offset_unit", "utf8") != "unicode_codepoints"
        else lambda value: value
    )
    for entity in item.get("entities", []):
        entity["start"], entity["end"] = convert(entity["start"]), convert(entity["end"])
    for relation in item.get("relations", []):
        for side in ("source", "target"):
            relation[side]["start"], relation[side]["end"] = convert(relation[side]["start"]), convert(relation[side]["end"])
    entities = sorted(
        [entity["label"], entity["start"], entity["end"], round(entity.get("score", 0), 4)]
        for entity in item.get("entities", [])
    )
    classifications: dict = {}
    for classification in item.get("classifications", []):
        value = classification.get("label", classification.get("labels"))
        name = classification["name"]
        if name in classifications:
            previous = classifications[name]
            classifications[name] = (previous if isinstance(previous, list) else [previous]) + (
                value if isinstance(value, list) else [value]
            )
        else:
            classifications[name] = value
    relations = sorted(
        [
            relation["type"],
            relation["source"]["start"],
            relation["source"]["end"],
            relation["target"]["start"],
            relation["target"]["end"],
        ]
        for relation in item.get("relations", [])
    )
    return {"e": entities, "c": classifications, "r": relations}


def normalize_python(result: dict, schema: dict) -> dict:
    entities = []
    for label, values in (result.get("entities") or {}).items():
        for value in values:
            if isinstance(value, dict):
                entities.append([label, value.get("start"), value.get("end"), round(value.get("confidence", 0), 4)])
    classifications = {}
    for classification in schema.get("classifications", []):
        value = result.get(classification["name"])
        if isinstance(value, dict):
            classifications[classification["name"]] = value.get("label")
        elif isinstance(value, list):
            classifications[classification["name"]] = [
                item.get("label") if isinstance(item, dict) else item for item in value
            ]
        else:
            classifications[classification["name"]] = value
    relations = []
    for relation_type, values in (result.get("relation_extraction") or {}).items():
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("head"), dict):
                relations.append(
                    [
                        relation_type,
                        value["head"].get("start"),
                        value["head"].get("end"),
                        value["tail"].get("start"),
                        value["tail"].get("end"),
                    ]
                )
    return {"e": sorted(entities), "c": classifications, "r": sorted(relations)}


def upstream_schema(schema: dict) -> dict:
    value: dict = {}
    if "entities" in schema:
        value["entities"] = schema["entities"]
    if "classifications" in schema:
        value["classifications"] = []
        for classification in schema["classifications"]:
            item = {"task": classification["name"], "labels": classification["labels"]}
            if classification.get("multi_label"):
                item["multi_label"] = True
            if "threshold" in classification:
                item["cls_threshold"] = classification["threshold"]
            value["classifications"].append(item)
    if "relations" in schema:
        value["relations"] = [relation["type"] for relation in schema["relations"]]
    return value


def word_count(texts: list[str]) -> int:
    return sum(len(text.split()) for text in texts)


def request_body(model_id: str, texts: list[str], cell: dict) -> dict:
    options = {"include_spans": True, "include_confidence": True}
    if cell["long"] is not None:
        options["long_document"] = {"mode": "window", "max_windows": 128, **cell["long"]}
    return {
        "schema_version": 2,
        "model": model_id,
        "schema": cell["schema"],
        "inputs": [{"content": text} for text in texts],
        "options": options,
    }


def run_antfly(args, cells: list[dict], out: Path, lifecycle: int) -> dict:
    model_id = MODELS[args.model]
    server = C.AntflyServer(args.antfly, out, f"lc{lifecycle}-antfly")
    info = {"arm": "antfly", "binary": C.public_path(args.antfly), "host_before": C.host_snapshot()}
    info.update(server.start())
    calls = C.JsonlWriter(out / "calls.jsonl")
    outputs = C.JsonlWriter(out / "outputs.jsonl") if lifecycle == 0 else None
    try:
        client = C.Client(server.port)
        started = time.perf_counter()
        status, body, _ = client.post("/ai/v1/extract", request_body(model_id, cells[0]["calls"][0], cells[0]))
        info["first_call_ms"] = round((time.perf_counter() - started) * 1000, 2)
        info["first_call_status"] = status
        if status != 200:
            info["first_call_error"] = json.dumps(body, ensure_ascii=False)[:400]
        for cell in cells:
            for index in range(cell["warmup"]):
                C.Client(server.port).post(
                    "/ai/v1/extract",
                    request_body(model_id, cell["calls"][index % len(cell["calls"])], cell),
                )
            client = C.Client(server.port)
            cell_started = time.perf_counter()
            rows = []
            for index, texts in enumerate(cell["calls"]):
                started = time.perf_counter()
                try:
                    status, body, bytes_read = client.post("/ai/v1/extract", request_body(model_id, texts, cell))
                    error = None if status == 200 else json.dumps(body, ensure_ascii=False)[:400]
                except Exception as exc:
                    status, body, bytes_read, error = -1, {}, 0, C.public_error(exc)
                rows.append((index, texts, (time.perf_counter() - started) * 1000, status, body, bytes_read, error))
            cell_wall = time.perf_counter() - cell_started
            for index, texts, elapsed_ms, status, body, bytes_read, error in rows:
                data = body.get("data") if isinstance(body, dict) else None
                ok = status == 200 and isinstance(data, list) and len(data) == len(texts)
                calls.write(
                    {
                        "arm": "antfly",
                        "model": args.model,
                        "lc": lifecycle,
                        "cell": cell["id"],
                        "k": index,
                        "n_inputs": len(texts),
                        "words": word_count(texts),
                        "ms": round(elapsed_ms, 3),
                        "ok": ok,
                        "status": status,
                        "bytes": bytes_read,
                        "err": error,
                        "cell_wall_s": round(cell_wall, 4),
                    }
                )
                if outputs is not None and ok:
                    for output_index, item in enumerate(data):
                        outputs.write(
                            {
                                "arm": "antfly",
                                "cell": cell["id"],
                                "k": index,
                                "j": output_index,
                                "o": normalize_antfly(item, texts[output_index]),
                                "windows": (item.get("long_document") or {}).get("window_count"),
                            }
                        )
    finally:
        calls.close()
        if outputs is not None:
            outputs.close()
        info["host_after"] = C.host_snapshot()
        info["footprint"] = server.stop()
    return info


def run_python(args, cells: list[dict], out: Path, lifecycle: int) -> dict:
    import_started = time.perf_counter()
    import torch
    from gliner2 import AutoExtractor, Schema

    import_s = time.perf_counter() - import_started
    started = time.perf_counter()
    model = AutoExtractor.from_pretrained(C.reference_paths()[MODELS[args.model]])
    model.to(C.ACCEL)
    model.eval()
    load_s = time.perf_counter() - started
    parameter = next(model.parameters())
    info = {
        "arm": "py",
        "provider": "gliner2",
        "import_s": round(import_s, 3),
        "load_s": round(load_s, 3),
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "gliner2": __import__("gliner2").__version__,
        "type": type(model).__name__,
        "host_before": C.host_snapshot(),
    }
    sync = C.torch_sync(torch)
    calls = C.JsonlWriter(out / "calls.jsonl")
    outputs = C.JsonlWriter(out / "outputs.jsonl") if lifecycle == 0 else None

    def call(texts, cell, schema):
        with torch.inference_mode():
            if cell["long"] is not None:
                return model.batch_extract_long(
                    texts,
                    schema,
                    batch_size=cell["python_batch_size"] or 8,
                    include_spans=True,
                    include_confidence=True,
                    chunk_size=cell["long"]["window_words"],
                    chunk_overlap=cell["long"]["overlap_words"],
                )
            if len(texts) == 1:
                return [model.extract(texts[0], schema, include_spans=True, include_confidence=True)]
            return model.batch_extract(
                texts,
                schema,
                batch_size=cell["python_batch_size"] or len(texts),
                include_spans=True,
                include_confidence=True,
            )

    first = cells[0]
    started = time.perf_counter()
    call(first["calls"][0], first, Schema.from_dict(upstream_schema(first["schema"])))
    sync()
    info["first_call_ms"] = round((time.perf_counter() - started) * 1000, 2)
    try:
        for cell in cells:
            schema = Schema.from_dict(upstream_schema(cell["schema"]))
            for index in range(cell["warmup"]):
                call(cell["calls"][index % len(cell["calls"])], cell, schema)
            sync()
            cell_started = time.perf_counter()
            rows = []
            for index, texts in enumerate(cell["calls"]):
                started = time.perf_counter()
                error, result = None, None
                try:
                    result = call(texts, cell, schema)
                    sync()
                except Exception as exc:
                    error = C.public_error(exc)
                rows.append((index, texts, (time.perf_counter() - started) * 1000, error, result))
            cell_wall = time.perf_counter() - cell_started
            for index, texts, elapsed_ms, error, result in rows:
                ok = error is None and isinstance(result, list) and len(result) == len(texts)
                calls.write(
                    {
                        "arm": "py",
                        "model": args.model,
                        "lc": lifecycle,
                        "cell": cell["id"],
                        "k": index,
                        "n_inputs": len(texts),
                        "words": word_count(texts),
                        "ms": round(elapsed_ms, 3),
                        "ok": ok,
                        "status": 200 if ok else -1,
                        "err": error,
                        "cell_wall_s": round(cell_wall, 4),
                    }
                )
                if outputs is not None and ok:
                    for output_index, item in enumerate(result):
                        outputs.write(
                            {
                                "arm": "py",
                                "cell": cell["id"],
                                "k": index,
                                "j": output_index,
                                "o": normalize_python(item, cell["schema"]),
                            }
                        )
    finally:
        calls.close()
        if outputs is not None:
            outputs.close()
    info["host_after"] = C.host_snapshot()
    info.update(C.torch_device_bytes(torch))
    return info


def orchestrate(args) -> bool:
    C.configure_backend(args.backend)
    print(f"benchmark lock acquired after {C.exclusive_bench_lock()}s", flush=True)
    out = Path(args.out)
    C.prepare_output(out)
    cells = make_cells(args.model, args.only_cell)
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
                    child_args = [
                        "--model",
                        args.model,
                        "--lc",
                        str(lifecycle),
                        "--out",
                        str(out),
                        "--backend",
                        C.ANTFLY_BACKEND,
                    ]
                    if args.only_cell:
                        child_args.extend(["--only-cell", args.only_cell])
                    info = C.run_python_arm(__file__, child_args, out, f"lc{lifecycle}-py")
                    info.setdefault("arm", arm)
                info.update(
                    {
                        "lc": lifecycle,
                        "model": args.model,
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
    run.add_argument("--model", required=True, choices=sorted(MODELS))
    run.add_argument("--antfly", type=Path, required=True)
    run.add_argument("--backend", choices=("auto", "metal", "cuda"), default=os.environ.get("TERMITE_BACKEND", "auto"))
    run.add_argument("--only-cell")
    run.add_argument("--lifecycles", type=int, default=3)
    run.add_argument("--max-load", type=float)
    run.add_argument("--out", required=True)
    child = subparsers.add_parser("pyarm")
    child.add_argument("--model", required=True, choices=sorted(MODELS))
    child.add_argument("--lc", type=int, required=True)
    child.add_argument("--out", required=True)
    child.add_argument("--only-cell")
    child.add_argument("--backend", choices=("metal", "cuda"), required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    C.configure_backend(args.backend)
    if args.only_cell and args.only_cell not in CELLS[args.model]:
        parser().error(f"--only-cell must be one of {CELLS[args.model]}")
    if args.command == "run":
        return 0 if orchestrate(args) else 1
    print(
        json.dumps(run_python(args, make_cells(args.model, args.only_cell), Path(args.out), args.lc), separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
