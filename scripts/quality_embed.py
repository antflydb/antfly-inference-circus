#!/usr/bin/env python3
"""Offline, judged SciFact retrieval on a frozen 24-query/256-document subset.

This is a restricted-candidate smoke evaluation, not the full BEIR benchmark.
Only setup downloads assets. Relevance comes from the public SciFact test qrels.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import embed_bench as E

ASSETS = C.DATA / "quality" / "embed"
REVISION = "984eed826375f18d27936c4a32bf0f8491e3f414"
QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
LICENSE = "CC-BY-SA-4.0"
SOURCES = {
    "corpus.jsonl.gz": {
        "url": f"https://huggingface.co/datasets/BeIR/scifact/resolve/{REVISION}/corpus.jsonl.gz",
        "sha256": "2f4a2a264485e5b5b95ae9cd5534817447757423991a4767d6da811539c0853e"},
    "queries.jsonl.gz": {
        "url": f"https://huggingface.co/datasets/BeIR/scifact/resolve/{REVISION}/queries.jsonl.gz",
        "sha256": "c29574b9cb8fdf043df8494f689de15f4604ac4b585eb34bda9b6a389aa6c4d3"},
    "test.tsv": {
        "url": f"https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/{QRELS_REVISION}/test.tsv",
        "sha256": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c"},
}


def canonical(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def make_fixture(blobs: dict[str, bytes]) -> dict:
    corpus = {str(row["_id"]): row for row in
              (json.loads(line) for line in gzip.decompress(blobs["corpus.jsonl.gz"]).splitlines())}
    queries = {str(row["_id"]): row for row in
               (json.loads(line) for line in gzip.decompress(blobs["queries.jsonl.gz"]).splitlines())}
    judgments: dict[str, dict[str, int]] = {}
    for row in csv.DictReader(io.StringIO(blobs["test.tsv"].decode("utf-8")), delimiter="\t"):
        if int(row["score"]) > 0:
            judgments.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    query_ids = sorted(judgments)[:24]
    required = {doc for query in query_ids for doc in judgments[query]}
    document_ids = sorted(required | set(sorted(set(corpus) - required)[:256 - len(required)]))
    if len(document_ids) != 256 or len(query_ids) != 24:
        raise ValueError("unexpected public dataset cardinality")
    return {
        "dataset": "BEIR/SciFact", "license": LICENSE,
        "selection": "first 24 lexicographically sorted test query IDs; all positive qrels plus lexicographically first remaining documents to 256",
        "documents": [{"id": key, "text": (corpus[key]["title"] + "\n" + corpus[key]["text"]).strip()}
                      for key in document_ids],
        "queries": [{"id": key, "text": queries[key]["text"], "relevant": judgments[key]} for key in query_ids],
    }


def setup(root: Path = ASSETS) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        path = root / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]:
            continue
        with urllib.request.urlopen(source["url"], timeout=120) as response:
            blob = response.read()
        if hashlib.sha256(blob).hexdigest() != source["sha256"]:
            raise ValueError(f"public dataset hash mismatch: {name}")
        path.write_bytes(blob)
    fixture = make_fixture({name: (root / name).read_bytes() for name in SOURCES})
    (root / "fixture.json").write_bytes(canonical(fixture))
    return check(root)


def check(root: Path = ASSETS) -> dict:
    blobs = {}
    for name, source in SOURCES.items():
        blob = (root / name).read_bytes()
        if hashlib.sha256(blob).hexdigest() != source["sha256"]:
            raise ValueError(f"public dataset hash mismatch: {name}; run setup explicitly")
        blobs[name] = blob
    expected = canonical(make_fixture(blobs))
    if (root / "fixture.json").read_bytes() != expected:
        raise ValueError("retrieval fixture differs from frozen public sources; run setup explicitly")
    return {"dataset": "BEIR/SciFact", "license": LICENSE, "revision": REVISION,
            "qrels_revision": QRELS_REVISION, "fixture_sha256": hashlib.sha256(expected).hexdigest(),
            "sources": SOURCES, "queries": 24, "documents": 256,
            "scope": "restricted-candidate subset; unjudged documents treated as nonrelevant"}


def score_vectors(fixture: dict, documents, queries) -> dict:
    """Deterministic cosine ranking with ID tie-break, macro recall and full-rank MRR."""
    docs = fixture["documents"]
    tasks = fixture["queries"]
    if len(documents) != len(docs) or len(queries) != len(tasks):
        raise ValueError("embedding count differs from retrieval fixture")
    vectors = [[float(x) for x in vector] for vector in [*documents, *queries]]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or not all(E.valid_vector(vector) for vector in vectors):
        raise ValueError("invalid retrieval embeddings (dimension, nonfinite, empty or zero norm)")
    norms = [math.sqrt(sum(x * x for x in vector)) for vector in vectors]
    normalized = [[x / norm for x in vector] for vector, norm in zip(vectors, norms)]
    doc_vectors, query_vectors = normalized[:len(docs)], normalized[len(docs):]
    rows = []
    for query, vector in zip(tasks, query_vectors):
        ranked = sorted(zip(docs, doc_vectors),
                        key=lambda pair: (-sum(x * y for x, y in zip(vector, pair[1])), pair[0]["id"]))
        ids = [doc["id"] for doc, _ in ranked]
        relevant = set(query["relevant"])
        if not relevant or not relevant.issubset(ids):
            raise ValueError("query has missing relevance judgments in candidate corpus")
        first = min(ids.index(doc) + 1 for doc in relevant)
        row = {"query_id": query["id"], "relevant_documents": len(relevant),
               "reciprocal_rank": 1 / first, "first_relevant_rank": first, "top10": ids[:10]}
        for k in (1, 5, 10):
            row[f"hits@{k}"] = len(relevant.intersection(ids[:k]))
            row[f"recall@{k}"] = row[f"hits@{k}"] / len(relevant)
        rows.append(row)
    if not rows:
        raise ValueError("retrieval fixture has no queries")
    metrics = {"queries": len(rows), "documents": len(docs),
               "relevant_documents": sum(row["relevant_documents"] for row in rows),
               "mrr": sum(row["reciprocal_rank"] for row in rows) / len(rows),
               "reciprocal_rank_sum": sum(row["reciprocal_rank"] for row in rows)}
    for k in (1, 5, 10):
        metrics[f"recall@{k}"] = sum(row[f"recall@{k}"] for row in rows) / len(rows)
        metrics[f"hits@{k}"] = sum(row[f"hits@{k}"] for row in rows)
    return {"metrics": metrics, "queries": rows}


def compare_scores(native: dict, reference: dict) -> dict:
    left = {row["query_id"]: row for row in native["queries"]}
    right = {row["query_id"]: row for row in reference["queries"]}
    if left.keys() != right.keys():
        raise ValueError("retrieval providers did not score identical query sets")
    metrics = ("recall@1", "recall@5", "recall@10", "mrr")
    return {"direction": "antfly_minus_sentence_transformers", "paired_queries": len(left),
            "differences": {key: native["metrics"][key] - reference["metrics"][key] for key in metrics},
            "queries": [{"query_id": key,
                         **{metric: left[key][metric] - right[key][metric]
                            for metric in ("recall@1", "recall@5", "recall@10", "reciprocal_rank")}}
                        for key in sorted(left)]}


def native_provider(args, fixture: dict, out: Path) -> dict:
    server = C.AntflyServer(args.antfly, out, "quality-embed-antfly", extra_args=["--scratch-budget-mb", "8192"])
    info = {"arm": "antfly"}
    try:
        info.update(server.start())
        client = C.Client(server.port)

        def encode(rows, task):
            values = []
            for start in range(0, len(rows), 8):
                texts = [row["text"] for row in rows[start:start + 8]]
                status, body, _ = client.post("/ai/v1/embed", {"model": E.MODEL, "input": texts, "task_type": task})
                data = body.get("data") or []
                if status != 200 or len(data) != len(texts):
                    raise RuntimeError(f"retrieval native request failed: {status} {C.public_error(body)}")
                values.extend(item["embedding"] for item in data)
            return values

        documents = encode(fixture["documents"], "RETRIEVAL_DOCUMENT")
        queries = encode(fixture["queries"], "RETRIEVAL_QUERY")
        info["precision"] = C.native_precision(args.antfly, E.MODEL)
        info.update(score_vectors(fixture, documents, queries))
        info["status"] = "complete"
    except Exception as exc:
        info.update(status="failed", error=C.public_error(exc))
    finally:
        info["footprint"] = server.stop()
    return info


def reference_provider(fixture: dict) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(C.reference_paths()[E.MODEL], device=C.ACCEL)
    with torch.inference_mode():
        documents = model.encode([row["text"] for row in fixture["documents"]], prompt_name="document",
                                 batch_size=8, convert_to_numpy=True)
        queries = model.encode([row["text"] for row in fixture["queries"]], prompt_name="query",
                               batch_size=8, convert_to_numpy=True)
    C.torch_sync(torch)()
    return {"arm": "py", "status": "complete", "precision": C.reference_precision(model, default_policy="library_default"),
            "sentence_transformers": __import__("sentence_transformers").__version__, "torch": torch.__version__,
            **score_vectors(fixture, documents, queries)}


def run(args) -> bool:
    C.configure_backend(args.backend)
    C.exclusive_bench_lock()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "model": E.MODEL_KEY, "status": "failed", "providers": {}, "comparisons": {},
              "provenance": {"backend": C.ANTFLY_BACKEND, "metric": "cosine; macro Recall@1/5/10; MRR over all 256 candidates",
                             "batch_size": 8, "document_prompt": "document", "query_prompt": "query"}}
    try:
        result["provenance"].update(check())
        references = json.loads(C.REFERENCES.read_text(encoding="utf-8"))
        result["provenance"]["reference_model"] = [
            {key: row[key] for key in ("repo", "revision") if key in row}
            for row in references if row["repo"] == E.MODEL]
        fixture = json.loads((ASSETS / "fixture.json").read_text(encoding="utf-8"))
        result["providers"]["antfly"] = native_provider(args, fixture, out)
        reference = C.run_python_arm(__file__, ["--backend", C.ANTFLY_BACKEND], out, "quality-embed-py")
        reference.setdefault("status", "failed")
        result["providers"]["sentence-transformers"] = reference
        if result["providers"]["antfly"]["status"] == "complete" and reference.get("status") == "complete" and not reference.get("error"):
            result["comparisons"] = compare_scores(result["providers"]["antfly"], reference)
            result["status"] = "complete"
    except Exception as exc:
        result["error"] = C.public_error(exc)
    (out / "result.json").write_bytes(canonical(result))
    return result["status"] == "complete"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup")
    commands.add_parser("check")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--antfly", type=Path, required=True)
    run_parser.add_argument("--backend", choices=("metal", "cuda"), required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    child = commands.add_parser("pyarm")
    child.add_argument("--backend", choices=("metal", "cuda"), required=True)
    args = parser.parse_args()
    if args.command == "run":
        return 0 if run(args) else 1
    if args.command == "setup":
        print(json.dumps(setup(), separators=(",", ":")))
    elif args.command == "check":
        print(json.dumps(check(), separators=(",", ":")))
    else:
        C.configure_backend(args.backend)
        check()
        print(json.dumps(reference_provider(json.loads((ASSETS / "fixture.json").read_text(encoding="utf-8"))), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
