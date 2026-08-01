#!/usr/bin/env python3
"""Explicitly populate the offline Hugging Face cache for reference providers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def reference_model(benchmark: dict, provider: str) -> str | None:
    if provider in {"onnxruntime", "sentence-transformers", "gliner2", "transformers"}:
        return str(benchmark.get("reference_model") or benchmark["model"])
    if provider == "faster-whisper":
        return "Systran/faster-whisper-tiny"
    if provider == "tokenizers":
        if benchmark["model"] == "fixed_bert":
            return "google-bert/bert-base-uncased"
        if benchmark["model"] == "fixed_bpe":
            return "openai-community/gpt2"
        return str(benchmark["model"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(ROOT / "matrix.json"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--category")
    args = parser.parse_args()
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    models: set[str] = set()
    for benchmark in matrix["benchmarks"]:
        if args.profile not in benchmark["profiles"] or (args.category and benchmark["category"] != args.category):
            continue
        for provider in benchmark["providers"] if args.profile == "full" else []:
            model = reference_model(benchmark, provider)
            if model:
                models.add(model)
    for model in sorted(models):
        print(f"+ huggingface snapshot {model}", flush=True)
        snapshot_download(repo_id=model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
