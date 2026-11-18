#!/usr/bin/env python3
"""Summarize latency, throughput, memory metadata, and paired output agreement."""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PAIRS = [
    ("antfly", "py"),
    ("antfly", "llama"),
    ("antfly", "transformers"),
    ("antfly", "whisper.cpp"),
]


def percentile(values: list[float], quantile: float):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_jsonl(path: Path) -> list[dict]:
    return C.load_jsonl(path)


def summarize_run(run: Path) -> dict:
    calls = load_jsonl(run / "calls.jsonl")
    lifecycles = load_jsonl(run / "lifecycles.jsonl")
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in calls:
        grouped[(row["cell"], row["arm"])][row["lc"]].append(row)
    cells: dict = {}
    for (cell, arm), per_lifecycle in sorted(grouped.items()):
        lifecycle_stats = []
        total = successful = 0
        errors = collections.Counter()
        for lifecycle, rows in sorted(per_lifecycle.items()):
            total += len(rows)
            ok = [row for row in rows if row.get("ok") is True]
            successful += len(ok)
            for row in rows:
                if row.get("ok") is not True:
                    errors[(row.get("err") or "")[:140]] += 1
            if len(ok) != len(rows) or not ok or any(
                not isinstance(row.get("ms"), (int, float)) or not math.isfinite(row["ms"]) or row["ms"] <= 0 for row in ok
            ):
                continue
            elapsed = [float(row["ms"]) for row in ok]
            items = sum(int(row["n_inputs"]) for row in ok)
            words = sum(int(row["words"]) for row in ok)
            concurrency = rows[0].get("conc", 1) or 1
            denominator = float(rows[0]["cell_wall_s"]) if concurrency > 1 else sum(elapsed) / 1000
            stats = {
                "lc": lifecycle,
                "p50_ms": statistics.median(elapsed),
                "p95_ms": percentile(elapsed, 0.95),
                "mean_ms": statistics.mean(elapsed),
                "items_per_s": items / denominator,
                "words_per_s": words / denominator,
                "n": len(elapsed),
                "total_s": denominator,
            }
            quality = [row["quality"] for row in ok if isinstance(row.get("quality"), dict)]
            if quality:
                stats["quality"] = quality[0] if len(quality) == 1 else quality
            provider_fields = (
                "timing_basis",
                "batch_semantics",
                "warmup_state",
                "audio_x_realtime_basis",
                "audio_x_realtime",
                "pages_per_second",
                "wall_ms",
                "model_load_ms",
                "native_timings_ms",
                "request_ms",
            )
            provider_metrics = {}
            for key in provider_fields:
                values = [row[key] for row in ok if row.get(key) is not None]
                if values:
                    provider_metrics[key] = values[0] if len(values) == 1 else values
            if provider_metrics:
                stats["provider_metrics"] = provider_metrics
            first_token = [row["ttft_ms"] for row in ok if row.get("ttft_ms") is not None]
            decode_rate = [row["decode_tok_s"] for row in ok if row.get("decode_tok_s") is not None]
            if first_token:
                stats["ttft_p50_ms"] = statistics.median(first_token)
            if decode_rate:
                stats["decode_tok_s_p50"] = statistics.median(decode_rate)
            lifecycle_stats.append(stats)
        entry = {
            "ok": successful,
            "total": total,
            "errors": dict(errors.most_common(3)),
            "lifecycles": lifecycle_stats,
        }
        for key in (
            "p50_ms",
            "p95_ms",
            "items_per_s",
            "words_per_s",
            "total_s",
            "ttft_p50_ms",
            "decode_tok_s_p50",
        ):
            values = [stats[key] for stats in lifecycle_stats if key in stats]
            if values:
                entry[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        cells.setdefault(cell, {})[arm] = entry
    for cell, arms in cells.items():
        comparisons = {}
        for first, second in PAIRS:
            if first not in arms or second not in arms:
                continue
            first_lifecycles = {stats["lc"]: stats for stats in arms[first]["lifecycles"]}
            second_lifecycles = {stats["lc"]: stats for stats in arms[second]["lifecycles"]}
            common = sorted(set(first_lifecycles) & set(second_lifecycles))
            if not common:
                continue
            latency_ratios = [
                second_lifecycles[index]["p50_ms"] / first_lifecycles[index]["p50_ms"] for index in common
            ]
            throughput_ratios = [
                first_lifecycles[index]["items_per_s"] / second_lifecycles[index]["items_per_s"] for index in common
            ]
            comparisons[f"{first}_vs_{second}"] = {
                "p50_ratio": {
                    "median": statistics.median(latency_ratios),
                    "min": min(latency_ratios),
                    "max": max(latency_ratios),
                },
                "throughput_ratio": {
                    "median": statistics.median(throughput_ratios),
                    "min": min(throughput_ratios),
                    "max": max(throughput_ratios),
                },
                "lifecycles": len(common),
            }
        if comparisons:
            arms["_comparisons"] = comparisons
    output = {
        "run": C.public_path(run),
        "cells": cells,
        "lifecycle_meta": [
            {
                key: value
                for key, value in lifecycle.items()
                if key
                in (
                    "arm",
                    "provider",
                    "lc",
                    "ready_s",
                    "first_call_ms",
                    "import_s",
                    "load_s",
                    "device",
                    "dtype",
                    "threads",
                    "torch",
                    "backend",
                    "error",
                    "rc",
                    "wall_s",
                    "mps_driver_allocated_bytes",
                    "cuda_max_memory_allocated_bytes",
                    "cuda_max_memory_reserved_bytes",
                    "version",
                    "version_source",
                    "required_backend",
                    "first_call_status",
                    "first_call_error",
                    "precision",
                    "weight_backing",
                    "weight_loading",
                    "cache_probe",
                    "decoding",
                    "warmup",
                    "warmups",
                    "repetitions",
                    "command",
                    "config",
                    "loading",
                )
            }
            | {
                "footprint": lifecycle.get("footprint"),
                "load1": (lifecycle.get("quiet") or {}).get("load1"),
            }
            for lifecycle in lifecycles
        ],
    }
    output["agreement"] = agreement(run)
    return output


def f1(left, right) -> float:
    left, right = set(map(tuple, left)), set(map(tuple, right))
    if not left and not right:
        return 1.0
    return 2 * len(left & right) / (len(left) + len(right))


def agreement(run: Path) -> dict:
    results: dict = {}
    outputs = load_jsonl(run / "outputs.jsonl")
    if outputs and "text" in outputs[0]:
        results.update(text_agreement(outputs))
    elif outputs:
        indexed = collections.defaultdict(dict)
        duplicates = set()
        for output in outputs:
            key = (output["cell"], output["lc"], output["k"], output["j"])
            if output["arm"] in indexed[key]:
                duplicates.add(key)
            indexed[key][output["arm"]] = output
        arms = {output["arm"] for output in outputs}
        for first, second in PAIRS:
            if first not in arms or second not in arms:
                continue
            per_cell = collections.defaultdict(lambda: {"span_f1": [], "cls_match": [], "rel_f1": [],
                "exact_entity_inputs": 0, "exact_structured_inputs": 0, "expected_inputs": 0, "invalid_or_unpaired_inputs": 0})
            for key, values in indexed.items():
                if not ({first, second} & values.keys()):
                    continue
                cell = key[0]
                metrics = per_cell[cell]
                metrics["expected_inputs"] += 1
                left_row, right_row = values.get(first, {}), values.get(second, {})
                left, right = left_row.get("o"), right_row.get("o")
                if (key in duplicates or not isinstance(left, dict) or not isinstance(right, dict)
                        or not left_row.get("input_sha256") or left_row["input_sha256"] != right_row.get("input_sha256")):
                    metrics["invalid_or_unpaired_inputs"] += 1
                    continue
                entities_left, entities_right = [entity[:3] for entity in left["e"]], [entity[:3] for entity in right["e"]]
                span = f1(entities_left, entities_right)
                metrics["span_f1"].append(span)
                canonical = lambda value: {key: sorted(labels) if isinstance(labels, list) else ([labels] if labels is not None else [])
                                           for key, labels in value.items()}
                classification = canonical(left["c"]) == canonical(right["c"])
                relations = f1(left["r"], right["r"])
                if left["c"] or right["c"]:
                    metrics["cls_match"].append(float(classification))
                if left["r"] or right["r"]:
                    metrics["rel_f1"].append(relations)
                metrics["exact_entity_inputs"] += span == 1.0
                metrics["exact_structured_inputs"] += span == 1.0 and classification and relations == 1.0
            results[f"{first}_vs_{second}"] = {
                cell: {key: statistics.mean(values) if values else None for key, values in metrics.items() if isinstance(values, list)}
                | {key: value for key, value in metrics.items() if not isinstance(value, list)}
                | {"paired_inputs": len(metrics["span_f1"])}
                for cell, metrics in per_cell.items()}
    recorded = load_jsonl(run / "agreement.jsonl")
    if recorded:
        results["recorded"] = recorded
    return results


SPOKEN = {"MR": "MISTER", "MRS": "MISSUS", "DR": "DOCTOR", "ST": "SAINT"}


def words(text: str) -> list[str]:
    normalized = "".join(character if character.isalnum() or character in " '" else " " for character in text.upper())
    return [SPOKEN.get(word, word) for word in normalized.split()]


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words, hypothesis_words = words(reference), words(hypothesis.strip())
    distances = list(range(len(hypothesis_words) + 1))
    for index, reference_word in enumerate(reference_words, 1):
        previous, distances[0] = distances[0], index
        for second_index, hypothesis_word in enumerate(hypothesis_words, 1):
            previous, distances[second_index] = distances[second_index], min(
                distances[second_index] + 1,
                distances[second_index - 1] + 1,
                previous + (reference_word != hypothesis_word),
            )
    return distances[len(hypothesis_words)] / max(len(reference_words), 1)


def text_agreement(outputs: list[dict]) -> dict:
    indexed = collections.defaultdict(dict)
    for output in outputs:
        indexed[(output["cell"], output["lc"], output["k"], output.get("repetition"))][output["arm"]] = output
    results = {}
    for first, second in PAIRS:
        per_cell = collections.defaultdict(
            lambda: {"exact": [], "prefix_share": [], "same_prompt_tokens": [], f"wer_{first}": [], f"wer_{second}": []}
        )
        for (cell, _lifecycle, _call, _repetition), values in indexed.items():
            if first not in values or second not in values:
                continue
            left, right = values[first]["text"], values[second]["text"]
            shared = 0
            while shared < min(len(left), len(right)) and left[shared] == right[shared]:
                shared += 1
            metrics = per_cell[cell]
            metrics["exact"].append(float(left == right))
            metrics["prefix_share"].append(shared / max(min(len(left), len(right)), 1))
            if values[first].get("prompt_tokens") is not None and values[second].get("prompt_tokens") is not None:
                metrics["same_prompt_tokens"].append(
                    float(values[first]["prompt_tokens"] == values[second]["prompt_tokens"])
                )
            if values[first].get("ref"):
                metrics[f"wer_{first}"].append(word_error_rate(values[first]["ref"], left))
                metrics[f"wer_{second}"].append(word_error_rate(values[first]["ref"], right))
        if per_cell:
            results[f"{first}_vs_{second}"] = {
                cell: {
                    "inputs": len(metrics["exact"]),
                    "exact": round(statistics.mean(metrics["exact"]), 4),
                    "prefix_share_median": round(statistics.median(metrics["prefix_share"]), 4),
                }
                | (
                    {"equal_prompt_token_count_fraction": round(statistics.mean(metrics["same_prompt_tokens"]), 4),
                     "token_identity_proven": False}
                    if metrics["same_prompt_tokens"]
                    else {}
                )
                | {
                    key: round(statistics.mean(values), 4)
                    for key, values in metrics.items()
                    if key.startswith("wer_") and values
                }
                for cell, metrics in per_cell.items()
            }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summaries = [summarize_run(run) for run in args.runs]
    encoded = json.dumps(summaries, ensure_ascii=False, separators=(",", ":"))
    if args.out:
        args.out.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
