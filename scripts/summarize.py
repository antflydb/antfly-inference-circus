#!/usr/bin/env python3
"""Build compact provider metrics from speed and isolated-memory measurements."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402
import common as C  # noqa: E402

WORKLOADS: dict[str, dict[str, Any]] = {
    "embed": {
        "workload": "embed",
        "model": "google/embeddinggemma-2",
        "model_key": "embeddinggemma2",
        "script": "embed_bench.py",
        "script_args": [],
        "arms": {"antfly": "antfly", "py": "sentence-transformers"},
        "cells": {
            "sentence_x1": 64,
            "sentence_b32": 8,
            "paragraph_x1": 64,
            "paragraph_b32": 8,
            "page_b8": 8,
            "long2k_x1": 8,
            "long8k_x1": 6,
        },
    },
    "gliner-base": {
        "workload": "gliner",
        "model": "fastino/gliner2.5-base-v1",
        "model_key": "gliner25-base",
        "script": "gliner_bench.py",
        "script_args": ["--model", "base"],
        "arms": {"antfly": "antfly", "py": "gliner2"},
        "cells": {
            "ent_sentence_x1": 64,
            "ent_paragraph_x1": 64,
            "ent_page_x1": 32,
            "ent_wiki_x1": 64,
            "mixed_paragraph_x1": 32,
            "ent_paragraph_b8": 8,
            "ent_paragraph_b32": 4,
        },
    },
    "gliner-decide": {
        "workload": "gliner",
        "model": "fastino/GLiNER2.5-Decide",
        "model_key": "gliner25-decide",
        "script": "gliner_bench.py",
        "script_args": ["--model", "decide"],
        "arms": {"antfly": "antfly", "py": "gliner2"},
        "cells": {"decide_x1": 64, "decide_b8": 16, "decide_b32": 4},
    },
    "gemma4": {
        "workload": "gemma4",
        "model": "google/gemma-4-E4B-it-qat-q4_0-gguf",
        "model_key": "gemma4-e4b",
        "script": "gemma4_bench.py",
        "script_args": [],
        "arms": {"antfly": "antfly", "llama": "llama.cpp"},
        "cells": {
            "text_short_256": 12,
            "page_summary_128": 12,
            "long2k_summary_64": 6,
            "image_describe_64": 8,
            "audio_transcribe_64": 8,
        },
    },
    "whisper": {
        "workload": "whisper",
        "model": "openai/whisper-small",
        "model_key": "whisper-small",
        "script": "whisper_bench.py",
        "script_args": [],
        "arms": {"antfly": "antfly", "transformers": "transformers", "whisper.cpp": "whisper.cpp"},
        "cells": {"short": 1, "medium": 1, "long": 1},
    },
    "ocr": {
        "workload": "ocr",
        "model": "lightonai/LightOnOCR-3-0.8B",
        "model_key": "lightonocr-3-0.8b",
        "script": "ocr_bench.py",
        "script_args": [],
        "arms": {"antfly": "antfly", "transformers": "transformers"},
        "cells": {"single_page": 1, "batch4_pages": 1},
    },
}


def selected_cells(spec: dict, only_cell: str | None, repetitions: int = 10) -> dict[str, int]:
    cells = spec["cells"]
    if only_cell is not None and only_cell not in cells:
        raise ValueError(f"unknown cell {only_cell!r} for {spec['model_key']}")
    multiplier = repetitions if spec.get("workload") in {"whisper", "ocr"} else 1
    return {cell: count * multiplier for cell, count in cells.items() if only_cell is None or cell == only_cell}


def aggregate_status(statuses) -> str:
    values = set(statuses)
    return next(iter(values)) if len(values) == 1 else "partial" if values else "failed"


def cell_completion(run: Path, arm: str, cell: str, expected_calls: int, lifecycles: int) -> dict:
    lifecycle_rows = [row for row in C.load_jsonl(run / "lifecycles.jsonl") if row.get("arm") == arm]
    calls = [row for row in C.load_jsonl(run / "calls.jsonl") if row.get("arm") == arm and row.get("cell") == cell]
    reasons = []
    expected = set(range(lifecycles))
    if len(lifecycle_rows) != lifecycles or {row.get("lc") for row in lifecycle_rows} != expected:
        reasons.append("missing or duplicate lifecycle record")
    for lifecycle in expected:
        rows = [row for row in calls if row.get("lc") == lifecycle]
        if len(rows) != expected_calls:
            reasons.append(f"lifecycle {lifecycle}: expected {expected_calls} calls, recorded {len(rows)}")
        identifiers = [(row.get("k"), row.get("repetition")) for row in rows]
        if len(set(identifiers)) != len(identifiers):
            reasons.append(f"lifecycle {lifecycle}: duplicate call identifiers")
    if any(row.get("lc") not in expected for row in calls):
        reasons.append("unexpected lifecycle")
    refused = [row for row in calls if C.call_refused(row)]
    failed = [row for row in calls if row.get("ok") is not True and not C.call_refused(row)]
    for row in calls:
        if row.get("ok") is True:
            if row.get("warmup_state", "complete") != "complete":
                reasons.append("successful call without completed per-cell warmup")
            value = row.get("ms")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                reasons.append("invalid successful-call duration")
    if failed:
        reasons.append("unclassified failed calls")
    status = "failed" if reasons else "refused" if len(refused) == len(calls) else "failed" if refused else "complete"
    if refused and len(refused) != len(calls):
        reasons.append("mixed success/refusal across requests; no complete-cell ratio")
    return {"status": status, "expected_calls": expected_calls * lifecycles, "recorded_calls": len(calls),
            "successful_calls": sum(row.get("ok") is True for row in calls),
            "refused_calls": len(refused), "failed_calls": len(failed),
            "reasons": list(dict.fromkeys(reasons)),
            **({"refusals": list({json.dumps({"status": row["status"], "error": row["response_error"]}, sort_keys=True):
                                  {"status": row["status"], "error": row["response_error"]} for row in refused}.values())} if refused else {})}



def metric_entry(entry: dict) -> dict:
    keys = (
        "ok",
        "total",
        "lifecycles",
        "p50_ms",
        "p95_ms",
        "items_per_s",
        "words_per_s",
        "total_s",
        "ttft_p50_ms",
        "decode_tok_s_p50",
    )
    return {key: entry[key] for key in keys if key in entry}


def recorded_agreement(analysis: dict, cells: dict[str, int], lifecycles: int) -> dict[str, list[dict]]:
    by_cell = {}
    for row in analysis.get("agreement", {}).get("recorded", []):
        if row.get("cell") in cells:
            by_cell.setdefault(row["cell"], []).append(row)
    return {cell: rows for cell, rows in by_cell.items()
            if {row.get("lc") for row in rows} == set(range(lifecycles))
            and all(row.get("complete") is True for row in rows)}


def practical_category(ratio: float) -> str:
    if 1 / 1.05 <= ratio <= 1.05:
        return "practical_tie"
    return "antfly_faster" if ratio > 1.05 else "reference_faster"


def provider_step(spec: dict, provider: str, phase: str, outcomes: dict, metrics: dict) -> dict:
    step = {"workload": spec["workload"], "model": spec["model"], "provider": provider, "phase": phase,
            "status": aggregate_status(value["status"] for value in outcomes.values()), "cells": outcomes}
    if metrics:
        step["metrics"] = {"cells": metrics}
    if step["status"] == "failed":
        step["error"] = "; ".join(f"{cell}: {reason}" for cell, value in outcomes.items()
                                  for reason in value.get("reasons", [])) or "no complete measurements"
    return step


def speed_summary(root: Path, key: str, spec: dict, cells: dict[str, int], lifecycles: int) -> tuple[list[dict], list[dict]]:
    run = root / "speed" / key
    analysis = analyze.summarize_run(run)
    if run.is_dir():
        C.write_json(run / "summary.json", [analysis])
    outcomes, steps = {}, []
    for arm, provider in spec["arms"].items():
        outcomes[arm], metrics = {}, {}
        for cell, count in cells.items():
            outcome = cell_completion(run, arm, cell, count, lifecycles)
            entry = analysis.get("cells", {}).get(cell, {}).get(arm)
            if outcome["status"] == "complete":
                if not entry or len(entry.get("lifecycles", [])) != lifecycles:
                    outcome["status"] = "failed"
                    outcome["reasons"].append("analyzed lifecycle metrics incomplete")
                else:
                    metrics[cell] = metric_entry(entry)
            outcomes[arm][cell] = outcome
        steps.append(provider_step(spec, provider, "speed", outcomes[arm], metrics))
    comparisons = []
    recorded = recorded_agreement(analysis, cells, lifecycles)
    for cell in cells:
        cell_data = analysis.get("cells", {}).get(cell, {})
        for pair, values in cell_data.get("_comparisons", {}).items():
            first, second = pair.split("_vs_", 1)
            if (outcomes.get(first, {}).get(cell, {}).get("status") != "complete"
                    or outcomes.get(second, {}).get(cell, {}).get("status") != "complete"
                    or values.get("lifecycles") != lifecycles):
                continue
            comparison = {"workload": spec["workload"], "model": spec["model"], "cell": cell, "phase": "speed",
                          "providers": [spec["arms"][first], spec["arms"][second]], "metrics": values,
                          "practical_category": practical_category(values["p50_ratio"]["median"]),
                          "tie_policy": {"multiplicative_margin": 0.05, "statistical_significance_claimed": False}}
            agreement = analysis.get("agreement", {}).get(pair, {}).get(cell)
            if agreement is not None:
                comparison["agreement"] = agreement
            if cell in recorded:
                comparison["recorded_agreement"] = recorded[cell]
            if key == "gemma4":
                comparison["prompt_parity"] = {"http_token_ids_observed": False,
                    "count_equality_is_not_token_identity": True, "diagnostics": "prompt_comparisons.json"}
            comparisons.append(comparison)
    return steps, comparisons


def footprint_metric(lifecycle: dict) -> dict | None:
    footprint = lifecycle.get("footprint") or {}
    lower, upper = footprint.get("memory_peak_lower_bytes"), footprint.get("memory_peak_upper_bytes")
    if (footprint.get("valid") is not True or type(lower) is not int or type(upper) is not int
            or not 0 < lower <= upper):
        return None
    return {key: value for key, value in footprint.items() if key in {
        "metric_version", "memory_peak_lower_bytes", "memory_peak_upper_bytes",
        "raw_process_tree_footprint_peak_bytes", "peak_rss_bytes", "method", "samples", "invalid_samples",
        "observer_seconds", "max_sample_gap_seconds", "interval_seconds", "coverage_errors", "coverage_scope", "coverage_evidence",
        "nvml_available", "nvml_counted", "nvml_errors", "read_failures"}}


def spread(values: list[float | int]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def memory_summary(root: Path, key: str, spec: dict, cells: dict[str, int], lifecycles: int = 3) -> list[dict]:
    steps = []
    for arm, provider in spec["arms"].items():
        metrics, outcomes = {}, {}
        for cell, count in cells.items():
            run = root / "memory" / key / cell
            outcome = cell_completion(run, arm, cell, count, lifecycles)
            rows = sorted((row for row in C.load_jsonl(run / "lifecycles.jsonl") if row.get("arm") == arm),
                          key=lambda row: row["lc"])
            observations = [footprint_metric(row) for row in rows]
            if outcome["status"] == "complete":
                if len(observations) != lifecycles or any(value is None for value in observations):
                    outcome["status"] = "failed"
                    outcome["reasons"].append("memory sampler incomplete or coverage invalid")
                elif len({value["metric_version"] for value in observations}) != 1:
                    outcome["status"] = "failed"
                    outcome["reasons"].append("inconsistent memory metric versions")
                else:
                    metrics[cell] = {
                        "metric_version": observations[0]["metric_version"], "lifecycles": observations,
                        "memory_peak_lower_bytes": spread([value["memory_peak_lower_bytes"] for value in observations]),
                        "memory_peak_upper_bytes": spread([value["memory_peak_upper_bytes"] for value in observations]),
                        "isolation": "one cell per fresh process lifecycle"}
            outcomes[cell] = outcome
        steps.append(provider_step(spec, provider, "memory", outcomes, metrics))
    return steps


def memory_comparisons(steps: list[dict], spec: dict, cells: dict[str, int]) -> list[dict]:
    by_provider = {step["provider"]: step for step in steps}
    antfly = by_provider.get("antfly", {}).get("metrics", {}).get("cells", {})
    comparisons = []
    for provider in spec["arms"].values():
        if provider == "antfly":
            continue
        reference = by_provider.get(provider, {}).get("metrics", {}).get("cells", {})
        for cell in cells.keys() & antfly.keys() & reference.keys():
            first, second = antfly[cell], reference[cell]
            if first["metric_version"] != second["metric_version"]:
                continue
            lower = second["memory_peak_lower_bytes"]["median"] / first["memory_peak_upper_bytes"]["median"]
            upper = second["memory_peak_upper_bytes"]["median"] / first["memory_peak_lower_bytes"]["median"]
            metrics = {"reference_to_antfly_ratio_interval": {"lower": lower, "upper": upper},
                       "antfly": {key: first[key] for key in ("memory_peak_lower_bytes", "memory_peak_upper_bytes")},
                       "reference": {key: second[key] for key in ("memory_peak_lower_bytes", "memory_peak_upper_bytes")},
                       "metric_version": first["metric_version"]}
            if lower == upper:
                metrics["reference_to_antfly_ratio"] = lower
            comparisons.append({"workload": spec["workload"], "model": spec["model"], "cell": cell,
                                "phase": "memory", "providers": ["antfly", provider], "metrics": metrics})
    return comparisons


def write_gliner_quality(root: Path, key: str, lifecycles: int, only_cell: str | None) -> None:
    spec = WORKLOADS[key]
    cells = selected_cells(spec, only_cell)
    run = root / "speed" / key
    analysis = analyze.summarize_run(run)
    agreement = analysis.get("agreement", {}).get("antfly_vs_py", {})
    providers = {}
    for arm, provider in spec["arms"].items():
        outcomes = {cell: cell_completion(run, arm, cell, count, lifecycles) for cell, count in cells.items()}
        providers[provider] = {"status": aggregate_status(value["status"] for value in outcomes.values()), "cells": outcomes}
    for cell in cells:
        paired = agreement.get(cell, {}).get("paired_inputs", 0)
        expected_inputs = sum(row.get("n_inputs", 0) for row in C.load_jsonl(run / "calls.jsonl")
                              if row.get("arm") == "antfly" and row.get("cell") == cell)
        if any(providers[provider]["cells"][cell]["status"] != "complete" for provider in providers):
            continue
        if paired != expected_inputs or expected_inputs == 0:
            for provider in providers:
                providers[provider]["cells"][cell]["status"] = "failed"
                providers[provider]["cells"][cell]["reasons"].append("structured output pairing incomplete")
                providers[provider]["status"] = aggregate_status(value["status"] for value in providers[provider]["cells"].values())
    C.write_json(root / "quality" / key / "result.json", {
        "schema_version": 1, "model": spec["model"], "status": aggregate_status(row["status"] for row in providers.values()),
        "providers": providers, "comparisons": agreement,
        "provenance": {"basis": "all measured structured outputs, paired by lifecycle/call/item/input digest",
                       "scope": "cross-provider labels/spans/classifications/relations agreement, not labeled task accuracy"}})


def quality_summary(root: Path, key: str, spec: dict) -> tuple[list[dict], list[dict]]:
    path = root / "quality" / key / "result.json"
    result = json.loads(path.read_text()) if path.is_file() else {}
    providers = result.get("providers", {})
    steps = []
    for provider in spec["arms"].values():
        metrics = providers.get(provider, {})
        status = metrics.get("status", "complete" if metrics and result.get("status") == "complete" else "failed")
        step = {"workload": spec["workload"], "model": spec["model"], "provider": provider, "phase": "quality",
                "status": status, "evidence": f"quality/{key}/result.json"}
        if metrics:
            step["metrics"] = metrics
        if status != "complete":
            step["error"] = metrics.get("error", result.get("error", "quality evidence missing or incomplete"))
        steps.append(step)
    return steps, [{"workload": spec["workload"], "model": spec["model"], "phase": "quality",
                    "providers": list(providers), "evidence": f"quality/{key}/result.json",
                    "metrics": result.get("comparisons", {}), "provenance": result.get("provenance", {})}] if providers else []


def build_summary(root: Path, keys: list[str], phases: set[str], lifecycles: int, only_cell: str | None = None,
                  memory_lifecycles: int = 3, repetitions: int = 10) -> dict:
    steps, comparisons = [], []
    for key in keys:
        spec = WORKLOADS[key]
        cells = selected_cells(spec, only_cell, repetitions)
        if "speed" in phases:
            values, pairs = speed_summary(root, key, spec, cells, lifecycles)
            steps.extend(values)
            comparisons.extend(pairs)
        if "memory" in phases:
            values = memory_summary(root, key, spec, cells, memory_lifecycles)
            steps.extend(values)
            comparisons.extend(memory_comparisons(values, spec, cells))
        if "quality" in phases:
            values, pairs = quality_summary(root, key, spec)
            steps.extend(values)
            comparisons.extend(pairs)
    return {"status": aggregate_status(step["status"] for step in steps), "steps": steps, "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--key", action="append", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--phase", action="append", choices=("speed", "memory", "quality"), required=True)
    parser.add_argument("--lifecycles", type=int, default=3)
    parser.add_argument("--memory-lifecycles", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--only-cell")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        summary = build_summary(args.run, args.key, set(args.phase), args.lifecycles, args.only_cell,
                                args.memory_lifecycles, args.repetitions)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        summary = {"status": "failed", "steps": [], "comparisons": [], "error": C.public_error(exc)}
    encoded = json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.out:
        args.out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
