#!/usr/bin/env python3
"""Build compact provider metrics from speed and isolated-memory measurements."""
from __future__ import annotations

import argparse
import json
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
            "long_book_w384": 15,
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
        "cells": {"short": 1, "batch4": 1, "multiwindow60": 1},
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


def selected_cells(spec: dict, only_cell: str | None) -> dict[str, int]:
    cells = spec["cells"]
    if only_cell is None:
        return dict(cells)
    if only_cell not in cells:
        raise ValueError(f"unknown cell {only_cell!r} for {spec['model_key']}")
    return {only_cell: cells[only_cell]}


def lifecycle_ok(row: dict) -> bool:
    return not row.get("error") and row.get("rc", 0) == 0 and row.get("status", "ok") in {"ok", "complete"}


def provider_completeness(
    run: Path,
    arm: str,
    cells: dict[str, int],
    lifecycles: int,
) -> tuple[bool, list[str]]:
    lifecycle_rows = [row for row in C.load_jsonl(run / "lifecycles.jsonl") if row.get("arm") == arm]
    calls = [row for row in C.load_jsonl(run / "calls.jsonl") if row.get("arm") == arm]
    reasons: list[str] = []
    expected_lifecycles = set(range(lifecycles))
    observed_lifecycles = [int(row.get("lc", -1)) for row in lifecycle_rows]
    if len(lifecycle_rows) != lifecycles or set(observed_lifecycles) != expected_lifecycles:
        reasons.append("missing or duplicate lifecycle record")
    if any(not lifecycle_ok(row) for row in lifecycle_rows):
        reasons.append("failed lifecycle")
    expected_groups = {(cell, lifecycle) for cell in cells for lifecycle in expected_lifecycles}
    observed_groups: dict[tuple[str, int], list[dict]] = {}
    for row in calls:
        key = (str(row.get("cell")), int(row.get("lc", -1)))
        observed_groups.setdefault(key, []).append(row)
    if set(observed_groups) != expected_groups:
        reasons.append("missing or unexpected cell measurement")
    for cell, lifecycle in expected_groups:
        rows = observed_groups.get((cell, lifecycle), [])
        if len(rows) != cells[cell]:
            reasons.append(f"{cell} lifecycle {lifecycle} call count incomplete")
        elif any(row.get("ok") is not True for row in rows):
            reasons.append(f"{cell} lifecycle {lifecycle} contains a failed call")
    return not reasons, list(dict.fromkeys(reasons))


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
    recorded = analysis.get("agreement", {}).get("recorded") or []
    by_cell: dict[str, list[dict]] = {}
    for row in recorded:
        cell = row.get("cell")
        if cell in cells:
            by_cell.setdefault(cell, []).append(row)
    complete = {}
    for cell in cells:
        rows = by_cell.get(cell, [])
        if (
            len(rows) == lifecycles
            and {int(row.get("lc", -1)) for row in rows} == set(range(lifecycles))
            and all(row.get("complete") is True for row in rows)
        ):
            complete[cell] = rows
    return complete


def speed_summary(root: Path, key: str, spec: dict, cells: dict[str, int], lifecycles: int) -> tuple[list[dict], list[dict]]:
    run = root / "speed" / key
    analysis = analyze.summarize_run(run)
    C.write_json(run / "summary.json", [analysis])
    complete_by_arm = {}
    steps = []
    for arm, provider in spec["arms"].items():
        complete, reasons = provider_completeness(run, arm, cells, lifecycles)
        complete_by_arm[arm] = complete
        metrics = {}
        if complete:
            for cell in cells:
                entry = analysis.get("cells", {}).get(cell, {}).get(arm)
                if not entry or len(entry.get("lifecycles", [])) != lifecycles:
                    complete = False
                    reasons.append(f"{cell} analyzed lifecycle metrics incomplete")
                    metrics = {}
                    break
                metrics[cell] = metric_entry(entry)
        complete_by_arm[arm] = complete
        step = {
            "workload": spec["workload"],
            "model": spec["model"],
            "provider": provider,
            "phase": "speed",
            "status": "complete" if complete else "failed",
        }
        if complete:
            step["metrics"] = {"cells": metrics}
        else:
            step["error"] = "; ".join(dict.fromkeys(reasons)) or "incomplete speed measurements"
        steps.append(step)
    comparisons = []
    recorded = recorded_agreement(analysis, cells, lifecycles)
    for cell in cells:
        cell_data = analysis.get("cells", {}).get(cell, {})
        for pair, values in (cell_data.get("_comparisons") or {}).items():
            first, second = pair.split("_vs_", 1)
            if not complete_by_arm.get(first) or not complete_by_arm.get(second):
                continue
            if values.get("lifecycles") != lifecycles:
                continue
            agreement = analysis.get("agreement", {}).get(pair, {}).get(cell)
            if key == "gemma4" and (
                agreement is None
                or agreement.get("inputs") != cells[cell]
                or agreement.get("same_prompt_tokens") != 1.0
            ):
                continue
            comparison = {
                "workload": spec["workload"],
                "model": spec["model"],
                "cell": cell,
                "providers": [spec["arms"][first], spec["arms"][second]],
                "metrics": values,
            }
            if key == "whisper" and spec["arms"][second] == "whisper.cpp":
                comparison["qualified_metrics"] = comparison.pop("metrics")
                comparison["qualification"] = {
                    "directly_comparable": False,
                    "reason": (
                        "whisper.cpp measures fresh-process first-inference compute with model load excluded, "
                        "while Antfly is server-resident and warmed; batch4 is sequential files in whisper.cpp "
                        "but concurrent requests in Antfly"
                    ),
                }
            if agreement is not None:
                comparison["agreement"] = agreement
            comparisons.append(comparison)
        if cell in recorded and all(complete_by_arm.values()):
            comparisons.append(
                {
                    "workload": spec["workload"],
                    "model": spec["model"],
                    "cell": cell,
                    "providers": list(spec["arms"].values()),
                    "agreement": recorded[cell],
                }
            )
    return steps, comparisons


def footprint_metric(lifecycle: dict) -> dict | None:
    footprint = lifecycle.get("footprint") or {}
    peak = footprint.get("process_tree_footprint_peak_bytes")
    if footprint.get("valid") is not True or not isinstance(peak, int) or peak <= 0:
        return None
    metric = {
        "process_tree_footprint_peak_bytes": peak,
        "peak_rss_bytes": footprint.get("peak_rss_bytes"),
        "method": footprint.get("method"),
    }
    return {key: value for key, value in metric.items() if value is not None}


def memory_summary(root: Path, key: str, spec: dict, cells: dict[str, int]) -> list[dict]:
    steps = []
    for arm, provider in spec["arms"].items():
        metrics = {}
        reasons = []
        for cell, expected_calls in cells.items():
            run = root / "memory" / key / cell
            complete, cell_reasons = provider_completeness(run, arm, {cell: expected_calls}, 1)
            lifecycles = [row for row in C.load_jsonl(run / "lifecycles.jsonl") if row.get("arm") == arm]
            footprint = footprint_metric(lifecycles[0]) if len(lifecycles) == 1 else None
            if not complete or footprint is None:
                reasons.extend(f"{cell}: {reason}" for reason in cell_reasons)
                if footprint is None:
                    reasons.append(f"{cell}: memory sampler incomplete")
                continue
            metrics[cell] = footprint
        complete = len(metrics) == len(cells)
        step = {
            "workload": spec["workload"],
            "model": spec["model"],
            "provider": provider,
            "phase": "memory",
            "status": "complete" if complete else "failed",
        }
        if complete:
            step["metrics"] = {"cells": metrics, "isolation": "one cell per fresh process lifecycle"}
        else:
            step["error"] = "; ".join(dict.fromkeys(reasons)) or "incomplete isolated-memory measurements"
        steps.append(step)
    return steps


def memory_comparisons(steps: list[dict], spec: dict, cells: dict[str, int]) -> list[dict]:
    by_provider = {step["provider"]: step for step in steps if step["status"] == "complete"}
    antfly = by_provider.get("antfly")
    if antfly is None:
        return []
    comparisons = []
    for provider in spec["arms"].values():
        if provider == "antfly" or provider not in by_provider:
            continue
        for cell in cells:
            antfly_peak = antfly["metrics"]["cells"][cell]["process_tree_footprint_peak_bytes"]
            reference_peak = by_provider[provider]["metrics"]["cells"][cell]["process_tree_footprint_peak_bytes"]
            comparisons.append(
                {
                    "workload": spec["workload"],
                    "model": spec["model"],
                    "cell": cell,
                    "phase": "memory",
                    "providers": ["antfly", provider],
                    "metrics": {
                        "antfly_peak_bytes": antfly_peak,
                        "reference_peak_bytes": reference_peak,
                        "reference_to_antfly_ratio": reference_peak / antfly_peak,
                        "isolation": "one cell per fresh process lifecycle",
                    },
                }
            )
    return comparisons


def build_summary(
    root: Path,
    keys: list[str],
    phases: set[str],
    lifecycles: int,
    only_cell: str | None = None,
) -> dict:
    steps, comparisons = [], []
    for key in keys:
        spec = WORKLOADS[key]
        cells = selected_cells(spec, only_cell)
        if "speed" in phases:
            speed_steps, speed_comparisons = speed_summary(root, key, spec, cells, lifecycles)
            steps.extend(speed_steps)
            comparisons.extend(speed_comparisons)
        if "memory" in phases:
            memory_steps = memory_summary(root, key, spec, cells)
            steps.extend(memory_steps)
            comparisons.extend(memory_comparisons(memory_steps, spec, cells))
    complete = bool(steps) and all(step["status"] == "complete" for step in steps)
    return {"status": "complete" if complete else "failed", "steps": steps, "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--key", action="append", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--phase", action="append", choices=("speed", "memory"), required=True)
    parser.add_argument("--lifecycles", type=int, default=3)
    parser.add_argument("--only-cell")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        summary = build_summary(args.run, args.key, set(args.phase), args.lifecycles, args.only_cell)
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
