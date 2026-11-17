#!/usr/bin/env python3
"""Offline Gemma task accuracy: 12 GSM8K and 12 ARC-Challenge test questions.

ScienceQA is intentionally replaced by ARC-Challenge: ScienceQA's data license
is noncommercial; ARC's official dataset card specifies CC-BY-SA-4.0. These tiny,
ordered subsets are diagnostics, not estimates of full benchmark accuracy.
Only setup downloads data. Both providers reuse the timing server and request
policies (including the projector, greedy decoding, disabled thinking and EOS).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import gemma4_bench as G

DATA = C.DATA / "quality" / "gemma"
GSM_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
ARC_IDS = ["Mercury_7175875", "Mercury_SC_409171", "Mercury_SC_408547", "Mercury_407327",
           "MCAS_2006_9_44", "Mercury_7270393", "MCAS_2014_5_7", "Mercury_7086660",
           "Mercury_7168805", "MCAS_2003_8_11", "Mercury_7250058", "Mercury_7012740"]
SOURCES = {
    "gsm8k.jsonl": {
        "url": f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REVISION}/grade_school_math/data/test.jsonl",
        "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        "revision": GSM_REVISION, "license": "MIT", "attribution": "OpenAI GSM8K",
        "license_url": f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REVISION}/LICENSE",
    },
    "arc.parquet": {
        "url": f"https://huggingface.co/datasets/allenai/ai2_arc/resolve/{ARC_REVISION}/ARC-Challenge/test-00000-of-00001.parquet",
        "sha256": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
        "revision": ARC_REVISION, "license": "CC-BY-SA-4.0", "attribution": "AI2 Reasoning Challenge, Clark et al. (2018)",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "license_evidence": f"https://huggingface.co/datasets/allenai/ai2_arc/blob/{ARC_REVISION}/README.md",
    },
}
MAX_TOKENS = 256


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def tasks_from_sources(sources: dict[str, bytes]) -> list[dict]:
    import pyarrow.parquet as pq

    gsm = [json.loads(line) for line in sources["gsm8k.jsonl"].splitlines() if line]
    arc = pq.read_table(io.BytesIO(sources["arc.parquet"])).to_pylist()
    tasks = []
    for index, row in enumerate(gsm[:12]):
        tasks.append({"dataset": "gsm8k", "id": f"test:{index}", "question": row["question"],
                      "answer": row["answer"].split("####")[-1].strip()})
    for row in arc[:12]:
        tasks.append({"dataset": "arc_challenge", "id": row["id"], "question": row["question"],
                      "choices": row["choices"], "answer": row["answerKey"]})
    return tasks


def check() -> tuple[list[dict], dict]:
    blobs = {}
    for name, source in SOURCES.items():
        data = (DATA / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise RuntimeError(f"quality Gemma source checksum mismatch: {name}; run setup")
        blobs[name] = data
    tasks = tasks_from_sources(blobs)
    if len(tasks) != 24 or [t["id"] for t in tasks[12:]] != ARC_IDS:
        raise RuntimeError("frozen quality subset IDs changed")
    fixture = (DATA / "tasks.json").read_bytes()
    if fixture != encoded(tasks):
        raise RuntimeError("quality Gemma task fixture differs from frozen sources")
    return tasks, {"sources": SOURCES, "subset_sha256": hashlib.sha256(fixture).hexdigest(),
                   "selection": "first 12 test rows from each frozen source; no training examples, no filtering",
                   "sample_ids": {key: [t["id"] for t in tasks if t["dataset"] == key]
                                  for key in ("gsm8k", "arc_challenge")},
                   "modifications": "ordered subset and final GSM8K numeric answers extracted; ARC choices unchanged",
                   "science_substitution": "ARC-Challenge replaces noncommercial ScienceQA; no oracle facts supplied"}


def setup() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    blobs = {}
    for name, source in SOURCES.items():
        path = DATA / name
        data = path.read_bytes() if path.exists() else b""
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            with urllib.request.urlopen(source["url"], timeout=120) as response:
                data = response.read(2 << 20)
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise RuntimeError(f"download checksum mismatch: {name}")
        path.write_bytes(data)
        blobs[name] = data
    (DATA / "tasks.json").write_bytes(encoded(tasks_from_sources(blobs)))
    _, provenance = check()
    C.write_json(DATA / "provenance.json", provenance)
    return provenance


def prompt(task: dict) -> str:
    question = task["question"]
    if task["dataset"] == "gsm8k":
        return question + "\nSolve the problem. End with a line 'Final answer: NUMBER'."
    choices = task["choices"]
    options = "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
    return question + "\n" + options + "\nChoose one option. End with a line 'Final answer: LETTER'."


def final_answer(text: str, dataset: str) -> str | None:
    # Explicit final-answer markers take precedence over incidental reasoning.
    markers = re.findall(r"(?im)^\s*(?:\*\*)?(?:Final answer\s*:|####)\s*(.+?)\s*$", text)
    value = markers[-1] if markers else next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    value = value.replace("**", "").strip()
    if dataset == "arc_challenge":
        match = re.fullmatch(r"\(?([A-Da-d])\)?[.!]?", value)
        return match.group(1).upper() if match else None
    match = re.fullmatch(r"\$?\s*([+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*\.?", value)
    if match is None:
        return None
    try:
        return str(Decimal(match.group(1).replace(",", "")).normalize())
    except InvalidOperation:
        return None


def score(task: dict, text: str, ok: bool = True) -> dict:
    answer = final_answer(text, task["dataset"]) if ok else None
    expected = final_answer(task["answer"], task["dataset"])
    return {"answer": answer, "expected": expected, "correct": bool(ok and answer is not None and answer == expected)}


def summarize(tasks: list[dict], providers: dict) -> dict:
    by_dataset = {}
    for dataset in ("gsm8k", "arc_challenge"):
        indices = [i for i, t in enumerate(tasks) if t["dataset"] == dataset]
        denominator = len(indices)
        scores = {}
        for arm, provider in providers.items():
            correct = sum(provider["samples"][i]["correct"] for i in indices)
            scores[arm] = {"correct": correct, "denominator": denominator, "accuracy": correct / denominator}
        a, b = providers["antfly"]["samples"], providers["llama.cpp"]["samples"]
        valid = [i for i in indices if a[i]["ok"] and b[i]["ok"]]
        by_dataset[dataset] = {
            "scores": scores, "accuracy_difference_antfly_minus_reference": scores["antfly"]["accuracy"] - scores["llama.cpp"]["accuracy"],
            "paired_denominator": denominator, "both_correct": sum(a[i]["correct"] and b[i]["correct"] for i in indices),
            "antfly_only_correct": sum(a[i]["correct"] and not b[i]["correct"] for i in indices),
            "reference_only_correct": sum(b[i]["correct"] and not a[i]["correct"] for i in indices),
            "text_equality": {"equal": sum(a[i]["text_sha256"] == b[i]["text_sha256"] for i in valid),
                              "denominator": len(valid), "scope": "separate from task accuracy; successful paired responses only"},
        }
    return by_dataset


def run(args) -> dict:
    C.configure_backend(args.backend)
    tasks, provenance = check()
    files = G.verify_files()
    C.exclusive_bench_lock()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise RuntimeError("quality result already exists; choose a fresh output directory")
    result = {"schema_version": 1, "model": G.MODEL_KEY, "status": "complete", "providers": {},
              "comparisons": {}, "provenance": provenance,
              "decoding": {"temperature": 0, "max_tokens": MAX_TOKENS, "ignore_eos": True,
                           "enable_thinking": False, "prompt_cache": False},
              "scope": "small frozen text-only task diagnostics, not full benchmark or multimodal task accuracy"}
    native_diagnostics = {}
    for index in (0, 12):
        native_diagnostics[tasks[index]["dataset"]] = G.native_prompt_evidence(args.antfly, {"text": prompt(tasks[index])})
    for arm in ("antfly", "llama"):
        key = "antfly" if arm == "antfly" else "llama.cpp"
        provider = {"samples": [], "lifecycle": {}}
        result["providers"][key] = provider
        server = (C.AntflyServer(args.antfly, out, "quality-antfly", models_dir=G.GEN_MODELS_DIR,
                                extra_args=["--preload-model", f"generator:{C.ANTFLY_BACKEND}:{G.MODEL}",
                                            "--host-budget-mb", str(G.HOST_BUDGET_MB)])
                  if arm == "antfly" else G.LlamaServer(args.llama_server, files, out, "quality-llama"))
        endpoint = "/ai/v1/generate" if arm == "antfly" else "/v1/chat/completions"
        try:
            provider["lifecycle"].update(server.start())
            provider["lifecycle"]["precision"] = (C.native_precision(args.antfly, G.MODEL, models_dir=G.GEN_MODELS_DIR)
                if arm == "antfly" else {"implementation": "llama.cpp", "binary": C.file_identity(Path(shutil.which(args.llama_server) or args.llama_server)),
                                        "version": G.llama_version(args.llama_server), "requested_kv": {"key": "f16", "value": "f16"},
                                        "compute_dtype": "not directly observed; GGUF storage is not arithmetic precision"})
            for index, task in enumerate(tasks):
                payload = G.request_body({"text": prompt(task)}, {"max_tokens": MAX_TOKENS}, arm)
                if index in (0, 12):
                    diagnostic = native_diagnostics[task["dataset"]] if arm == "antfly" else G.llama_prompt_evidence(server.port, payload)
                    provider.setdefault("prompt_diagnostics", {})[task["dataset"]] = diagnostic
                response = G.stream(server.port, endpoint, payload)
                text = response.get("text", "")
                ok = response["status"] == 200 and bool(text.strip())
                provider["samples"].append({"dataset": task["dataset"], "id": task["id"], "ok": ok,
                    "attempted": True, "status": response["status"], "error": response.get("error"),
                    "response_error": response.get("response_error"),
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "usage": response.get("usage"), **score(task, text, ok)})
                if not ok:
                    result["status"] = "failed"
        except Exception as exc:
            result["status"] = "failed"
            provider["error"] = C.public_error(exc)
        finally:
            for field, observe in (
                ("footprint", server.stop),
                ("weight_loading", lambda: G.loading_evidence(server, files, arm)),
            ):
                try:
                    provider["lifecycle"][field] = observe()
                except Exception as exc:
                    result["status"] = "failed"
                    provider["lifecycle"].setdefault("teardown_errors", {})[field] = C.public_error(exc)
            if provider["lifecycle"].get("teardown_errors"):
                provider["error"] = "; ".join(filter(None, [
                    provider.get("error"), *provider["lifecycle"]["teardown_errors"].values()]))
        # Failed/missing operations remain in the original denominator.
        for task in tasks[len(provider["samples"]):]:
            provider["samples"].append({"dataset": task["dataset"], "id": task["id"], "ok": False,
                                        "attempted": False, "text_sha256": None, **score(task, "", False)})
        provider["status"] = "complete" if all(sample["ok"] for sample in provider["samples"]) and not provider.get("error") else "failed"
    result["comparisons"] = summarize(tasks, result["providers"])
    result["prompt_comparisons"] = {dataset: G.compare_prompts(native_diagnostics[dataset],
        result["providers"]["llama.cpp"].get("prompt_diagnostics", {}).get(dataset, {})) for dataset in native_diagnostics}
    if len(encoded(result)) >= 1 << 20:
        raise RuntimeError("quality result exceeds compact evidence limit")
    C.write_json(out / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup")
    commands.add_parser("check")
    execute = commands.add_parser("run")
    execute.add_argument("--antfly", type=Path, required=True)
    execute.add_argument("--backend", choices=("metal", "cuda"), required=True)
    execute.add_argument("--out", required=True)
    execute.add_argument("--llama-server", default=os.environ.get("TERMITE_LLAMA_SERVER", shutil.which("llama-server") or "llama-server"))
    args = parser.parse_args()
    if args.command == "setup":
        print(json.dumps(setup(), separators=(",", ":")))
    elif args.command == "check":
        print(json.dumps(check()[1], separators=(",", ":")))
    else:
        return 0 if run(args)["status"] == "complete" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
