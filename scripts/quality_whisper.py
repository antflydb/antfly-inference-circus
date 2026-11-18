#!/usr/bin/env python3
"""Offline Whisper task quality on a frozen 32-clip public LibriSpeech subset.

Only setup downloads. WER uses NFKC, casefold, ASCII words and apostrophes;
CER uses the same normalized text with spaces removed. Neither score is timing.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import whisper_bench as W

DATASET = "hf-internal-testing/librispeech_asr_dummy"
REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
FILE = "clean/validation-00000-of-00001.parquet"
SOURCE_SHA256 = "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILE}"
ASSETS = C.DATA / "quality" / "whisper"
CLIPS = 32
TIMING_RECORDINGS = {
    "medium": {"id": "1272-128104-0011", "seconds": 15.115},
    "long": {"id": "1272-128104-0004", "seconds": 29.4},
}
PROVENANCE = {
    "dataset": DATASET, "revision": REVISION, "file": FILE,
    "source_sha256": SOURCE_SHA256,
    "selection": "first 32 rows sorted lexicographically by unique LibriSpeech id",
    "license": "CC-BY-4.0", "license_url": "https://www.openslr.org/12",
    "attribution": "LibriSpeech: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur; audio derived from LibriVox",
    "normalization": "NFKC then casefold; ASCII alphanumeric words with internal apostrophes; CER removes spaces",
}


def _fixture() -> tuple[dict, list[bytes]]:
    import pyarrow.parquet as parquet
    import soundfile

    source = ASSETS / "source.parquet"
    if C.sha256_file(source) != SOURCE_SHA256:
        raise ValueError("quality Whisper source checksum differs; run setup")
    rows = sorted(parquet.read_table(source).to_pylist(), key=lambda row: row["id"])
    if len(rows) < CLIPS or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("quality Whisper needs at least 32 distinct source clips")
    entries, files = [], []
    for row in rows[:CLIPS]:
        samples, sample_rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
        if sample_rate != 16000 or samples.ndim != 1:
            raise ValueError("expected 16kHz mono source audio")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(samples.astype("<i2", copy=False).tobytes())
        content = buffer.getvalue()
        entries.append({"id": row["id"], "name": f"{row['id']}.wav", "text": row["text"],
                        "seconds": len(samples) / sample_rate,
                        "sha256": hashlib.sha256(content).hexdigest()})
        files.append(content)
    manifest = {"schema_version": 1, "provenance": PROVENANCE, "audio": entries,
                "clips": CLIPS,
                "reference_words": sum(len(W.normalize_text(entry["text"]).split()) for entry in entries)}
    return manifest, files


def setup() -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)
    source = ASSETS / "source.parquet"
    if not source.is_file() or C.sha256_file(source) != SOURCE_SHA256:
        request = urllib.request.Request(URL, headers={"User-Agent": "termite-circus-quality/1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
            raise ValueError("downloaded quality Whisper source checksum differs")
        source.write_bytes(content)
    manifest, files = _fixture()
    for entry, content in zip(manifest["audio"], files, strict=True):
        (ASSETS / entry["name"]).write_bytes(content)
    C.write_json(ASSETS / "manifest.json", manifest)
    return check()


def check() -> dict:
    # Re-derive from the hash-pinned source, so a locally edited reference/hash
    # cannot become an accepted ground-truth fixture merely by editing manifest.
    expected, _ = _fixture()
    observed = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    if observed != expected:
        raise ValueError("quality Whisper manifest differs from frozen source/selection")
    for entry in expected["audio"]:
        if C.sha256_file(ASSETS / entry["name"]) != entry["sha256"]:
            raise ValueError(f"quality Whisper audio checksum differs: {entry['id']}")
    return expected


def cells(manifest: dict) -> list[dict]:
    return [{"id": entry["id"], "calls": [{"id": entry["id"], "path": ASSETS / entry["name"],
             "ref": entry["text"], "seconds": entry["seconds"], "sha256": entry["sha256"]}]}
            for entry in manifest["audio"]]


def timing_clips(only: str | None = None) -> list[dict]:
    """Return intact, distinct source recordings; never synthesize duration."""
    if only not in (None, *TIMING_RECORDINGS):
        raise ValueError(f"unknown extended Whisper timing cell: {only}")
    manifest = check()
    by_id = {entry["id"]: entry for entry in manifest["audio"]}
    result = []
    for cell, selected in TIMING_RECORDINGS.items():
        if only is not None and only != cell:
            continue
        entry = by_id.get(selected["id"])
        if entry is None or entry["seconds"] != selected["seconds"]:
            raise ValueError(f"frozen Whisper {cell} recording identity/duration differs")
        result.append({"id": cell, "calls": [{
            "id": entry["id"], "path": ASSETS / entry["name"], "ref": entry["text"],
            "seconds": entry["seconds"], "sha256": entry["sha256"]}]})
    return result


def score_provider(result: dict, manifest: dict) -> dict:
    rows = result.get("rows", [])
    expected_ids = [entry["id"] for entry in manifest["audio"]]
    observed_ids = [row["call"]["cell"] for row in rows]
    if observed_ids != expected_ids or any(len(row["texts"]) != 1 or row["call"].get("ok") is False for row in rows):
        raise ValueError("quality provider returned missing, duplicated, reordered, or batched clips")
    transcripts = [row["texts"][0] for row in rows]
    score = W.quality(transcripts, [entry["text"] for entry in manifest["audio"]])
    score["items"] = [{**item, "id": entry["id"], "output_sha256": hashlib.sha256(text.encode()).hexdigest()}
                      for item, entry, text in zip(score["items"], manifest["audio"], transcripts, strict=True)]
    metadata = {key: value for key, value in result.items() if key not in {"rows", "footprint"}}
    problem = result.get("error") or (f"quality provider exited {result['rc']}" if result.get("rc", 0) else None)
    return {"status": "failed" if problem else "complete", "scores": score, "lifecycle": metadata,
            **({"error": C.public_error(problem)} if problem else {})}


def provider_outcome(arm: dict, manifest: dict) -> dict:
    try:
        return score_provider(arm, manifest)
    except Exception as error:
        rows = arm.get("rows", [])
        calls = [row["call"] for row in rows]
        expected = [entry["id"] for entry in manifest["audio"]]
        complete_attempts = [call["cell"] for call in calls] == expected
        refused = (complete_attempts and bool(calls) and all(C.call_refused(call) for call in calls)
                   and not arm.get("error") and not arm.get("rc", 0))
        return {"status": "refused" if refused else "failed",
                "error": "all planned clips were refused" if refused else C.public_error(error),
                "coverage": {"expected_clips": len(expected), "recorded_requests": len(calls),
                    "successful_clips": sum(call.get("ok", True) for call in calls),
                    "refused_clips": sum(C.call_refused(call) for call in calls),
                    "failed_clips": sum(call.get("ok") is False and not C.call_refused(call) for call in calls),
                    "missing_clips": max(0, len(expected) - len(calls))},
                "requests": [{key: call[key] for key in ("cell", "n_inputs", "ok", "status", "response_error") if key in call}
                             for call in calls],
                "lifecycle": {key: value for key, value in arm.items() if key not in {"rows", "footprint"}}}


def comparisons(providers: dict) -> list[dict]:
    output = []
    for index, left in enumerate(W.PROVIDERS):
        for right in W.PROVIDERS[index + 1:]:
            if providers.get(left, {}).get("status") != "complete" or providers.get(right, {}).get("status") != "complete":
                continue
            ls, rs = providers[left]["scores"], providers[right]["scores"]
            if (ls["clips"], ls["reference_words"], ls["reference_characters"]) != (rs["clips"], rs["reference_words"], rs["reference_characters"]):
                raise ValueError("paired quality denominators differ")
            output.append({"left": left, "right": right, "difference_direction": "left minus right; lower error is better",
                           "wer_difference": ls["wer"] - rs["wer"], "cer_difference": ls["cer"] - rs["cer"],
                           "reference_words": ls["reference_words"], "reference_characters": ls["reference_characters"],
                           "clips": ls["clips"],
                           "paired_word_edit_differences": [l["word_edits"] - r["word_edits"] for l, r in zip(ls["items"], rs["items"], strict=True)],
                           "paired_character_edit_differences": [l["character_edits"] - r["character_edits"] for l, r in zip(ls["items"], rs["items"], strict=True)]})
    return output


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "model": W.MODEL, "status": "failed", "providers": {},
              "comparisons": [], "provenance": dict(PROVENANCE)}
    try:
        C.configure_backend(args.backend)
        C.exclusive_bench_lock()
        manifest = check()
        result["provenance"].update({"clips": manifest["clips"], "reference_words": manifest["reference_words"],
            "fixture_manifest_sha256": C.sha256_file(ASSETS / "manifest.json"),
            "items": [{key: entry[key] for key in ("id", "sha256", "seconds")} for entry in manifest["audio"]],
            "matched_source_weights": W.verify_same_weights()})
        fixture_cells = cells(manifest)
        for provider in W.PROVIDERS:
            arm = {}
            try:
                if provider == "antfly":
                    args.repetitions = 1
                    arm = W.run_antfly(args, fixture_cells, args.out, 0)
                elif provider == "whisper.cpp":
                    arm = W.run_cpp_arm(fixture_cells, args.out, 0, 1)
                else:
                    arm = C.run_python_arm(str(Path(__file__)), ["--backend", args.backend], args.out, "quality-transformers")
                    W.sanitize_log(args.out / "quality-transformers.py.log")
                result["providers"][provider] = provider_outcome(arm, manifest)
            except Exception as exc:
                preserved = provider_outcome(arm, manifest) if arm else {}
                result["providers"][provider] = {**preserved, "status": "failed", "error": C.public_error(exc)}
        result["comparisons"] = comparisons(result["providers"])
        if all(result["providers"][p]["status"] == "complete" for p in W.PROVIDERS):
            result["status"] = "complete"
        elif all(result["providers"][p]["status"] in {"complete", "refused"} for p in W.PROVIDERS):
            result["status"] = "partial"
    except Exception as exc:
        result["error"] = C.public_error(exc)
    C.write_json(args.out / "result.json", result)
    return 0 if result["status"] in {"complete", "partial"} else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "pyarm":
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", required=True, choices=["metal", "cuda"])
        args = parser.parse_args(argv[1:])
        try:
            C.configure_backend(args.backend)
            result = W.transformers_arm(None, 0, 1, cells(check()))
        except Exception as exc:
            result = {"error": C.public_error(exc)}
        print(json.dumps(result, separators=(",", ":")))
        return 1 if "error" in result else 0
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup")
    sub.add_parser("check")
    command = sub.add_parser("run")
    command.add_argument("--antfly", type=Path, required=True)
    command.add_argument("--backend", choices=["metal", "cuda"], required=True)
    command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    manifest = setup() if args.command == "setup" else check()
    print(json.dumps({"clips": manifest["clips"], "reference_words": manifest["reference_words"], "verified": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
