#!/usr/bin/env python3
"""Offline CORD16 real-receipt OCR quality; only setup accesses the network.

CORD-v2 data is CC-BY-4.0 (commercial reuse allowed with attribution).
Raw scores are primary. Formatting/content normalization is diagnostic only.
Only valid_line words are annotated; visible unannotated text can increase
full-image insertion errors. These scores are not full-page ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.request
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import ocr_bench as O

ASSETS = C.DATA / "quality" / "ocr"
DATASET = "naver-clova-ix/cord-v2"
DATASET_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
SHARD = "data/test-00000-of-00001-9c204eb3f4e11791.parquet"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/{SHARD}"
SHARD_SHA256 = "51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
LICENSE_SOURCE_URL = "https://raw.githubusercontent.com/clovaai/cord/327310ce58c1623255821d062b3a759ff3789e3c/README.md"
LICENSE_SHA256 = "827f4828fa7e8262cb942f424e0c2db695b8422636977537949a24e483e9e7f6"
# First 16 rows of the frozen official test shard; no performance selection.
PINS = [
    ("test-0000", "8612d04b70f430f3aef07fbbd5200e382dcc4152b344cc2eff9f735f05a257c8", "2cee3d83487a721463376c8bcc6ba03a8106d2b1f2bb5e7179cf8b60c648f2af"),
    ("test-0001", "5852cbe48df03646524e717e987eb3d0ac329a52fd192070b5fda1222a053789", "b74f83fd2becd63f860dc755786565facd0d6554aba00e03204919478895a69d"),
    ("test-0002", "571ac6af26e75de8c06747a8c536c778f19a4562879090813571eddc74603d34", "610cf2a115b3fed4ee5e90710d8ca38337af131cb2a31c912fd77552acabb966"),
    ("test-0003", "8f73f0c4803ea8815040a50e3389d4f3fa15e8c28cb6d50ea9854f6f6c22bd63", "5f55140f35e4d9288a81aab4a2da52691ef59a0b9427ea3593d8a0ce5b2a48f1"),
    ("test-0004", "8f3eee7068c96e86cdb2e4b5c53085cb5e1439462edd55c373548cb1962801ad", "fb2ca69cbc120b5ac665d579b3db235c3fcc3bff020f021d9a694f2aaa7b1798"),
    ("test-0005", "0b63b666b54ea615d1d05733ee2c6fc24f20e1a6c77b333872fa80fff28b2c48", "3be86eb63c54d4c2d87e6478258b80e0162a18f6a652d73563f0b96475743dd3"),
    ("test-0006", "1c091eb7436b3e5a813eb6492c0350ef4eb14e3fdad78d42da35df669c757292", "2ef179382cad023dfd76ae0f9ec6a9d2a433480e1df1470f103df0cf1dd6f6a1"),
    ("test-0007", "810d3545f4105b19757c4b8b35934a477cde8cd1875302d9092315d2f960ef93", "ac259fc9cea1a868786a40dc56cb671d77354ff09ebff95942c81fb594a8248a"),
    ("test-0008", "58716693a6c0733358e66e64d7064a5a845424280bb55f9b5fa137b1af8c92fe", "2bb9009cd40e823c45de9903cc59c1f84e068c891d4aa9460f01096c37d5c4ec"),
    ("test-0009", "0411c4d9beb9be65165fdb70c2b60f8bbf9035cc9661be2b8d4c73bbfcff7972", "814abe36017fec68e511a91990f597518c52b83a3088a0eb118ccce20e387fac"),
    ("test-0010", "2a5d18c199bdbb75010a54430429cf694b35bdca8bf560d1d0db1fbb090dedb8", "f67b0d9c70705498ed0fdc2cc41f33a46795b53983d773660138844c661b4dbf"),
    ("test-0011", "d92479ab212713e7a65e56aacc5e68c0713597476608cbc394d997971043010b", "dfbd628562f2c6804201043d186163ad5e34340297a42e0fa1c6cb4b540d411e"),
    ("test-0012", "b5f8a3a9a5431b8e38a8e5a16011d6a5a457f7ae4c4ef31da77e589c0b1ad882", "570e70e7b09654bb420c858a389dd047e9783578e683675554a737b10fd8f0f3"),
    ("test-0013", "ca4bf89562b3f55c2c91968ca221a629e48386744a5322596d9c7345648ada13", "e9e3dcf80b9e4c72d10e9f7f3f744ee5bafb57ddb5680d55aa0805e841845196"),
    ("test-0014", "9dcf1f9f86c909b762767cb07d1cc0db30773d556154ce42e0210f65270ee39d", "89eb592ec9e63216e79f3603d0b6296bc3043eed9debabfcbe9e566bea9b304b"),
    ("test-0015", "f1cd67a7d67d04ba8d5b685553b1b5addbf2039d5954ab08c92139c0ad88a796", "ee76bd6376f62937fcd71e2d92ab9a2056baae65028be5d3c3bf65555235eb6a"),
]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def reference_text(annotation: dict[str, Any]) -> str:
    """Use actual CORD row_id line annotations, geometric reading order."""
    rows: dict[int, list[dict[str, Any]]] = {}
    for entity in annotation["valid_line"]:
        for word in entity["words"]:
            if not word["text"].strip():
                continue
            quad = word["quad"]
            xs, ys = [quad[f"x{i}"] for i in range(1, 5)], [quad[f"y{i}"] for i in range(1, 5)]
            if max(xs) <= min(xs) or max(ys) <= min(ys):
                raise ValueError("CORD word has invalid quadrilateral")
            rows.setdefault(int(word["row_id"]), []).append(
                {"text": word["text"].strip(), "left": min(xs), "top": min(ys)})
    ordered = sorted(rows.values(), key=lambda words: (min(w["top"] for w in words),
                                                       min(w["left"] for w in words)))
    text = "\n".join(" ".join(word["text"] for word in sorted(words, key=lambda w: (w["left"], w["top"])))
                     for words in ordered)
    if not text.strip():
        raise ValueError("CORD receipt contains no annotated words")
    return text


def provenance() -> dict[str, Any]:
    return {"dataset": DATASET, "split": "test", "receipts": len(PINS),
            "source_url": SOURCE_URL, "revision": DATASET_REVISION, "shard_sha256": SHARD_SHA256,
            "selection": "first 16 test-shard rows in published row order",
            "license": {"id": "CC-BY-4.0", "url": LICENSE_URL, "source_url": LICENSE_SOURCE_URL,
                        "snapshot_sha256": LICENSE_SHA256,
                        "terms": "commercial reuse permitted with attribution, license link and change notice"},
            "attribution": "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing; Park et al.; Clova AI Research / NAVER; 2019; https://github.com/clovaai/cord",
            "changes": "16 test rows selected; original image and ground_truth bytes preserved; reference text linearized from word annotations",
            "reference_policy": "all nonempty valid_line words, grouped by annotated row_id; rows ordered top-to-bottom, words left-to-right; no gt_parse semantic flattening",
            "coverage_limit": "CORD excludes some store/payment/other categories; annotations may omit visible text. Full-image raw OCR scores include insertion penalties for unannotated visible text and reading-order differences; not complete-page transcription ground truth.",
            "pins": [{"id": item, "image_sha256": image, "annotation_sha256": annotation}
                     for item, image, annotation in PINS]}


def setup() -> dict[str, Any]:
    import pyarrow.parquet as pq

    ASSETS.mkdir(parents=True, exist_ok=True)
    try:
        return check()
    except (FileNotFoundError, ValueError):
        pass
    with urllib.request.urlopen(LICENSE_SOURCE_URL, timeout=120) as response:
        license_bytes = response.read()
    if digest(license_bytes) != LICENSE_SHA256:
        raise ValueError("CORD license-source content differs from frozen SHA-256")
    (ASSETS / "cord-license-source.txt").write_bytes(license_bytes)
    # The 234 MB shard is setup-only scratch, not a retained benchmark asset.
    with tempfile.TemporaryDirectory(prefix="cord-setup-", dir=ASSETS) as temporary:
        shard_path = Path(temporary) / "test.parquet"
        checksum = hashlib.sha256()
        with urllib.request.urlopen(SOURCE_URL, timeout=180) as response, shard_path.open("wb") as target:
            while block := response.read(1024 * 1024):
                checksum.update(block)
                target.write(block)
        if checksum.hexdigest() != SHARD_SHA256:
            raise ValueError("CORD test shard differs from frozen SHA-256")
        parquet = pq.ParquetFile(shard_path)
        try:
            rows = next(parquet.iter_batches(batch_size=len(PINS), columns=["image", "ground_truth"])).to_pylist()
        finally:
            parquet.close()
        if len(rows) != len(PINS):
            raise ValueError("CORD test shard contains fewer than 16 rows")
        for (item, image_hash, annotation_hash), row in zip(PINS, rows, strict=True):
            image = row["image"]["bytes"]
            annotation = row["ground_truth"].encode("utf-8")
            if digest(image) != image_hash or digest(annotation) != annotation_hash:
                raise ValueError("CORD selected image/annotation differs from frozen SHA-256")
            if not image.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("CORD selected original image is not PNG")
            for folder, extension, value in [("images", "png", image), ("annotations", "json", annotation)]:
                target = ASSETS / folder / f"{item}.{extension}"
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(value)
    C.write_json(ASSETS / "provenance.json", provenance())
    return check()


def check() -> dict[str, Any]:
    path = ASSETS / "cord-license-source.txt"
    if not path.is_file() or C.sha256_file(path) != LICENSE_SHA256:
        raise ValueError("CORD license-source missing or hash mismatch; run setup")
    if json.loads((ASSETS / "provenance.json").read_text()) != provenance():
        raise ValueError("CORD provenance differs from frozen policy")
    refs = []
    for item, image_hash, annotation_hash in PINS:
        for folder, extension, expected in [("images", "png", image_hash), ("annotations", "json", annotation_hash)]:
            path = ASSETS / folder / f"{item}.{extension}"
            if not path.is_file() or C.sha256_file(path) != expected:
                raise ValueError(f"CORD missing or hash mismatch: {item}.{extension}")
        annotation = json.loads((ASSETS / "annotations" / f"{item}.json").read_text())
        if annotation["meta"]["split"] != "test" or int(annotation["meta"]["image_id"]) != int(item[-4:]):
            raise ValueError("CORD annotation test-row identity differs")
        text = reference_text(annotation)
        refs.append({"id": item, "reference_sha256": digest(text.encode()),
                     "reference_words": len(text.split()), "reference_characters": len(text)})
    return {"status": "complete", "provenance": provenance(), "references": refs}


def make_cells() -> list[dict[str, Any]]:
    check()
    pages = [{"id": item, "path": ASSETS / "images" / f"{item}.png",
              "ref": reference_text(json.loads((ASSETS / "annotations" / f"{item}.json").read_text()))}
             for item, _, _ in PINS]
    return [{"id": f"cord_batch4_{start // 4}", "calls": pages[start:start + 4]}
            for start in range(0, len(pages), 4)]


def normalize_markup(text: str) -> str:
    text = re.sub(r"<[^>]*>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"(?m)^\s*```[^\n]*$", "", text)
    text = re.sub(r"(?m)^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*$", "", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|>\s*)", "", text)
    text = re.sub(r"!?(\[([^\]]+)\])\([^)]*\)", r"\2", text)
    text = text.replace("|", " ")
    text = re.sub(r"[*_`~]", "", text)
    return " ".join(text.split())


def errors(text: str, reference: str) -> dict[str, Any]:
    words, reference_words = text.split(), reference.split()
    if not reference_words or not reference:
        raise ValueError("OCR references require nonzero word and character denominators")
    wedits = O.edit_distance(words, reference_words)
    cedits = O.edit_distance(text, reference)
    return {"word_edits": wedits, "reference_words": len(reference_words),
            "character_edits": cedits, "reference_characters": len(reference),
            "wer": wedits / len(reference_words), "cer": cedits / len(reference)}


def score_pages(texts: list[str], references: list[str], ids: list[str]) -> dict[str, Any]:
    if len(texts) != len(references) or len(texts) != len(ids):
        raise ValueError("OCR quality output/reference/id counts differ")
    items = []
    normalizers = {"raw": lambda text: text, "markup_normalized": normalize_markup,
                   "text_content_normalized": lambda text: O.normalize_characters(normalize_markup(text))}
    for item_id, text, reference in zip(ids, texts, references, strict=True):
        scores = {name: errors(normalize(text), normalize(reference)) for name, normalize in normalizers.items()}
        items.append({"id": item_id, "scores": scores, "output_sha256": digest(text.encode()),
                      "reference_sha256": digest(reference.encode()),
                      "formatting": {"output_lines": len(text.splitlines()), "reference_lines": len(reference.splitlines()),
                                     "output_characters": len(text), "markup_normalized_characters": len(normalize_markup(text)),
                                     "html_tags": len(re.findall(r"<[^>]*>", text)), "table_separators": text.count("|")}})
    totals = {}
    for name in normalizers:
        total = {key: sum(item["scores"][name][key] for item in items)
                 for key in ("word_edits", "reference_words", "character_edits", "reference_characters")}
        if not total["reference_words"] or not total["reference_characters"]:
            raise ValueError("OCR quality requires nonempty references")
        totals[name] = {**total, "wer": total["word_edits"] / total["reference_words"],
                        "cer": total["character_edits"] / total["reference_characters"]}
    return {"pages": len(items), "scores": totals, "items": items}


def comparisons(providers: dict[str, Any]) -> list[dict[str, Any]]:
    if any(providers.get(name, {}).get("status") != "complete" for name in O.PROVIDERS):
        return []
    left, right = (providers[name]["quality"] for name in O.PROVIDERS)
    if [item["id"] for item in left["items"]] != [item["id"] for item in right["items"]]:
        raise ValueError("paired OCR quality receipt IDs differ")
    return [{"metric": name, "difference_direction": "antfly minus transformers; lower WER/CER is better",
             "pages": len(left["items"]),
             "wer_difference": left["scores"][name]["wer"] - right["scores"][name]["wer"],
             "cer_difference": left["scores"][name]["cer"] - right["scores"][name]["cer"],
             "reference_words": left["scores"][name]["reference_words"],
             "reference_characters": left["scores"][name]["reference_characters"],
             "paired_pages": [{"id": a["id"], "wer_difference": a["scores"][name]["wer"] - b["scores"][name]["wer"],
                               "cer_difference": a["scores"][name]["cer"] - b["scores"][name]["cer"]}
                              for a, b in zip(left["items"], right["items"], strict=True)]}
            for name in left["scores"]]


def provider_outcome(info: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    rows = info.get("rows", [])
    calls = [row["call"] for row in rows]
    references = [call["ref"] for cell in cells for call in cell["calls"]]
    ids = [call["id"] for cell in cells for call in cell["calls"]]
    expected = [cell["id"] for cell in cells]
    observed = [call["cell"] for call in calls]
    attempted_pages = sum(call.get("n_inputs", 0) for call in calls)
    problem = info.get("error") or (f"reference subprocess exited {info['rc']}" if info.get("rc", 0) else None)
    result = {"status": "failed",
              "coverage": {"expected_pages": len(ids), "expected_requests": len(cells),
                  "recorded_requests": len(calls), "attempted_pages": attempted_pages,
                  "successful_pages": sum(len(row["texts"]) for row in rows if row["call"].get("ok", True)),
                  "refused_pages": sum(call.get("n_inputs", 0) for call in calls if C.call_refused(call)),
                  "failed_pages": sum(call.get("n_inputs", 0) for call in calls
                                      if call.get("ok") is False and not C.call_refused(call)),
                  "missing_pages": max(0, len(ids) - attempted_pages)},
              "requests": [{key: call[key] for key in ("cell", "n_inputs", "ok", "status", "response_error") if key in call}
                           for call in calls],
              "lifecycle": {key: value for key, value in info.items() if key not in {"rows", "footprint"}}}
    try:
        if observed != expected:
            raise ValueError("OCR quality returned missing/reordered batches")
        if calls and all(C.call_refused(call) for call in calls):
            if problem:
                raise RuntimeError(problem)
            result["status"] = "refused"
            return result
        if any(call.get("ok") is False for call in calls):
            raise ValueError("OCR quality includes unsuccessful requests")
        texts = [text for row in rows for text in row["texts"]]
        result["quality"] = score_pages(texts, references, ids)
        if problem:
            raise RuntimeError(problem)
        result["status"] = "complete"
    except Exception as error:
        result["error"] = C.public_error(error)
    return result


def run(args: argparse.Namespace) -> int:
    C.configure_backend(args.backend)
    C.exclusive_bench_lock()
    C.prepare_output(args.out)
    providers: dict[str, Any] = {}
    result: dict[str, Any] = {"schema_version": 1, "model": O.MODEL, "status": "failed",
                              "providers": providers, "comparisons": [], "provenance": provenance(),
                              "score_policy": {"primary": "raw case/punctuation-sensitive whitespace-token WER and all-character CER",
                                               "markup_normalized": "remove HTML/Markdown decoration; collapse whitespace; preserve case/punctuation",
                                               "text_content_normalized": "markup normalization then NFKC, lowercase, ASCII alphanumeric/numeric-punctuation tokens; diagnostic only"}}
    try:
        cells = make_cells()
        result["weight_identity"] = O.verify_same_weights()
        references = [call["ref"] for cell in cells for call in cell["calls"]]
        ids = [call["id"] for cell in cells for call in cell["calls"]]
        for provider in O.PROVIDERS:
            info = {}
            try:
                if provider == "antfly":
                    info = O.run_antfly(argparse.Namespace(antfly=args.antfly, repetitions=1), cells, args.out, 0)
                else:
                    info = C.run_python_arm(str(Path(__file__)), ["--backend", args.backend, "--out", str(args.out)],
                                            args.out, "cord-transformers")
                    O.sanitize_log(args.out / "cord-transformers.py.log")
                providers[provider] = provider_outcome(info, cells)
                if providers[provider]["status"] != "complete":
                    continue
                texts = [text for row in info["rows"] for text in row["texts"]]
                with (args.out / f"{provider}-transcriptions.jsonl").open("w", encoding="utf-8") as stream:
                    for item_id, text, reference in zip(ids, texts, references, strict=True):
                        stream.write(json.dumps({"id": item_id, "text": text, "reference": reference},
                                                ensure_ascii=False, separators=(",", ":")) + "\n")
            except Exception as exc:
                preserved = providers.get(provider)
                if preserved is None and info:
                    preserved = provider_outcome(info, cells)
                providers[provider] = {**(preserved or {}), "status": "failed", "error": C.public_error(exc)}
        result["comparisons"] = comparisons(providers)
        states = {providers[p]["status"] for p in O.PROVIDERS}
        result["status"] = "complete" if states == {"complete"} else "partial" if states <= {"complete", "refused"} else "failed"
    except Exception as exc:
        result["error"] = C.public_error(exc)
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode()) >= 1024 * 1024:
        raise ValueError("OCR quality result exceeds compact evidence limit")
    (args.out / "result.json").write_text(encoded + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": "result.json"}))
    return 0 if result["status"] in {"complete", "partial"} else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "pyarm":
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", required=True, choices=["metal", "cuda"])
        parser.add_argument("--out", required=True, type=Path)
        args = parser.parse_args(argv[1:])
        C.configure_backend(args.backend)
        try:
            value = O.transformers_arm(None, args.out, 0, 1, cells=make_cells())
        except Exception as exc:
            value = {"error": C.public_error(exc)}
        print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        return 1 if "error" in value else 0
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup")
    commands.add_parser("check")
    execute = commands.add_parser("run")
    execute.add_argument("--antfly", type=Path, required=True)
    execute.add_argument("--backend", choices=["metal", "cuda"], required=True)
    execute.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    print(json.dumps(setup() if args.command == "setup" else check(), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
