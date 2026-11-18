#!/usr/bin/env python3
"""Paired LightOnOCR-3-0.8B benchmark using Antfly and Transformers.

The deterministic public pages make reference accuracy measurable without a
private document corpus.  Both arms receive the same PNG bytes, empty OCR
prompt, 4,096-token ceiling, disabled thinking, and deterministic decoding.
"""
from __future__ import annotations

import argparse
import base64
import http.client
import importlib.metadata
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODEL = "lightonai/LightOnOCR-3-0.8B"
MODELS = [MODEL]
WORKLOAD = "ocr"
PROVIDERS = ["antfly", "transformers"]
CELLS = ["single_page", "batch4_pages"]
REVISION = "36d8636925327c671bd8e479c517a004f7566bd9"
SOURCE_WEIGHT_SHA256 = "ae46aa6fd7136d57297ac006ed9ec345b7a2083feb2e99501bedcb014ca3efdf"
MAX_TOKENS = 4096
PAGE_TEXTS = [
    "TERMITE SUPPLY COMPANY\nINVOICE 2026-0042\nCUSTOMER NORTH WIND LABORATORY\nBRASS FASTENERS 12 48.00\nSILK CORD 5 25.00\nSHIPPING 11.50\nTOTAL 84.50",
    "FIELD NOTE SEVEN\nDATE 13 NOVEMBER 2026\nEUROPEAN STARLINGS GATHERED NEAR THE WEST FENCE\nTHE WEATHER WAS CLEAR AND THE WIND WAS LIGHT\nOBSERVATION ENDED AT 16 30",
    "MONTHLY COUNTS\nSPECIES JAN FEB MAR\nMOTHS 18 24 31\nBEETLES 7 11 9\nBUTTERFLIES 3 8 14\nTOTAL 28 43 54",
    "LOCAL INFERENCE REPORT\nSMALL MODELS CAN KEEP PRIVATE DOCUMENTS ON THE DEVICE\nMEASURE ACCURACY LATENCY AND MEMORY TOGETHER\nREPEAT EACH LIFECYCLE BEFORE PUBLISHING RESULTS",
]


def public_error(exc: BaseException) -> str:
    return C.public_error(exc)

def sanitize_log(path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    private_paths = {C.ROOT, C.RESULTS, C.DATA, C.MODELS_DIR, C.HF_HOME,
                     C.CORPUS.parent, C.MEDIA, Path.home()}
    for private in sorted((str(path) for path in private_paths), key=len, reverse=True):
        text = text.replace(private, C.public_path(private))
    path.write_text(text, encoding="utf-8")


def normalize_words(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", text))


def normalize_characters(text: str) -> str:
    return " ".join(normalize_words(text).split())


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, rvalue in enumerate(right, 1):
        current = [row]
        for column, lvalue in enumerate(left, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (lvalue != rvalue)))
        previous = current
    return previous[-1]


def quality(texts: list[str], references: list[str]) -> dict[str, Any]:
    if len(texts) != len(references):
        raise ValueError("OCR output/reference count differs")
    word_edits = reference_words = character_edits = reference_characters = exact = 0
    items = []
    for text, reference in zip(texts, references, strict=True):
        observed_words = normalize_words(text).split()
        expected_words = normalize_words(reference).split()
        observed_chars = normalize_characters(text)
        expected_chars = normalize_characters(reference)
        wedits = edit_distance(observed_words, expected_words)
        cedits = edit_distance(observed_chars, expected_chars)
        word_edits += wedits
        reference_words += len(expected_words)
        character_edits += cedits
        reference_characters += len(expected_chars)
        is_exact = observed_words == expected_words
        exact += is_exact
        items.append({"word_edits": wedits, "reference_words": len(expected_words),
                      "character_edits": cedits, "reference_characters": len(expected_chars),
                      "exact_normalized": is_exact})
    wer = word_edits / max(1, reference_words)
    cer = character_edits / max(1, reference_characters)
    return {"wer": wer, "cer": cer, "word_accuracy": max(0.0, 1.0 - wer),
            "character_accuracy": max(0.0, 1.0 - cer), "word_edits": word_edits,
            "reference_words": reference_words, "character_edits": character_edits,
            "reference_characters": reference_characters, "exact_normalized_count": exact,
            "items": items}

def _wrap_page_text(draw, font, text: str, max_width: int) -> str:
    rendered: list[str] = []
    for source_line in text.splitlines():
        words = source_line.split()
        if not words:
            rendered.append("")
            continue
        line = words[0]
        if draw.textlength(line, font=font) > max_width:
            raise ValueError("OCR fixture contains a word wider than the text area")
        for word in words[1:]:
            candidate = f"{line} {word}"
            if draw.textlength(candidate, font=font) <= max_width:
                line = candidate
            else:
                rendered.append(line)
                line = word
                if draw.textlength(line, font=font) > max_width:
                    raise ValueError("OCR fixture contains a word wider than the text area")
        rendered.append(line)
    return "\n".join(rendered)


def render_pages(destination: Path, *, reuse_existing: bool = False) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw, ImageFont

    destination.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=36)
    pages = []
    for index, text in enumerate(PAGE_TEXTS):
        path = destination / f"ocr-page-{index}.png"
        image = Image.new("RGB", (1024, 1344), "white")
        draw = ImageDraw.Draw(image)
        rendered_text = _wrap_page_text(draw, font, text, 844)
        text_bbox = draw.multiline_textbbox((90, 100), rendered_text, font=font, spacing=24)
        if text_bbox[0] < 90 or text_bbox[1] < 100 or text_bbox[2] > 934 or text_bbox[3] > 1240:
            image.close()
            raise ValueError(f"OCR fixture text exceeds page margins: page-{index}")
        if not reuse_existing or not path.exists():
            draw.rectangle((55, 55, 969, 1289), outline="black", width=3)
            draw.multiline_text((90, 100), rendered_text, fill="black", font=font, spacing=24)
            image.save(path, format="PNG", optimize=False)
        image.close()
        with Image.open(path) as existing:
            if existing.size != (1024, 1344):
                raise ValueError(f"unexpected OCR page dimensions: {path.name}")
        pages.append({"id": f"page-{index}", "path": path, "ref": text,
                      "rendered_text": rendered_text, "rendered_lines": len(rendered_text.splitlines()),
                      "text_bbox": list(text_bbox), "sha256": C.sha256_file(path),
                      "width": 1024, "height": 1344})
    return pages


def make_cells(only: str | None, out: Path, *, reuse_existing: bool = False) -> list[dict[str, Any]]:
    pages = render_pages(out / "inputs", reuse_existing=reuse_existing)
    cells = [{"id": "single_page", "calls": pages[:1]},
             {"id": "batch4_pages", "calls": pages}]
    selected = [cell for cell in cells if only in (None, cell["id"])]
    if not selected:
        raise ValueError(f"unknown cell: {only}")
    return selected


def image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def antfly_body(cell: dict[str, Any]) -> dict[str, Any]:
    return {"model": MODEL, "images": [{"url": image_data_uri(Path(call["path"]))}
                                         for call in cell["calls"]],
            "prompt": "", "max_tokens": MAX_TOKENS}


def _cell_row(arm: str, cell: dict[str, Any], texts: list[str], elapsed_ms: float,
              lc: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    references = [call["ref"] for call in cell["calls"]]
    row: dict[str, Any] = {"workload": WORKLOAD, "model": MODEL, "provider": arm, "arm": arm,
        "lc": lc, "cell": cell["id"], "n_inputs": len(texts),
        "words": sum(len(normalize_words(reference).split()) for reference in references),
        "ms": round(elapsed_ms, 3), "cell_wall_s": elapsed_ms / 1000,
        "pages_per_second": len(texts) / max(elapsed_ms / 1000, 1e-9),
        "quality": quality(texts, references), "ok": True}
    if extra:
        row.update(extra)
    return row


def _record_error(info: dict[str, Any], exc: Exception, stage: str, **identity: Any) -> dict[str, Any]:
    fields = exc.call_fields() if isinstance(exc, C.HttpResponseError) else {
        "ok": False, "error": public_error(exc)}
    refused = C.call_refused(fields)
    detail = {**identity, **fields, "stage": stage, "state": "refused" if refused else "failed"}
    info.setdefault("errors", []).append(detail)
    if not refused:
        info.setdefault("error", fields["error"])
    return detail


def _measure_cells(info: dict[str, Any], cells: list[dict[str, Any]], lc: int,
                   repetitions: int, prepare, extra) -> None:
    """Keep actual attempts and completed siblings even when a later cell fails."""
    for cell in cells:
        try:
            infer = prepare(cell)
        except Exception as exc:
            _record_error(info, exc, "prepare", cell=cell["id"])
            continue
        warm_state = "complete"
        try:
            _, elapsed_ms, _ = infer()
            info["warmups"].append({"cell": cell["id"], "state": "complete",
                "ms": round(elapsed_ms, 3), "pages": len(cell["calls"]), "measured": False})
        except Exception as exc:
            detail = _record_error(info, exc, "warmup", cell=cell["id"])
            info["warmups"].append({**detail, "measured": False})
            warm_state = detail["state"]
            if warm_state != "refused":
                continue
        for repetition in range(repetitions):
            try:
                texts, elapsed_ms, details = infer()
                if len(texts) != len(cell["calls"]):
                    raise ValueError("OCR page count differs from request")
                call = _cell_row(info["arm"], cell, texts, elapsed_ms, lc, {
                    **extra(cell), **details, "repetition": repetition,
                    "warmup_state": warm_state})
                row = {"call": call, "texts": texts, "refs": [item["ref"] for item in cell["calls"]]}
            except Exception as exc:
                detail = _record_error(info, exc, "measured", cell=cell["id"], repetition=repetition)
                row = {"call": {"workload": WORKLOAD, "model": MODEL, "provider": info["arm"],
                    "arm": info["arm"], "lc": lc, "n_inputs": len(cell["calls"]),
                    "input_ids": [item["id"] for item in cell["calls"]],
                    "warmup_state": warm_state, **detail}, "texts": [], "refs": []}
            info["rows"].append(row)
            if not row["call"]["ok"] and not C.call_refused(row["call"]):
                break


def _capture_stage(info: dict[str, Any], stage: str, action, key: str | None = None) -> None:
    try:
        value = action()
        if key is not None:
            info[key] = value
    except Exception as exc:
        _record_error(info, exc, stage)


def _antfly_read(port: int, body: dict[str, Any]) -> tuple[list[str], float, dict[str, Any]]:
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    started = time.perf_counter()
    try:
        connection.request("POST", "/ai/v1/read", body=encoded, headers={"content-type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        connection.close()
    if response.status != 200:
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            payload = None
        raise C.HttpResponseError(response.status, payload)
    payload = json.loads(raw) if raw else {}
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not all(isinstance(row, dict) and isinstance(row.get("text"), str) for row in data):
        raise RuntimeError("Antfly read returned a malformed successful response")
    return [row["text"] for row in data], elapsed_ms, {
        "response_bytes": len(raw), "usage": payload.get("usage"),
        "execution": payload.get("execution")}


def run_antfly(args: argparse.Namespace, cells: list[dict[str, Any]], out: Path, lc: int) -> dict[str, Any]:
    tag = f"lc{lc}-antfly"
    server = C.AntflyServer(args.antfly, out, tag, models_dir=C.MODELS_DIR,
                           extra_args=["--scratch-budget-mb", os.environ.get("TERMITE_SCRATCH_BUDGET_MB", "8192")])
    info: dict[str, Any] = {"provider": "antfly", "arm": "antfly", "model": MODEL,
                            "binary": C.file_identity(args.antfly), "host_before": C.host_snapshot(),
                            "precision": C.native_precision(args.antfly, MODEL),
                            "weight_backing": {"format": "safetensors", "source_dtype": "BF16",
                                               "sha256": SOURCE_WEIGHT_SHA256},
                            "decoding": {"max_tokens": MAX_TOKENS, "prompt_override": None,
                                         "policy": "runtime defaults; no additional sampling overrides requested",
                                         "source_defaults": "see executable-bound lifecycle precision, when available"},
                            "precision_environment": {
                                key: C.public_path(value) if "/" in value else value
                                for key, value in os.environ.items()
                                if key.startswith(("ANTFLY_CUDA_LIGHTON", "ANTFLY_CUDA_ENABLE_LIGHTON",
                                                   "ANTFLY_CUDA_DISABLE_LIGHTON", "ANTFLY_LIGHTONOCR3",
                                                   "TERMITE_METAL_ENABLE_LIGHTONOCR3", "TERMITE_METAL_DISABLE_LIGHTONOCR3"))}}
    info.update({"rows": [], "warmups": [], "repetitions": args.repetitions})
    try:
        info.update(server.start(timeout_s=300))

        def prepare(cell):
            body = antfly_body(cell)
            return lambda: _antfly_read(server.port, body)

        _measure_cells(info, cells, lc, args.repetitions, prepare, lambda cell: {
            "batch_semantics": f"one HTTP read request containing {len(cell['calls'])} image(s)",
            "timing_basis": "complete HTTP send through response-body read; client image/base64/JSON preparation excluded"})
    except Exception as exc:
        _record_error(info, exc, "lifecycle")
    finally:
        _capture_stage(info, "host_after", C.host_snapshot, "host_after")
        _capture_stage(info, "process_state",
                       lambda: server.proc.poll() if getattr(server, "proc", None) is not None else None,
                       "exit_code_before_teardown")
        _capture_stage(info, "cleanup", server.stop, "footprint")
        _capture_stage(info, "log_cleanup", lambda: sanitize_log(out / f"{tag}.server.log"))
    return info


def _model_dir() -> Path:
    owner, name = MODEL.split("/", 1)
    roots = [C.MODELS_DIR / owner / name, C.MODELS_DIR / "readers" / owner / name,
             C.MODELS_DIR / "generators" / owner / name]
    roots.extend(sorted((C.MODELS_DIR / owner).glob(f"{name}--antfly-*")))
    found = [path for path in roots if (path / "model.safetensors").is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one installed {MODEL} model, found {[C.public_path(p) for p in found]}")
    return found[0]


def verify_same_weights() -> dict[str, Any]:
    antfly = _model_dir()
    receipt = json.loads((antfly / ".antfly-download-complete.json").read_text(encoding="utf-8"))
    if (receipt.get("source") or {}).get("resolved_revision") != REVISION:
        raise ValueError("Antfly LightOnOCR source revision differs")
    reference = Path(C.reference_paths()[MODEL])
    antfly_hash = C.sha256_file(antfly / "model.safetensors")
    reference_hash = C.sha256_file(reference / "model.safetensors")
    if antfly_hash != SOURCE_WEIGHT_SHA256 or reference_hash != SOURCE_WEIGHT_SHA256:
        raise ValueError("LightOnOCR Transformers and Antfly weights are not identical")
    return {"model": MODEL, "revision": REVISION, "model_safetensors_sha256": SOURCE_WEIGHT_SHA256,
            "antfly_source_match": True, "transformers_source_match": True,
            "comparison": "SHA-256 of model.safetensors in both installed snapshots",
            "precision_note": ("Matched source weights are BF16. Source-derived native compute/cache facts "
                               "and observed reference parameter/runtime settings are in lifecycle precision.")}


def _conversation(image) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": image}]}]


def cache_summary(cache: Any) -> dict[str, Any]:
    """Inspect cache metadata only: never copy or retain tensor contents."""
    counts: dict[str, int] = {}
    seen: set[int] = set()
    tensors = elements = 0

    def visit(value: Any, depth: int = 0) -> None:
        nonlocal tensors, elements
        if value is None or id(value) in seen or depth > 5:
            return
        seen.add(id(value))
        if hasattr(value, "numel") and hasattr(value, "dtype"):
            count = int(value.numel())
            if count:
                tensors += 1
                elements += count
                key = str(value.dtype)
                counts[key] = counts.get(key, 0) + count
        elif isinstance(value, dict):
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, depth + 1)
        elif hasattr(value, "__dict__"):
            for child in vars(value).values():
                visit(child, depth + 1)

    visit(cache)
    length = int(cache.get_seq_length()) if cache is not None else 0
    return {"sequence_length": length, "populated_tensors": tensors,
            "tensor_elements": elements, "elements_by_dtype": counts}


def validate_cache_probe(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if len(steps) != 2:
        raise RuntimeError(f"OCR cache probe expected prefill and one decode, observed {len(steps)} forwards")
    prefill, decode = steps
    length = prefill["after"]["sequence_length"]
    if not (prefill["input_tokens"] > 1 and prefill["use_cache"] is True
            and prefill["before"]["sequence_length"] == 0
            and length == prefill["input_tokens"] and prefill["after"]["populated_tensors"] > 0
            and decode["input_tokens"] == 1 and decode["use_cache"] is True
            and decode["before"]["populated_tensors"] > 0
            and decode["before"]["sequence_length"] == length
            and decode["after"]["sequence_length"] == length + 1):
        raise RuntimeError("OCR generation did not demonstrate populated cache reuse for one-token decode")
    return {"validated": True, "measured": False, "forward_count": 2,
            "prefill": prefill, "decode": decode,
            "method": "real generate(max_new_tokens=2,min_new_tokens=2,use_cache=True), forward hooks"}


def probe_generation_cache(model: Any, inputs: Any) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []

    def before(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        token_ids = kwargs.get("input_ids")
        embeds = kwargs.get("inputs_embeds")
        width = token_ids.shape[1] if token_ids is not None else embeds.shape[1]
        steps.append({"input_tokens": int(width), "use_cache": kwargs.get("use_cache"),
                      "before": cache_summary(kwargs.get("past_key_values"))})

    def after(_module: Any, _args: Any, _kwargs: Any, result: Any) -> None:
        steps[-1]["after"] = cache_summary(result.past_key_values)

    pre = model.register_forward_pre_hook(before, with_kwargs=True)
    post = model.register_forward_hook(after, with_kwargs=True)
    try:
        model.generate(**inputs, max_new_tokens=2, min_new_tokens=2, do_sample=False,
                       num_beams=1, use_cache=True)
    finally:
        pre.remove()
        post.remove()
    return validate_cache_probe(steps)


def transformers_arm(only: str | None, out: Path, lc: int, repetitions: int = 10,
                     *, cells: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    cells = make_cells(only, out, reuse_existing=True) if cells is None else cells
    reference = Path(C.reference_paths()[MODEL])
    device = C.ACCEL
    processor = AutoProcessor.from_pretrained(reference, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        reference, local_files_only=True).to(device).eval()

    cache_probe: dict[str, Any] = {}

    def infer(calls: list[dict[str, Any]], max_tokens: int = MAX_TOKENS, *, probe: bool = False) -> list[str]:
        images = []
        try:
            for call in calls:
                with Image.open(call["path"]) as source:
                    images.append(source.convert("RGB"))
            conversations = [_conversation(image) for image in images]
            value: Any = conversations if len(conversations) > 1 else conversations[0]
            inputs = processor.apply_chat_template(value, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=len(conversations) > 1,
                enable_thinking=False).to(device)
            with torch.inference_mode():
                if probe:
                    cache_probe.update(probe_generation_cache(model, inputs))
                    return []
                generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                           num_beams=1, use_cache=True)
            prompt_width = inputs["input_ids"].shape[1]
            return processor.batch_decode(generated[:, prompt_width:], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        finally:
            for image in images:
                image.close()

    info = {"provider": "transformers", "arm": "transformers", "model": MODEL,
            "requested_dtype": None, "device": device, "rows": [],
            "cache_probe": cache_probe, "warmups": [], "repetitions": repetitions,
            "api": "AutoProcessor.apply_chat_template + Qwen3_5ForConditionalGeneration.generate",
            "decoding": {"prompt": "", "enable_thinking": False, "max_new_tokens": MAX_TOKENS,
                         "do_sample": False, "num_beams": 1, "padding_side": "left", "use_cache": True}}
    try:
        infer(cells[0]["calls"][:1], max_tokens=1, probe=True)
        sync = C.torch_sync(torch)

        def prepare(cell):
            def measured():
                sync()
                started = time.perf_counter()
                texts = infer(cell["calls"])
                sync()
                return texts, (time.perf_counter() - started) * 1000, {}
            return measured

        _measure_cells(info, cells, lc, repetitions, prepare, lambda cell: {
            "batch_semantics": f"one batched processor/generate call containing {len(cell['calls'])} page(s)",
            "timing_basis": "processor image preparation, model inference, and decode"})
    except Exception as exc:
        _record_error(info, exc, "cache_probe" if not cache_probe else "measurement")
    _capture_stage(info, "precision", lambda: C.reference_precision(
        model, default_policy="library_default"), "precision")
    _capture_stage(info, "versions", lambda: {
        package: importlib.metadata.version(package) for package in ("torch", "transformers", "pillow")}, "versions")
    _capture_stage(info, "generation_config", lambda: info["decoding"].update({
        "generation_config": {key: getattr(model.generation_config, key, None)
            for key in ("eos_token_id", "pad_token_id", "bos_token_id", "repetition_penalty",
                        "no_repeat_ngram_size", "length_penalty", "forced_bos_token_id",
                        "forced_eos_token_id", "suppress_tokens", "begin_suppress_tokens")}}))
    _capture_stage(info, "device_memory", lambda: C.torch_device_bytes(torch), "device_memory")
    return info


def pyarm_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["auto", "cuda", "metal"])
    parser.add_argument("--only-cell", choices=CELLS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lc", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args(argv)
    C.configure_backend(args.backend)
    try:
        if args.repetitions < 1:
            raise ValueError("--repetitions must be positive")
        result = transformers_arm(args.only_cell, args.out, args.lc, args.repetitions)
    except Exception as exc:
        result = {"provider": "transformers", "arm": "transformers", "model": MODEL,
                  "error": public_error(exc)}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if "error" not in result else 1


def agreement_rows(lc: int, cells: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for cell in cells:
        repetitions = sorted({row["call"].get("repetition", 0)
                              for result in results.values() for row in result.get("rows", [])
                              if row["call"]["cell"] == cell["id"]}) or [0]
        for repetition in repetitions:
            by_arm: dict[str, list[str]] = {}
            for arm in PROVIDERS:
                match = next((row for row in (results.get(arm) or {}).get("rows", [])
                              if row["call"]["cell"] == cell["id"]
                              and row["call"].get("repetition", 0) == repetition
                              and row["call"].get("ok", True)), None)
                if match is not None:
                    by_arm[arm] = match["texts"]
            row: dict[str, Any] = {"workload": WORKLOAD, "model": MODEL, "lc": lc,
                                   "cell": cell["id"], "repetition": repetition, "providers": list(by_arm),
                                   "complete": len(by_arm) == len(PROVIDERS)}
            if row["complete"]:
                left = [normalize_characters(text) for text in by_arm["antfly"]]
                right = [normalize_characters(text) for text in by_arm["transformers"]]
                if len(left) != len(right):
                    raise ValueError("paired OCR output counts differ")
                exact = [a == b for a, b in zip(left, right, strict=True)]
                edits = sum(edit_distance(a, b) for a, b in zip(left, right, strict=True))
                chars = sum(len(value) for value in right)
                row.update({"normalized_exact_count": sum(exact), "items": len(exact),
                            "all_normalized_exact": all(exact), "character_edits": edits,
                            "character_error_rate": edits / max(1, chars),
                            "character_agreement": max(0.0, 1.0 - edits / max(1, chars))})
            output.append(row)
    return output


def _write_arm(out: Path, result: dict[str, Any], lc: int) -> None:
    calls = C.JsonlWriter(out / "calls.jsonl")
    outputs = C.JsonlWriter(out / "outputs.jsonl")
    try:
        for row in result.get("rows", []):
            calls.write(row["call"])
            for index, text in enumerate(row["texts"]):
                outputs.write({"workload": WORKLOAD, "model": MODEL, "provider": result["provider"],
                               "arm": result["arm"], "lc": lc, "cell": row["call"]["cell"],
                               "repetition": row["call"].get("repetition", 0), "k": index,
                               "text": text, "ref": row["refs"][index]})
    finally:
        calls.close()
        outputs.close()


def orchestrate(args: argparse.Namespace) -> int:
    C.configure_backend(args.backend)
    C.exclusive_bench_lock()
    out = Path(args.out)
    C.prepare_output(out)
    cells = make_cells(args.only_cell, out)
    C.write_json(out / "model_files.json", verify_same_weights())
    C.write_json(out / "inputs.json", {"source": "deterministic public synthetic OCR pages",
        "rendering": {"width": 1024, "height": 1344, "font": "Pillow bundled default, 36px"},
        "cells": [{"id": cell["id"], "items": [{"id": call["id"], "sha256": call["sha256"],
                   "width": call["width"], "height": call["height"], "text_bbox": call["text_bbox"],
                   "rendered_lines": call["rendered_lines"]} for call in cell["calls"]]}
                  for cell in cells]})
    lifecycle_writer = C.JsonlWriter(out / "lifecycles.jsonl")
    agreement_writer = C.JsonlWriter(out / "agreement.jsonl")
    failed = False
    try:
        for lc in range(args.lifecycles):
            ordered = PROVIDERS if lc % 2 == 0 else PROVIDERS[::-1]
            results: dict[str, dict[str, Any]] = {}
            for arm in ordered:
                started = time.time()
                try:
                    if arm == "antfly":
                        result = run_antfly(args, cells, out, lc)
                    else:
                        child_args = ["--backend", args.backend, "--out", str(out), "--lc", str(lc),
                                      "--repetitions", str(args.repetitions)]
                        if args.only_cell:
                            child_args.extend(["--only-cell", args.only_cell])
                        result = C.run_python_arm(str(Path(__file__)), child_args, out, f"lc{lc}-transformers")
                        _capture_stage(result, "log_cleanup", lambda: sanitize_log(out / f"lc{lc}-transformers.py.log"))
                except Exception as exc:
                    failed = True
                    result = {"provider": arm, "arm": arm, "model": MODEL, "error": public_error(exc)}
                if result.get("rc", 0) != 0 or "error" in result:
                    failed = True
                _write_arm(out, result, lc)
                results[arm] = result
                lifecycle = {key: value for key, value in result.items() if key != "rows"}
                lifecycle.update({"workload": WORKLOAD, "model": MODEL, "provider": arm, "lc": lc,
                                  "status": "failed" if "error" in result or result.get("rc", 0) else "ok",
                                  "wall_s": round(time.time() - started, 3)})
                lifecycle_writer.write(lifecycle)
            for row in agreement_rows(lc, cells, results):
                agreement_writer.write(row)
    finally:
        lifecycle_writer.close()
        agreement_writer.close()
    complete = C.benchmark_complete(out, set(PROVIDERS), args.lifecycles, len(cells) * args.repetitions)
    return 1 if failed or not complete else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "pyarm":
        return pyarm_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--antfly", type=Path, required=True)
    run.add_argument("--backend", choices=["auto", "cuda", "metal"], default="auto")
    run.add_argument("--lifecycles", type=int, default=3)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--only-cell", choices=CELLS)
    run.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args(argv)
    if args.lifecycles < 1 or args.repetitions < 1:
        parser.error("--lifecycles and --repetitions must be positive")
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
