#!/usr/bin/env python3
"""Discover, fetch, and verify deterministic image and audio benchmark fixtures."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MANIFEST = C.ROOT / "media_manifest.json"
USER_AGENT = "termite-circus-benchmark/1.0 (inference benchmark media fixture)"
COMMONS = "https://commons.wikimedia.org/w/api.php"
WIDTH = 1024
LIBRISPEECH_REPO = "hf-internal-testing/librispeech_asr_dummy"
LIBRISPEECH_REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
LIBRISPEECH_FILE = "clean/validation-00000-of-00001.parquet"


def get(url: str) -> bytes:
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def commons(parameters: dict) -> dict:
    query = urllib.parse.urlencode({**parameters, "format": "json", "formatversion": 2})
    return json.loads(get(COMMONS + "?" + query))


def thumbnail_url(title: str) -> str:
    pages = commons(
        {"action": "query", "titles": title, "prop": "imageinfo", "iiprop": "url", "iiurlwidth": WIDTH}
    )["query"]["pages"]
    return pages[0]["imageinfo"][0]["thumburl"]


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def librispeech_rows() -> list[dict]:
    import pyarrow.parquet as parquet

    url = f"https://huggingface.co/datasets/{LIBRISPEECH_REPO}/resolve/{LIBRISPEECH_REVISION}/{LIBRISPEECH_FILE}"
    return parquet.read_table(io.BytesIO(get(url))).to_pylist()


def to_wav(flac: bytes) -> tuple[bytes, float]:
    import numpy as np
    import soundfile as soundfile

    samples, sample_rate = soundfile.read(io.BytesIO(flac), dtype="int16")
    if sample_rate != 16000 or samples.ndim != 1:
        raise ValueError(f"unexpected audio sample rate={sample_rate} shape={samples.shape}")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.asarray(samples).tobytes())
    return buffer.getvalue(), len(samples) / sample_rate


def discover() -> None:
    members = commons(
        {
            "action": "query",
            "list": "categorymembers",
            "cmtype": "file",
            "cmlimit": 40,
            "cmtitle": "Category:Featured pictures on Wikimedia Commons",
            "cmsort": "timestamp",
            "cmstart": "2021-01-01T00:00:00Z",
        }
    )["query"]["categorymembers"]
    images = []
    for member in members:
        title = member["title"]
        if not title.lower().endswith((".jpg", ".jpeg")):
            continue
        content = get(thumbnail_url(title))
        images.append(
            {"name": f"img{len(images):02d}.jpg", "title": title, "bytes": len(content), "sha256": sha256(content)}
        )
        if len(images) == 8:
            break
    audio = []
    for row in librispeech_rows():
        content, seconds = to_wav(row["audio"]["bytes"])
        if 5 <= seconds <= 15:
            audio.append(
                {
                    "name": f"aud{len(audio):02d}.wav",
                    "id": row["id"],
                    "seconds": round(seconds, 2),
                    "text": row["text"],
                    "bytes": len(content),
                    "sha256": sha256(content),
                }
            )
        if len(audio) == 8:
            break
    manifest = {
        "width": WIDTH,
        "librispeech": {
            "repo": LIBRISPEECH_REPO,
            "revision": LIBRISPEECH_REVISION,
            "file": LIBRISPEECH_FILE,
        },
        "images": images,
        "audio": audio,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.name}: {len(images)} images, {len(audio)} clips")


def fetch() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    C.MEDIA.mkdir(parents=True, exist_ok=True)
    mismatches = []

    def have(entry: dict) -> bool:
        path = C.MEDIA / entry["name"]
        return path.exists() and sha256(path.read_bytes()) == entry["sha256"]

    def keep(entry: dict, content: bytes) -> None:
        if sha256(content) != entry["sha256"]:
            mismatches.append(entry["name"])
            return
        (C.MEDIA / entry["name"]).write_bytes(content)

    for image in manifest["images"]:
        if not have(image):
            keep(image, get(thumbnail_url(image["title"])))
    needed = [item for item in manifest["audio"] if not have(item)]
    if needed:
        rows = {row["id"]: row for row in librispeech_rows()}
        for audio in needed:
            keep(audio, to_wav(rows[audio["id"]]["audio"]["bytes"])[0])
    if mismatches:
        raise RuntimeError(f"SHA-256 mismatch: {mismatches}")
    print(f"{len(manifest['images'])} images and {len(manifest['audio'])} clips verified in {C.public_path(C.MEDIA)}")


def verify() -> bool:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = [*manifest["images"], *manifest["audio"]]
    failures = []
    for entry in entries:
        path = C.MEDIA / entry["name"]
        if not path.is_file() or sha256(path.read_bytes()) != entry["sha256"]:
            failures.append(entry["name"])
    print(json.dumps({"status": "complete" if not failures else "failed", "missing_or_mismatched": failures}, separators=(",", ":")))
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("discover", "fetch", "verify"))
    args = parser.parse_args()
    if args.command == "discover":
        discover()
        return 0
    if args.command == "fetch":
        fetch()
        return 0
    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
