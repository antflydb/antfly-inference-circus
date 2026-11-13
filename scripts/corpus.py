#!/usr/bin/env python3
"""Build a repeatable text corpus from retained, recorded source snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

USER_AGENT = "termite-circus-benchmark/1.0 (inference benchmark corpus builder)"
SEED = 20261103
DARWIN_URL = "https://www.gutenberg.org/cache/epub/1228/pg1228.txt"
WIKIPEDIA = {
    "en": [
        "Charles Darwin",
        "Apple Inc.",
        "European Union",
        "2008 financial crisis",
        "Apollo 11",
        "World Health Organization",
        "Taylor Swift",
        "Microsoft",
        "Tokyo",
        "Nelson Mandela",
        "Amazon rainforest",
        "Premier League",
    ],
}


def fetch_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def fetch_wikipedia(raw_dir: Path, language: str, title: str) -> tuple[str, dict]:
    stem = f"wiki-{language}-{hashlib.sha1(title.encode()).hexdigest()[:10]}"
    path = raw_dir / f"{stem}.txt"
    metadata_path = raw_dir / f"{stem}.json"
    if path.exists() and metadata_path.exists():
        text = path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("extract_sha256") != hashlib.sha256(text.encode()).hexdigest():
            raise RuntimeError(f"retained Wikipedia snapshot checksum mismatch for {language}:{title}")
        if (
            not isinstance(metadata.get("resolved_title"), str)
            or not isinstance(metadata.get("pageid"), int)
            or not isinstance(metadata.get("lastrevid"), int)
            or not isinstance(metadata.get("url"), str)
        ):
            raise RuntimeError(f"retained Wikipedia snapshot provenance missing for {language}:{title}")
        return text, metadata
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts|info",
            "inprop": "url",
            "explaintext": 1,
            "format": "json",
            "redirects": 1,
            "titles": title,
        }
    )
    request = urllib.request.Request(
        f"https://{language}.wikipedia.org/w/api.php?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(5):
        time.sleep(3 + 10 * attempt)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                pages = json.load(response)["query"]["pages"]
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
    page = next(iter(pages.values()))
    text = page.get("extract", "")
    if not text or page.get("missing"):
        raise RuntimeError(f"Wikipedia returned no extract for {language}:{title}")
    metadata = {
        "language": language,
        "requested_title": title,
        "resolved_title": page.get("title"),
        "pageid": page.get("pageid"),
        "lastrevid": page.get("lastrevid"),
        "url": page.get("fullurl"),
        "extract_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    if (
        not isinstance(metadata["resolved_title"], str)
        or not isinstance(metadata["pageid"], int)
        or not isinstance(metadata["lastrevid"], int)
        or not isinstance(metadata["url"], str)
    ):
        raise RuntimeError(f"Wikipedia returned incomplete provenance for {language}:{title}")
    path.write_text(text, encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return text, metadata


def word_count(text: str) -> int:
    return len(text.split())




def darwin_sections(text: str) -> list[dict]:
    start = text.index("*** START OF THE PROJECT GUTENBERG")
    end = text.index("*** END OF THE PROJECT GUTENBERG")
    body = text[start:end]
    introductions = [match.start() for match in re.finditer(r"^INTRODUCTION\.\s*$", body, re.M)]
    if len(introductions) < 2:
        raise RuntimeError("the Gutenberg input does not have the expected introduction markers")
    index_match = re.search(r"^INDEX\.\s*$", body, re.M)
    if index_match is None:
        raise RuntimeError("the Gutenberg input does not have the expected index marker")
    body = body[introductions[1] : index_match.start()]
    headings = list(re.finditer(r"^(INTRODUCTION\.|CHAPTER [IVXL]+\.)\s*$", body, re.M))
    sections = []
    for index, heading in enumerate(headings):
        next_start = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        segment = re.sub(r"\r", "", body[heading.start() : next_start]).strip()
        sections.append({"id": heading.group(1).rstrip(".").replace(" ", "_").lower(), "text": segment})
    return sections


def paragraphs_of(text: str) -> list[str]:
    paragraphs = [re.sub(r"\s+", " ", paragraph).strip() for paragraph in re.split(r"\n\s*\n", text)]
    return [paragraph for paragraph in paragraphs if paragraph]


def sentences_of(paragraphs: list[str]) -> list[str]:
    sentences = []
    for paragraph in paragraphs:
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph):
            if 8 <= word_count(sentence) <= 60:
                sentences.append(sentence)
    return sentences


def pages_of(paragraphs: list[str], target: int = 350) -> list[str]:
    tokens = " ".join(paragraphs).split()
    return [" ".join(tokens[index : index + target]) for index in range(0, len(tokens) - target, target)]


def build(output: Path, raw_dir: Path) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    darwin_path = raw_dir / "origin-1859-pg1228.txt"
    if not darwin_path.exists():
        darwin_path.write_text(fetch_url(DARWIN_URL), encoding="utf-8")
    darwin_bytes = darwin_path.read_bytes()
    darwin = darwin_bytes.decode("utf-8")
    sections = darwin_sections(darwin)
    all_paragraphs = [paragraph for section in sections for paragraph in paragraphs_of(section["text"])[1:]]
    randomizer = random.Random(SEED)
    paragraphs = [paragraph for paragraph in all_paragraphs if 60 <= word_count(paragraph) <= 250]
    sentences = sentences_of(all_paragraphs)
    pages = pages_of(all_paragraphs)
    wiki_en = []
    wikipedia_sources = {"en": []}
    for title in WIKIPEDIA["en"]:
        text, source = fetch_wikipedia(raw_dir, "en", title)
        wikipedia_sources["en"].append(source)
        candidates = [paragraph for paragraph in paragraphs_of(text) if not paragraph.startswith("=")]
        candidates = [paragraph for paragraph in candidates if 60 <= word_count(paragraph) <= 250]
        wiki_en.extend(
            {"title": source["resolved_title"], "text": paragraph}
            for paragraph in candidates
        )
    for values in (sentences, paragraphs, pages, wiki_en):
        randomizer.shuffle(values)
    corpus = {
        "seed": SEED,
        "sampling": "Deterministic seed and slicing over retained source snapshots. Upstream URLs and Wikipedia titles are mutable; rebuilding without the retained raw snapshots may differ.",
        "sources": {
            "darwin": {"file": darwin_path.name, "sha256": hashlib.sha256(darwin_bytes).hexdigest(), "url": DARWIN_URL},
            "wikipedia": wikipedia_sources,
        },
        "sentences": sentences[:256],
        "paragraphs": paragraphs[:256],
        "pages": pages[:128],
        "chapters": [
            {
                "id": section["id"],
                "text": section["text"],
                "words": word_count(section["text"]),
                "bytes": len(section["text"].encode()),
            }
            for section in sections
        ],
        "wiki_en": [item["text"] for item in wiki_en[:256]],
    }
    minimums = {"sentences": 256, "paragraphs": 256, "pages": 128, "wiki_en": 64}
    short = {key: (len(corpus[key]), minimum) for key, minimum in minimums.items() if len(corpus[key]) < minimum}
    if short:
        raise RuntimeError(f"corpus is too short for benchmark cells: {short}")
    if len(corpus["chapters"]) != 15:
        raise RuntimeError(f"expected 15 book sections, found {len(corpus['chapters'])}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(corpus, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {
        **{key: len(value) for key, value in corpus.items() if isinstance(value, list)},
        "book_words": sum(chapter["words"] for chapter in corpus["chapters"]),
        "sha256": C.sha256_file(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.CORPUS)
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    output = args.output.expanduser()
    raw_dir = args.raw_dir.expanduser() if args.raw_dir else output.parent / "raw"
    print(json.dumps(build(output, raw_dir), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
