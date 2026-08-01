#!/usr/bin/env python3
"""Generate deterministic local media fixtures without adding binary files to git."""

from __future__ import annotations

import argparse
import binascii
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path


FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    " ": ("00000",) * 7,
}


def png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)


def write_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    rows = b"".join(b"\x00" + pixels[y * width : (y + 1) * width] for y in range(height))
    raw = b"\x89PNG\r\n\x1a\n"
    raw += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    raw += png_chunk(b"IDAT", zlib.compress(rows, 9))
    raw += png_chunk(b"IEND", b"")
    path.write_bytes(raw)


def render_text(path: Path, text: str, scale: int = 8) -> None:
    glyph_width = 6
    width = (len(text) * glyph_width + 2) * scale
    height = 11 * scale
    pixels = bytearray([255] * (width * height))
    for index, char in enumerate(text.upper()):
        glyph = FONT[char]
        x0 = (index * glyph_width + 1) * scale
        y0 = 2 * scale
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit != "1":
                    continue
                for dy in range(scale):
                    for dx in range(scale):
                        x = x0 + gx * scale + dx
                        y = y0 + gy * scale + dy
                        pixels[y * width + x] = 0
    write_png(path, width, height, bytes(pixels))


def render_solid_square(path: Path, value: int) -> None:
    write_png(path, 64, 64, bytes([value] * (64 * 64)))


def render_audio(path: Path, text: str) -> bool:
    say = shutil.which("say")
    afconvert = shutil.which("afconvert")
    if not say or not afconvert:
        return False
    aiff = path.with_suffix(".aiff")
    subprocess.run([say, "-v", "Samantha", "-r", "165", "-o", str(aiff), text], check=True)
    try:
        subprocess.run([afconvert, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(path)], check=True)
    finally:
        aiff.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/fixtures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    render_text(out / "ocr.png", "ANTFLY MOTHS")
    render_solid_square(out / "black-square.png", 0)
    render_solid_square(out / "white-square.png", 255)
    if not render_audio(out / "transcription.wav", "Antfly moths measure local inference"):
        print("warning: say/afconvert unavailable; transcription fixture was not generated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
