#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BACKEND=auto
CONVERT_WHISPER=0

usage() {
  cat <<'EOF'
usage: scripts/setup_references.sh [--backend auto|metal|cuda] [--convert-whisper]

Creates the locked Python reference environment and builds pinned llama.cpp and
whisper.cpp binaries. --convert-whisper explicitly downloads the pinned
openai/whisper-small snapshot and converts it to whisper.cpp's F16 GGML format.
EOF
}

while (($#)); do
  case "$1" in
    --backend)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      BACKEND=$2
      shift 2
      ;;
    --convert-whisper)
      CONVERT_WHISPER=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$BACKEND" in
  auto)
    case "$(uname -s)" in
      Darwin) BACKEND=metal ;;
      Linux) BACKEND=cuda ;;
      *) printf 'auto backend supports only macOS Metal and Linux CUDA\n' >&2; exit 2 ;;
    esac
    ;;
  metal|cuda) ;;
  *) printf 'backend must be auto, metal, or cuda\n' >&2; exit 2 ;;
esac

if [[ "$BACKEND" == metal && "$(uname -s)" != Darwin ]]; then
  printf 'the Metal reference build requires macOS\n' >&2
  exit 2
fi
if [[ "$BACKEND" == cuda && "$(uname -s)" != Linux ]]; then
  printf 'the CUDA reference build requires Linux\n' >&2
  exit 2
fi

for command in uv git cmake; do
  command -v "$command" >/dev/null || { printf 'missing required command: %s\n' "$command" >&2; exit 2; }
done

cd "$ROOT"
if [[ "$BACKEND" == cuda ]]; then
  uv sync --frozen --python 3.12 --extra cuda
else
  uv sync --frozen --python 3.12
fi

PY=${TERMITE_REFERENCE_PYTHON:-$ROOT/.venv/bin/python}
[[ -x "$PY" ]] || { printf 'reference Python is not executable: %s\n' "$PY" >&2; exit 2; }

NATIVE_ROOT=${TERMITE_NATIVE_ROOT:-$ROOT/.cache/native}
BIN_DIR=$ROOT/.cache/bin
LLAMA_DIR=$NATIVE_ROOT/llama.cpp
WHISPER_DIR=$NATIVE_ROOT/whisper.cpp
# llama.cpp b10809
LLAMA_COMMIT=5266f24da75dc449bd56cbed7addb9c8e4a6a73e
# whisper.cpp v1.9.2
WHISPER_COMMIT=306c88f4d1286aec1bf96e544632897886af5501
# openai/whisper v20250625; whisper.cpp uses its mel filter assets.
OPENAI_WHISPER_COMMIT=31243bad24cc746f07d4c8bfdd2d974872cb1803
WHISPER_HF_REVISION=973afd24965f72e36ca33b3055d56a652f456b4d
WHISPER_SOURCE_SHA256=1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b

checkout_pinned() {
  local url=$1 commit=$2 destination=$3
  if [[ ! -d "$destination/.git" ]]; then
    mkdir -p "$destination"
    git -C "$destination" init -q
    git -C "$destination" remote add origin "$url"
  fi
  if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]]; then
    printf 'refusing to replace modified generated checkout: %s\n' "$destination" >&2
    exit 2
  fi
  git -C "$destination" fetch -q --depth 1 origin "$commit"
  git -C "$destination" checkout -q --detach FETCH_HEAD
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]] || {
    printf 'checkout did not resolve to pinned commit %s\n' "$commit" >&2
    exit 2
  }
}

checkout_pinned https://github.com/ggml-org/llama.cpp.git "$LLAMA_COMMIT" "$LLAMA_DIR"
checkout_pinned https://github.com/ggml-org/whisper.cpp.git "$WHISPER_COMMIT" "$WHISPER_DIR"

CUDA_ROOT=
CUDA_LIB=
CMAKE_ACCEL=()
if [[ "$BACKEND" == cuda ]]; then
  CUDA_ROOT=${CUDAToolkit_ROOT:-$("$PY" - <<'PY'
from pathlib import Path
import site

for base in map(Path, site.getsitepackages()):
    root = base / "nvidia" / "cu13"
    if (root / "bin" / "nvcc").is_file():
        print(root)
        break
else:
    raise SystemExit("CUDA 13.2 toolkit installed by the cuda extra was not found")
PY
  )}
  [[ -x "$CUDA_ROOT/bin/nvcc" ]] || { printf 'nvcc not found under CUDAToolkit_ROOT=%s\n' "$CUDA_ROOT" >&2; exit 2; }
  for candidate in "$CUDA_ROOT/lib" "$CUDA_ROOT/lib64"; do
    if compgen -G "$candidate/libcublas.so*" >/dev/null; then
      CUDA_LIB=$candidate
      break
    fi
  done
  [[ -n "$CUDA_LIB" ]] || {
    printf 'cuBLAS not found under the CUDA 13.2 toolkit root %s\n' "$CUDA_ROOT" >&2
    exit 2
  }
  export PATH="$CUDA_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  CMAKE_ACCEL=(
    -DGGML_CUDA=ON
    -DGGML_METAL=OFF
    "-DCUDAToolkit_ROOT=$CUDA_ROOT"
    "-DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc"
    "-DCMAKE_CUDA_ARCHITECTURES=${TERMITE_CUDA_ARCHITECTURES:-native}"
    "-DCMAKE_BUILD_RPATH=$CUDA_LIB"
  )
else
  CMAKE_ACCEL=(-DGGML_CUDA=OFF -DGGML_METAL=ON)
fi

BUILD_JOBS=${TERMITE_BUILD_JOBS:-2}
LLAMA_BUILD=$LLAMA_DIR/build-$BACKEND
WHISPER_BUILD=$WHISPER_DIR/build-$BACKEND

cmake -S "$LLAMA_DIR" -B "$LLAMA_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  "${CMAKE_ACCEL[@]}"
cmake --build "$LLAMA_BUILD" --config Release --target llama-server --parallel "$BUILD_JOBS"

cmake -S "$WHISPER_DIR" -B "$WHISPER_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
  "${CMAKE_ACCEL[@]}"
cmake --build "$WHISPER_BUILD" --config Release --target whisper-cli --parallel "$BUILD_JOBS"

mkdir -p "$BIN_DIR"
ln -sfn "$LLAMA_BUILD/bin/llama-server" "$BIN_DIR/llama-server"
ln -sfn "$WHISPER_BUILD/bin/whisper-cli" "$BIN_DIR/whisper-cli"

if ((CONVERT_WHISPER)); then
  OPENAI_WHISPER_DIR=$NATIVE_ROOT/openai-whisper
  checkout_pinned https://github.com/openai/whisper.git "$OPENAI_WHISPER_COMMIT" "$OPENAI_WHISPER_DIR"

  DATA=${TERMITE_DATA:-$ROOT/.data}
  HF_HOME=${TERMITE_HF_HOME:-$DATA/huggingface}
  HF_MODEL_DIR=$DATA/references/openai-whisper-small-$WHISPER_HF_REVISION
  CONVERTED_DIR=$DATA/references/whisper-small-whispercpp-v1.9.2
  CONVERTED_MODEL=${TERMITE_WHISPER_CPP_MODEL:-$CONVERTED_DIR/ggml-model.bin}
  PROVENANCE=${TERMITE_WHISPER_CPP_PROVENANCE:-$CONVERTED_MODEL.provenance.json}
  mkdir -p "$HF_HOME" "$HF_MODEL_DIR" "$CONVERTED_DIR" "$(dirname "$CONVERTED_MODEL")" "$(dirname "$PROVENANCE")"

  if "$PY" - "$CONVERTED_MODEL" "$PROVENANCE" "$WHISPER_COMMIT" "$WHISPER_HF_REVISION" "$WHISPER_SOURCE_SHA256" >/dev/null 2>&1 <<'PY'
from pathlib import Path
import hashlib
import json
import sys

model, provenance = map(Path, sys.argv[1:3])
converter_revision, source_revision, source_sha = sys.argv[3:]
record = json.loads(provenance.read_text())
expected = {
    "schema_version": 1,
    "model": "openai/whisper-small",
    "source_revision": source_revision,
    "source_model_sha256": source_sha,
    "converter": {"name": "whisper.cpp", "revision": converter_revision},
}
if set(record) != set(expected) | {"converted_model_sha256"}:
    raise SystemExit(1)
for key, value in expected.items():
    if record.get(key) != value:
        raise SystemExit(1)
h = hashlib.sha256()
with model.open("rb") as source:
    for block in iter(lambda: source.read(16 << 20), b""):
        h.update(block)
if record["converted_model_sha256"] != h.hexdigest():
    raise SystemExit(1)
PY
  then
    printf 'reusing verified whisper.cpp conversion: %s\n' "$CONVERTED_MODEL"
  else
  HF_HOME="$HF_HOME" "$PY" - "$HF_MODEL_DIR" "$WHISPER_HF_REVISION" <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

local_dir, revision = sys.argv[1:]
snapshot_download(
    "openai/whisper-small",
    revision=revision,
    local_dir=Path(local_dir),
    allow_patterns=[
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "normalizer.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
)
PY

  "$PY" - "$HF_MODEL_DIR/model.safetensors" "$WHISPER_SOURCE_SHA256" <<'PY'
from pathlib import Path
import hashlib
import sys

path, expected = Path(sys.argv[1]), sys.argv[2]
h = hashlib.sha256()
with path.open("rb") as source:
    for block in iter(lambda: source.read(16 << 20), b""):
        h.update(block)
actual = h.hexdigest()
if actual != expected:
    raise SystemExit(f"{path.name}: sha256 {actual} != {expected}")
PY

  TEMP_DIR=$CONVERTED_DIR/.convert-$$
  rm -rf "$TEMP_DIR"
  mkdir -p "$TEMP_DIR"
  trap 'rm -rf "$TEMP_DIR"' EXIT
  "$PY" "$WHISPER_DIR/models/convert-h5-to-ggml.py" "$HF_MODEL_DIR" "$OPENAI_WHISPER_DIR" "$TEMP_DIR"
  mv "$TEMP_DIR/ggml-model.bin" "$CONVERTED_MODEL"
  rm -rf "$TEMP_DIR"
  trap - EXIT

  "$PY" - "$CONVERTED_MODEL" "$PROVENANCE" "$WHISPER_COMMIT" "$WHISPER_HF_REVISION" "$WHISPER_SOURCE_SHA256" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

model, provenance = map(Path, sys.argv[1:3])
converter_revision, source_revision, source_sha = sys.argv[3:]
h = hashlib.sha256()
with model.open("rb") as source:
    for block in iter(lambda: source.read(16 << 20), b""):
        h.update(block)
record = {
    "schema_version": 1,
    "model": "openai/whisper-small",
    "source_revision": source_revision,
    "source_model_sha256": source_sha,
    "converted_model_sha256": h.hexdigest(),
    "converter": {"name": "whisper.cpp", "revision": converter_revision},
}
provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
  fi
fi

printf 'reference Python: %s\n' "$PY"
printf 'llama-server: %s\n' "$BIN_DIR/llama-server"
printf 'whisper-cli: %s\n' "$BIN_DIR/whisper-cli"
if ((CONVERT_WHISPER)); then
  printf 'whisper.cpp model: %s\n' "$CONVERTED_MODEL"
  printf 'whisper.cpp provenance: %s\n' "$PROVENANCE"
fi
