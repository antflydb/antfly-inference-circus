# termite-circus

`termite-circus` is a standalone, manual benchmark harness for the current
Antfly inference surface. It measures one operator-supplied Antfly executable
against local reference implementations on either Metal or CUDA. It does not
discover machines, schedule work, publish results, or depend on a larger
benchmark service.

Normal benchmark runs are offline and use loopback networking only. Downloads
and native builds happen only through the explicit `make setup` path.

## Workloads

The suite intentionally contains six logical models:

| Workload | Model | Paired reference |
| --- | --- | --- |
| embedding | `google/embeddinggemma-2` | Sentence Transformers 6.2.0 |
| schema extraction | `fastino/gliner2.5-base-v1` | GLiNER2 2.1.0 |
| typed decisions | `fastino/GLiNER2.5-Decide` | GLiNER2 2.1.0 |
| multimodal generation | `google/gemma-4-E4B-it-qat-q4_0-gguf` | llama.cpp `b10809` |
| speech transcription | `openai/whisper-small` | Transformers 5.22.2 and whisper.cpp 1.9.2 |
| document OCR | `lightonai/LightOnOCR-3-0.8B` | Transformers 5.22.2 |

Each lifecycle starts fresh provider processes. Results retain cell-level
latency, throughput, quality/agreement, memory, errors, and artifact
provenance. A provider error remains a failed result; it is not replaced with a
successful fallback.

A measured sample is one inference API operation, including a batch submitted
in one operation. The suite excludes document-window scans with merged results,
multi-window audio, and workloads assembled from concurrent independent
requests. Warmups and independent measured repetitions remain separate samples,
not a composite workload.

Batch workloads remain batch-API comparisons: native runtimes may microbatch
internally, so one API operation does not establish equal physical-forward
counts. Autoregressive decoding and image patch processing are intrinsic parts
of one inference operation, not harness request fan-out.

Gemma 4 uses the same QAT Q4_0 GGUF and multimodal projector bytes for Antfly
and llama.cpp, and setup verifies those files. Whisper is different by design:
Transformers reads the pinned Hugging Face safetensors while whisper.cpp reads
an F16 GGML conversion. Those files are provider-native representations of the
same logical model, not byte-identical weights.

Speed uses three fresh paired lifecycles by default; memory runs each cell
alone in a separate fresh lifecycle. Antfly and Python references are warmed.
whisper.cpp uses a fresh CLI process and reports its native total time minus
model-load time: that remains cold first-inference compute, not warmed serving
latency. Results retain that timing difference and do not present whisper.cpp
as an unqualified warm speedup. Speech measures one short recording per
operation; reference and server warmups remain outside the measured call.

Memory is sampled across the rooted process tree every 200 ms. Metal uses
macOS `phys_footprint`; CUDA uses Linux `Pss_Anon + Pss_Shmem + Pss_File` plus
per-process NVML device bytes. These are different platform yardsticks and
must not be pooled across backends. Missing or invalid samples suppress the
memory comparison. Quality and agreement remain separate measured outcomes,
not claims of general model quality.

## Requirements

- macOS with Metal, or Linux with NVIDIA CUDA. There is no CPU benchmark path.
- An Antfly executable selected and obtained by the operator, for example from
  [Antfly releases](https://github.com/antflydb/antfly/releases). Choose the archive
  for your operating system, architecture and accelerator, extract it, and pass
  its executable path. This repository never downloads or updates Antfly.
- [`uv`](https://docs.astral.sh/uv/), Git, CMake 3.24 or newer, and a C/C++
  toolchain.
- Network access and sufficient model storage for `make setup`. `make check`,
  `make run`, and `make test` do not intentionally download models.

Pass the absolute path of the selected executable. For example:

```sh
ANTFLY_BIN="$(command -v antfly)"
test -x "$ANTFLY_BIN"
```

The harness derives the executable name, size, SHA-256, and reported version
from that file. The version is provenance only and never changes the workload.

## Setup and run

Running `make` only prints commands; it does not install or download anything.

```sh
make

# Explicit networked setup on macOS:
make setup BACKEND=metal ANTFLY_BIN="$ANTFLY_BIN"

# Or explicit networked setup on Linux:
make setup BACKEND=cuda ANTFLY_BIN="$ANTFLY_BIN" CUDA_ARCHITECTURES=native

# Check without downloads or measurements:
make check BACKEND=metal ANTFLY_BIN="$ANTFLY_BIN"

# Use a new output directory for each measurement:
make run BACKEND=metal ANTFLY_BIN="$ANTFLY_BIN" OUTPUT=results/example

# Deterministic harness tests; no model inference:
make test
```

`BACKEND=auto` is the default and resolves to Metal on macOS and CUDA on
Linux. Passing the backend explicitly makes operator intent clearer.

`make setup` performs all networked preparation:

1. `uv sync --frozen` creates the Python 3.12 reference environment from
   `uv.lock` (`--extra cuda` is used for CUDA).
2. `scripts/setup_references.sh` checks out and builds pinned llama.cpp and
   whisper.cpp revisions.
3. Whisper small is downloaded at its pinned Hugging Face revision, converted
   to F16 GGML, hashed, and accompanied by a required provenance sidecar.
4. The supplied Antfly binary pulls the six models.
5. Revision-pinned model snapshots, a corpus built from recorded source
   responses, and checksummed media fixtures are downloaded and verified.

To prepare only Python and the two native reference binaries, run
`make setup-tools`. `make convert-whisper` additionally prepares the pinned
Whisper GGML artifact; a valid existing artifact is rehashed and reused. Both
targets accept `BACKEND`; neither obtains Antfly.

### Focused runs

The Make interface accepts one workload, model, or cell selector and a
lifecycle count:

```sh
make run BACKEND=cuda ANTFLY_BIN="$ANTFLY_BIN" OUTPUT=results/whisper-short \
  WORKLOAD=whisper MODEL=openai/whisper-small CELL=short LIFECYCLES=1
```

For repeated `--workload` or `--model` selectors, invoke the runner directly:

```sh
.venv/bin/python scripts/run_bench.py \
  --binary "$ANTFLY_BIN" \
  --backend metal \
  --output results/focused \
  --models-dir .data/models \
  --workload whisper \
  --workload ocr \
  --lifecycles 3
```

`--only-cell` is intended for focused real smoke measurements, not a substitute
for the complete paired suite.

## Reproducible references

The direct reference pins are:

- PyTorch `2.15.0`
- Sentence Transformers `6.2.0`
- GLiNER2 `[local]` `2.1.0`
- Transformers `5.22.2`
- llama.cpp `b10809` (`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`)
- whisper.cpp `v1.9.2` (`306c88f4d1286aec1bf96e544632897886af5501`)

`uv.lock` fixes the complete Python graph, including Pillow, PyArrow,
SoundFile, process sampling, and the platform wheels. The CUDA extra adds the
CUDA toolkit build pieces and `nvidia-ml-py` needed for native reference builds
and per-process GPU memory sampling.

### CUDA 13.2 and cuBLAS

The Linux PyTorch 2.15.0 wheel declares the CUDA Toolkit 13.2.2 runtime. The
locked `cuda-toolkit[cublas,nvcc]` extra resolves cuBLAS to
`nvidia-cublas==13.4.1.3` and supplies the CUDA 13.2 `nvcc`, headers, and
development files needed by the pinned native peers. The setup script discovers
the toolkit root and either `lib` or `lib64` from the active environment rather
than assuming a machine-specific installation path. `CUDA_ARCHITECTURES` is
passed to CMake as `CMAKE_CUDA_ARCHITECTURES` and defaults to `native`; it may be
a CMake semicolon-separated architecture list.

The NVIDIA driver must support CUDA 13.2. If a supplied Antfly executable
cannot locate the environment's cuBLAS runtime, derive the library directory
instead of hardcoding `site-packages`:

```sh
CUDA_ROOT="$(
  .venv/bin/python - <<'PY'
from pathlib import Path
import site
for base in map(Path, site.getsitepackages()):
    root = base / "nvidia" / "cu13"
    if (root / "lib").is_dir():
        print(root)
        break
else:
    raise SystemExit("CUDA 13.2 environment not found")
PY
)"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Alternatively, set `CUDAToolkit_ROOT` before `make setup-tools`; the setup
script honors it. The native builds embed an rpath to the selected toolkit
library directory, but an independently supplied Antfly binary may still need
the environment above.

### Whisper conversion provenance

Whisper setup pins:

- `openai/whisper-small` revision
  `973afd24965f72e36ca33b3055d56a652f456b4d`
- source `model.safetensors` SHA-256
  `1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b`
- whisper.cpp converter revision
  `306c88f4d1286aec1bf96e544632897886af5501`
- OpenAI Whisper conversion assets `v20250625`
  (`31243bad24cc746f07d4c8bfdd2d974872cb1803`)

The conversion writes `ggml-model.bin` and a JSON sidecar containing the source
revision and hash, converter revision, and converted file hash. Benchmark
prerequisites reject a missing or mismatched sidecar. Repeated setup rehashes
and reuses an exact valid conversion; otherwise it converts again. This makes
the artifact auditable without claiming that the GGML and safetensors files are
identical.

## Paths and environment

All runtime configuration uses the `TERMITE_` prefix. Make variables map to
these values:

| Make variable | Environment variable | Default |
| --- | --- | --- |
| `ANTFLY_BIN` | `TERMITE_BINARY` | required |
| `BACKEND` | `TERMITE_BACKEND` | `auto` |
| `RESULTS_DIR` | `TERMITE_RESULTS` | `./results` |
| `MODELS_DIR` | `TERMITE_MODELS_DIR` | `./.data/models` |
| `GEN_MODELS_DIR` | `TERMITE_GEN_MODELS_DIR` | `./.data/generation-models` |
| `DATA_DIR` | `TERMITE_DATA` | `./.data` |
| `CORPUS` | `TERMITE_CORPUS` | `./.data/corpus/corpus.json` |
| `MEDIA_DIR` | `TERMITE_MEDIA` | `./.data/media` |
| `HF_HOME` | `TERMITE_HF_HOME` | `./.data/huggingface` |
| `REFERENCE_PYTHON` | `TERMITE_REFERENCE_PYTHON` | `./.venv/bin/python` |
| `REFERENCE_SNAPSHOTS` | `TERMITE_REFERENCES` | `./.data/reference_snapshots.json` |
| `LLAMA_SERVER` | `TERMITE_LLAMA_SERVER` | `./.cache/bin/llama-server` |
| `WHISPER_CPP` | `TERMITE_WHISPER_CPP` | `./.cache/bin/whisper-cli` |
| `WHISPER_CPP_MODEL` | `TERMITE_WHISPER_CPP_MODEL` | converted Whisper small |
| `WHISPER_CPP_PROVENANCE` | `TERMITE_WHISPER_CPP_PROVENANCE` | conversion sidecar |

LightOnOCR uses an 8,192 MiB scratch admission budget for the declared page and
token limits; override it with `TERMITE_SCRATCH_BUDGET_MB`. This is a workload
budget, not a host-memory measurement or an instruction to allocate that amount.
Actual process memory is sampled separately.

The LightOnOCR Transformers reference explicitly enables decoding-state caching
(`use_cache=True`) and records that policy in its lifecycle metadata. The pinned
checkpoint defaults to caching disabled, which otherwise recomputes the decoder
prefix for every output token. Earlier cache-disabled OCR timings are not
comparable to this corrected reference and must be measured again.

The runner's canonical result has a schema version, run id, status, timestamps,
backend, neutral input/provenance records, and one step per
workload/model/provider. Each step carries its actual metrics or a visible
error. Paired quality and agreement records remain attached to the measured
cells rather than being reduced to subprocess exit status.

## Fixtures, attribution, and licenses

Setup downloads fixtures at runtime; large model and media files are not
redistributed by this repository.

- Text cells use a deterministic corpus made from *On the Origin of Species*
  (first edition), [Project Gutenberg ebook 1228](https://www.gutenberg.org/ebooks/1228),
  and named Wikipedia extracts. The ebook is public domain in the United States;
  Project Gutenberg terms still apply to its distribution and use elsewhere.
  Wikipedia's [CC BY-SA and GFDL terms](https://en.wikipedia.org/wiki/Wikipedia:Copyrights)
  continue to apply. The Gutenberg URL and Wikipedia titles are mutable
  upstream, so they do not pin the text by themselves. `corpus.json` records
  the Gutenberg response hash and, for every Wikipedia extract, the resolved
  title, page id, revision id, URL, and extract hash. The seed and slicing are
  deterministic over those retained responses; preserve `corpus.json` for an
  exact replay.
- Speech cells use checksummed clips from
  `hf-internal-testing/librispeech_asr_dummy` at revision
  `5be91486e11a2d616f4ec5db8d3fd248585ac07a`. [LibriSpeech](https://www.openslr.org/12)
  is CC BY 4.0 and was prepared by Vassil Panayotov with Daniel Povey from
  LibriVox recordings.
- Multimodal image cells use named Wikimedia Commons files.
  `media_manifest.json` records each Commons title and SHA-256. Every file
  retains the license and attribution requirements on its Commons description
  page; see [Commons reuse guidance](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia).
- OCR cells render deterministic synthetic pages with Pillow and contain no
  externally sourced text or images.

The six Hugging Face model cards currently identify their weights as
Apache-2.0. The Gemma 4 card also links Google's Gemma 4 license terms; review
the pinned model card and any applicable use restrictions before downloading
or distributing weights. llama.cpp and whisper.cpp, Python packages, fixture
sources, and downloaded models retain their own licenses. This repository's
[`LICENSE`](LICENSE) covers only the harness itself.

## Privacy and publication

The harness does not upload results or contact a notification service.
Benchmark subprocesses are forced into Hugging Face/Transformers offline mode
after setup. Public JSON records omit hostnames and reduce absolute local paths
to repository-relative paths or basenames by default. Binary and artifact
hashes, backend choice, package versions, and benchmark-relevant accelerator
state remain because they are necessary provenance.

Raw provider logs can still contain diagnostics emitted by an operator's
binary. Review raw output before publishing it. Use a fresh output directory
for every run; do not merge or overwrite independently measured results.
