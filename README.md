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
| speech transcription | `openai/whisper-small` | Transformers 5.23.0 and whisper.cpp 1.9.2 |
| document OCR | `lightonai/LightOnOCR-3-0.8B` | Transformers 5.23.0 |

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

Speed uses three fresh paired lifecycles by default. Isolated memory uses
three fresh lifecycles **per provider and cell**, retaining each observation
and the median/minimum/maximum. Speech and OCR perform ten independent warmed
requests per cell and lifecycle. Repetition increases measurement count, not
workload size. All Whisper providers remain resident: whisper.cpp uses the
pinned `whisper-server`, with real per-cell warmups excluded from timings.
There is no cold CLI timing subtraction.

Whisper has three distinct natural recordings: short (about 5.86 seconds),
medium (15.115 seconds, `1272-128104-0011`), and long (29.4 seconds,
`1272-128104-0004`). Durations are calculated from actual WAV samples. Each
recording fits in one 30-second audio window; none is padded, concatenated,
repeated, or assembled from multiple requests.

Memory observations target a 200 ms interval without suspending the rooted
process tree, only in isolated memory lifecycles. Darwin walks have a finite
two-second allowance per process; actual observation cost and gaps are retained,
so this is not a guaranteed 200 ms cadence or an unsampled lifetime maximum.
Typed target-unavailable reads allow an additional bounded 200 ms to confirm
the original process has exited, become a zombie, or been replaced. This does
not reread memory or alter the target. Incomplete observations and confirmation
diagnostics remain retained; live or unconfirmed failures still invalidate
coverage rather than contribute a partial peak.
Speed and task-quality runs do not start the observer. Direct per-workload
instruments enable it only with `TERMITE_SAMPLE_MEMORY=1`; the top-level runner
sets this by phase.
CUDA uses concurrent Linux `Pss_Anon + Pss_Shmem + Pss_File` plus discrete
per-process NVML device bytes (`linux-pss-nvml-v1`). Metal retains raw
`phys_footprint` and observed resident mappings of the actual model artifacts
and unlinked regular-file storage in the owned process tree. The latter includes
temporary repacked weights: dirty file-backed pages need not enter the ledger.
Because public unprivileged macOS APIs cannot prove ledger/mapped-page overlap,
Metal reports an inclusive **interval**, not a fictitious exact sum
(`darwin-ledger-mapped-storage-interval-v1`). If concurrent ledger bytes are
`F` and measured unique mapped-storage residency lies in `[L,U]`, the inclusive
result lies in `[max(F,L), F+U]`. Aliases and cross-process sharing widen it
conservatively. Neither model file sizes nor independently peaked components
are added to a peak.
This is VM-object residency, including file-cache pages in mapped windows
without a task PTE. It is not a touched-byte working set, minimum required
RAM, or whole-machine demand estimate. File-cache warmth can affect it;
the interval does not prove every mapped tensor was used.

Opaque mappings, observer-budget exhaustion, live-process read failures, and
other incomplete coverage invalidate the memory comparison rather than
silently dropping bytes. `TERMITE_METAL_MAPPING_EVIDENCE` may document pinned
source proof that active model and temporary-storage loaders retain ordinary
top-level POSIX mappings; only that audited original-mapping scope permits
ignoring opaque submaps. It does not attest Apple driver-created aliases or
total physical RAM. Do not reuse proof for a different executable or loader revision.
The source audits are retained in
[`native_precision_contracts.json`](scripts/native_precision_contracts.json) and
[`reference_mapping_contracts.json`](scripts/reference_mapping_contracts.json).
They cover the named revisions, not arbitrary replacement binaries.
Raw components, coverage, observation cost, sample gaps, and interval endpoints
remain in the data. Ratios are intervals unless both endpoints coincide.
These platform metrics must not be pooled across backends.

Every provider/cell is `complete`, `refused`, or `failed`. A structured HTTP
400/413/422 rejection is a request refusal, not a successful measurement or
a claim about every possible request to that model. Missing operations,
transport errors, and server failures remain failures. Completed sibling
cells retain their results; refused cells remain visible without ratios.
Provider/run `partial` status never means full workload support.
Hard warmup failures skip that cell without fabricating measured requests.
Structured warmup refusals remain refused warmups; already-planned independent
measurements are actually attempted. A later successful response without a
completed warmup does not become a valid warm-timing comparison. Quality
failures retain attempted/successful/refused/missing task counts and request
statuses; partial task coverage never becomes a full-subset accuracy score.

Speed ratios retain exact values and observed lifecycle spread. A ratio in
`[1/1.05,1.05]` is a practical tie, not a significance test.

### Quality and precision evidence

The default run includes task-quality diagnostics, separately from timing and
memory. Setup downloads public frozen inputs; checks and measurements are
offline. Each quality module records source revisions/hashes, sample IDs,
scoring policy, denominators, and provider differences in a compact
`quality/<model>/result.json`:

| Model | Evidence and limits |
| --- | --- |
| Embedding | Every timed vector is paired, not a sample. Frozen BEIR/SciFact: 24 queries, 256 candidates including all positive judgments; Recall@1/5/10 and full-rank MRR. Restricted-candidate diagnostic, not a full BEIR score. |
| GLiNER / Decide | All timed lifecycle/call/item outputs, paired by input digest: entity spans/labels, classification, and relations. Agreement is not labeled task accuracy. |
| Whisper | 32 frozen natural LibriSpeech recordings: micro WER/CER and per-recording errors for all three providers. |
| Gemma 4 | First 12 test examples each from GSM8K and ARC-Challenge: scored math/science accuracy with original denominators, failed responses, and paired differences. Small text-only diagnostics, not multimodal or full-benchmark accuracy. |
| OCR | First 16 CORD-v2 test receipt images: raw WER/CER plus separately named markup/text-normalized diagnostics. CORD annotations omit some visible text, so full-image errors also penalize text absent from the reference. |

Quality uses the same executables, backend, weights, and provider loading/cache
policies as timing, with declared task-specific prompts and output budgets.
Python models use library-default loading dtype. Parameter storage/device
inventories, loaded components, ambient autocast and TF32 settings are observed
at model-load time; they do not trace internal forward autocast or prove
kernel/accumulation dtype. OCR additionally proves a populated prefill cache
and an incremental decode outside measurement.

Native source-derived precision facts require an exact executable SHA-256
binding, model/backend selection, full source commit, and pinned public source
links. An optional `TERMITE_NATIVE_EVIDENCE` JSON file supplies
`schema_version: 1`, an `executables` object keyed by binary SHA-256, and
`source_commit` / `models[model][backend]` policies with a nonempty `sources`
list. Policies can describe checkpoint, parameter, compute, accumulation,
cache and component-specific dtype, plus route conditions. Without a matching
binding the output explicitly says evidence is unavailable; a version string
or output cosine never substitutes for it.

Gemma compares native CLI rendered prompts/token IDs and llama.cpp
template/tokenization diagnostics transiently. Retained JSON contains SHA-256
fingerprints, counts, scope and comparison booleans, not full prompts or token
arrays. Native IDs precede media expansion. The HTTP endpoint does not expose
token IDs: matching counts are descriptive, not HTTP token identity or prompt
parity.

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
   responses, checksummed media, and the frozen public quality subsets are
   downloaded and verified.

Whisper's native pull uses the documented `--tasks transcribe` registration
([official guide](https://github.com/antflydb/antfly/blob/30b018e12419f7b43dd36ef29e6ebab2a53338ff/docs/guides/transcription.mdx#L129-L135)).
With Antfly v0.3.1, omitting this flag can publish Whisper as an image-only
reader, causing an otherwise valid `/ai/v1/dictate` request to return
`UNSUPPORTED_MODALITY` before model execution. Existing installations prepared
without the flag need an explicit re-pull into the same model directory:

```sh
"$ANTFLY_BIN" inference pull openai/whisper-small:safetensors \
  --models-dir "${TERMITE_MODELS_DIR:-${TERMITE_DATA:-.data}/models}" \
  --tasks transcribe
```

This uses the official pull path to regenerate registration metadata; do not
edit the manifest manually. Preserve the original refusal records, verify the
re-pulled source revision and weight checksums against the pinned reference,
and use a fresh output directory for subsequent runs.

The private materialized reference manifest records each verified snapshot's
local path. Offline checks use that selected snapshot, rather than requesting
unneeded repository files such as model-card images. Do not transfer those
machine-local paths between hosts; materialize each host's manifest from its
own verified snapshots.

To prepare only Python and the two native reference binaries, run
`make setup-tools`. `make convert-whisper` additionally prepares the pinned
Whisper GGML artifact; a valid existing artifact is rehashed and reused. Both
targets accept `BACKEND`; neither obtains Antfly.

### Focused runs

The Make interface accepts workload, model, and cell selectors, plus lifecycle
counts and an independent speech/OCR repetition count:

```sh
make run BACKEND=cuda ANTFLY_BIN="$ANTFLY_BIN" OUTPUT=results/whisper-short \
  WORKLOAD=whisper MODEL=openai/whisper-small CELL=short \
  LIFECYCLES=1 MEMORY_LIFECYCLES=1 REPETITIONS=2
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
  --lifecycles 3 --memory-lifecycles 3 --repetitions 10
```

`--only-cell` is intended for focused real smoke measurements, not a substitute
for the complete paired suite.
Use repeated `--phase speed`, `--phase memory`, or `--phase quality` to select
phases. The default runs all three. A timing-cell selector does not reduce
the separate frozen quality subset.


## Reproducible references

The direct reference pins are:

- PyTorch `2.15.0`
- Sentence Transformers `6.2.0`
- GLiNER2 `[local]` `2.1.0`
- Transformers `5.23.0`
- llama.cpp upstream source tag `b10809` (`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`)
- whisper.cpp `v1.9.2` (`306c88f4d1286aec1bf96e544632897886af5501`)

llama.cpp's generated build number can differ in a shallow checkout. The harness
records the actual ready server's `GET /props.build_info` alongside executable
identity; it does not relabel that response as the upstream tag. A separate
`--version` process is not a metadata-only operation on this Metal build: static
backend initialization compiles shaders before argument handling.

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
than assuming a machine-specific installation path. It supplies versioned CUDA
library paths explicitly because NVIDIA's wheels omit unversioned linker
symlinks. `CUDA_ARCHITECTURES` is
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
| `WHISPER_CPP` | `TERMITE_WHISPER_CPP` | `./.cache/bin/whisper-server` |
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

The runner's canonical schema v2 result has a run id, status, timestamps,
backend, neutral input/provenance records, one step per model/provider/phase,
and explicit per-cell states. `result.json`, `provenance.json`,
`inputs.provenance.json`, and `terminal.json` are independently bounded below
1 MiB. `receipt.json` records SHA-256 checksums for JSON evidence. Provider
execution failures never erase completed sibling measurements. A fully
recorded refusal can produce exit code zero with `partial` support; missing
evidence and unexpected execution failures return nonzero.

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
- OCR timing cells render deterministic synthetic pages with Pillow. Separate
  quality inputs are original [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
  test receipts at revision `7f0115a4b758a71d6473b8d085751692da2fef98`,
  CC BY 4.0. Original image/annotation bytes are pinned; only reference reading
  order and explicitly named diagnostic normalizations are derived.
- BEIR/SciFact retrieval sources and judgments are CC BY-SA 4.0; GSM8K is MIT;
  ARC-Challenge is CC BY-SA 4.0. The quality modules pin each source revision
  and checksum and retain attribution/license pointers in provenance.

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
