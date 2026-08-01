<img width="2304" height="1848" alt="termite-circus" src="https://github.com/user-attachments/assets/f1feba1e-a4bf-4a37-a17d-b2c57389b10f" />

# termite-circus

Manual, category-wide benchmarks for the local inference surface bundled in
Antfly. A run measures exactly one caller-supplied Antfly binary. The harness
does not install, update, fetch, select, or build Antfly, and it never posts a
result anywhere.

The suite covers every inference role documented as working in Antfly
v0.2.0-rc.24: dense, sparse, and multimodal embedding; reranking; fixed and
model-based chunking; GLiNER2 schema extraction; document reading/OCR;
transcription; and local generation including Gemma 4 QAT/MTP cases.

`matrix.json` names `v0.2.0-rc.24` as its request/response contract. A newer or
older supplied binary is still measurable, but `prereqs` emits a visible
contract warning and provenance retains both versions.

Rewriters remain in the compatibility notes rather than the performance matrix
because rc.24 documents the encoder/decoder path as unsafe or nonfunctional.

## Interface

Running `make` only prints the available commands:

```bash
make
make prereqs PROFILE=smoke ANTFLY_BIN=/absolute/path/to/antfly
make models PROFILE=smoke ANTFLY_BIN=/absolute/path/to/antfly
make smoke ANTFLY_BIN=/absolute/path/to/antfly
make full ANTFLY_BIN=/absolute/path/to/antfly QUIET_HOST_ACK=1
make category CATEGORY=extraction ANTFLY_BIN=/absolute/path/to/antfly
make validate RESULT=results/campaigns/<id>
```

`models` is deliberately separate because it downloads artifacts. Likewise,
`setup-tools` explicitly creates the Python 3.12 reference-provider environment;
benchmark commands never install dependencies.

For a full run, populate reference-provider caches explicitly too:

```bash
make setup-tools
make models PROFILE=full ANTFLY_BIN=/absolute/path/to/antfly
make reference-models PROFILE=full
```

Reference subprocesses run with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Ollama, vLLM-Metal, and whisper.cpp models must be
installed separately and named through the environment variables below. The
prerequisite check fails before measurement when any of these inputs is absent.

The default model root is `~/.antfly/inference/models`. Override it with
`MODELS_DIR=/path`. `ANTFLY_SUBJECT_ID` is an optional display label; the
executable's version, architecture, and SHA-256 are always derived from the
binary itself.

## Profiles

`smoke` runs only the supplied Antfly binary, one lifecycle and one measured
request, against one certified model per working role. It is a compatibility
and correctness check, not publication evidence.

`full` adds local reference runtimes and uses three fresh lifecycles per cell:

- llama.cpp, Ollama, and vLLM-Metal for generators
- ONNX Runtime and Sentence Transformers for embedding and reranking
- the GLiNER2 local runtime for schema extraction
- Transformers for Florence-2 OCR
- faster-whisper and whisper.cpp for transcription

Provider-native formats are allowed when a runtime cannot consume the same
artifact. Every row therefore names the logical model, artifact, variant, and
provider; such rows must not be described as byte-identical comparisons.

Full memory results use the inference max-of-two-views method: peak rooted
process-tree RSS versus peak growth in anonymous, compressed, and wired system
memory. Set `QUIET_HOST_ACK=1` only when the host is quiet and the system delta
can reasonably be attributed to the benchmark. Otherwise memory is retained as
diagnostic data and marked with a dagger in the report.

## Campaign artifacts

Every attempted run writes `results/campaigns/<run-id>/`:

```text
report.md          compact category tables
result.json        canonical normalized rows
provenance.json    binary, model, provider, host, and input fingerprints
receipt.json       completeness and quality-gate status
raw/               provider logs and 200 ms memory timelines
```

A provider crash, timeout, missing quality result, or partial run remains a
visible failed row. Raw logs do not get pasted into Markdown or sent to Zulip.

## Quality before speed

Each performance cell has a small deterministic gate appropriate to its role:
semantic ordering for embeddings, relevant-document ordering for rerankers,
expected schema values for GLiNER2, valid chunk coverage, OCR/transcription word
recovery, and a nonempty constrained completion for generators. These fixtures
guard against timing a broken response; they are not broad model-quality claims.

The tracked matrix is [`matrix.json`](matrix.json), and deterministic text
fixtures are in [`fixtures/cases.json`](fixtures/cases.json). Media fixtures are
generated under the gitignored `results/fixtures/` directory with
`make fixtures`.

## Reference-provider configuration

The full profile recognizes these overrides:

```bash
LLAMA_CLI=/path/to/llama-cli
OLLAMA_BIN=/path/to/ollama
VLLM_METAL_BIN=/path/to/vllm-metal
WHISPER_CLI=/path/to/whisper-cli
WHISPER_CPP_MODEL=/path/to/ggml-tiny.bin
REFERENCE_PYTHON=/path/to/python3.12
```

Ollama and vLLM model names may be supplied globally with `OLLAMA_MODEL` and
`VLLM_MODEL`, or per benchmark with an uppercased benchmark identifier, for
example `OLLAMA_MODEL_GENERATION_GEMMA4_E4B`.

No scheduler or notification service is bundled. A caller such as
antfly-circus may archive, render, or publish a completed campaign after it has
validated the receipt.

The disposition of the two pre-existing generator PRs and the safe parent
repin order are recorded in [`PR_ASSESSMENT.md`](PR_ASSESSMENT.md).
