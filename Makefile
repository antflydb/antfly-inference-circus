REFERENCE_PYTHON    ?= $(CURDIR)/.venv/bin/python
ANTFLY_BIN          ?=
BACKEND             ?= auto
DATA_DIR            ?= $(CURDIR)/.data
MODELS_DIR          ?= $(DATA_DIR)/models
GEN_MODELS_DIR      ?= $(DATA_DIR)/generation-models
HF_HOME             ?= $(DATA_DIR)/huggingface
CORPUS              ?= $(DATA_DIR)/corpus/corpus.json
MEDIA_DIR           ?= $(DATA_DIR)/media
REFERENCE_SNAPSHOTS ?= $(DATA_DIR)/reference_snapshots.json
RESULTS_DIR         ?= $(CURDIR)/results
OUTPUT              ?= $(RESULTS_DIR)/run
LLAMA_SERVER        ?= $(CURDIR)/.cache/bin/llama-server
WHISPER_CPP         ?= $(CURDIR)/.cache/bin/whisper-server
WHISPER_CPP_MODEL   ?= $(DATA_DIR)/references/whisper-small-whispercpp-v1.9.2/ggml-model.bin
WHISPER_CPP_PROVENANCE ?= $(WHISPER_CPP_MODEL).provenance.json
CUDA_ARCHITECTURES  ?= native
LIFECYCLES          ?= 3
MEMORY_LIFECYCLES   ?= 3
REPETITIONS         ?= 10
WORKLOAD            ?=
MODEL               ?=
CELL                ?=

RUNNER := $(REFERENCE_PYTHON) scripts/run_bench.py
RUN_ENV := env \
	TERMITE_BINARY="$(ANTFLY_BIN)" \
	TERMITE_BACKEND="$(BACKEND)" \
	TERMITE_RESULTS="$(RESULTS_DIR)" \
	TERMITE_MODELS_DIR="$(MODELS_DIR)" \
	TERMITE_GEN_MODELS_DIR="$(GEN_MODELS_DIR)" \
	TERMITE_DATA="$(DATA_DIR)" \
	TERMITE_CORPUS="$(CORPUS)" \
	TERMITE_MEDIA="$(MEDIA_DIR)" \
	TERMITE_HF_HOME="$(HF_HOME)" \
	TERMITE_REFERENCE_PYTHON="$(REFERENCE_PYTHON)" \
	TERMITE_REFERENCES="$(REFERENCE_SNAPSHOTS)" \
	TERMITE_LLAMA_SERVER="$(LLAMA_SERVER)" \
	TERMITE_WHISPER_CPP="$(WHISPER_CPP)" \
	TERMITE_WHISPER_CPP_MODEL="$(WHISPER_CPP_MODEL)" \
	TERMITE_WHISPER_CPP_PROVENANCE="$(WHISPER_CPP_PROVENANCE)"
COMMON_ARGS := --binary "$(ANTFLY_BIN)" --backend "$(BACKEND)" --models-dir "$(MODELS_DIR)"
RUN_ARGS := $(COMMON_ARGS) --output "$(OUTPUT)" --lifecycles "$(LIFECYCLES)" --memory-lifecycles "$(MEMORY_LIFECYCLES)" --repetitions "$(REPETITIONS)"
ifneq ($(strip $(WORKLOAD)),)
RUN_ARGS += --workload "$(WORKLOAD)"
endif
ifneq ($(strip $(MODEL)),)
RUN_ARGS += --model "$(MODEL)"
endif
ifneq ($(strip $(CELL)),)
RUN_ARGS += --only-cell "$(CELL)"
endif

.DEFAULT_GOAL := help

.PHONY: help require-binary setup-tools convert-whisper setup check run test

help: ## Show commands without changing the system
	@printf '%s\n' \
	  'termite-circus — paired inference benchmarks for a supplied Antfly binary' \
	  '' \
	  '  make setup-tools BACKEND=metal|cuda' \
	  '      Create the locked Python environment and build pinned native peers.' \
	  '  make convert-whisper BACKEND=metal|cuda' \
	  '      Prepare and verify the pinned Whisper small conversion.' \
	  '  make setup ANTFLY_BIN=/path/to/antfly BACKEND=metal|cuda' \
	  '      Run all explicit networked setup: tools, models, corpus, and media.' \
	  '  make check ANTFLY_BIN=/path/to/antfly BACKEND=metal|cuda' \
	  '      Check prerequisites without downloading or running measurements.' \
	  '  make run ANTFLY_BIN=/path/to/antfly BACKEND=metal|cuda OUTPUT=results/name' \
	  '      Run quality, speed, and isolated memory. Select WORKLOAD, MODEL, or CELL.' \
	  '      LIFECYCLES=3 MEMORY_LIFECYCLES=3 REPETITIONS=10 (speech/OCR calls).' \
	  '  make test' \
	  '      Run deterministic harness tests; no models or Antfly process required.' \
	  '' \
	  'CUDA builds accept CUDA_ARCHITECTURES=native (default) or a CMake architecture list.'

require-binary:
	@test -n "$(ANTFLY_BIN)" || { printf 'ANTFLY_BIN is required\n' >&2; exit 2; }
	@test -x "$(ANTFLY_BIN)" || { printf 'ANTFLY_BIN is not executable: %s\n' "$(ANTFLY_BIN)" >&2; exit 2; }

setup-tools: ## Create the locked Python environment and pinned native peers
	TERMITE_CUDA_ARCHITECTURES="$(CUDA_ARCHITECTURES)" scripts/setup_references.sh --backend "$(BACKEND)"

convert-whisper: ## Prepare and verify the pinned Whisper small conversion
	TERMITE_DATA="$(DATA_DIR)" TERMITE_HF_HOME="$(HF_HOME)" \
	TERMITE_WHISPER_CPP_MODEL="$(WHISPER_CPP_MODEL)" \
	TERMITE_WHISPER_CPP_PROVENANCE="$(WHISPER_CPP_PROVENANCE)" \
	TERMITE_CUDA_ARCHITECTURES="$(CUDA_ARCHITECTURES)" \
	scripts/setup_references.sh --backend "$(BACKEND)" --convert-whisper

setup: require-binary convert-whisper ## Explicitly fetch and verify every benchmark input
	$(RUN_ENV) $(RUNNER) $(COMMON_ARGS) --phase setup

check: require-binary ## Check all prerequisites without downloads or measurements
	$(RUN_ENV) $(RUNNER) $(COMMON_ARGS) --check

run: require-binary ## Run paired workloads; optionally set WORKLOAD, MODEL, CELL, LIFECYCLES
	$(RUN_ENV) $(RUNNER) $(RUN_ARGS)

test: ## Run deterministic harness tests
	$(REFERENCE_PYTHON) -m unittest discover -s tests -p 'test_*.py'
