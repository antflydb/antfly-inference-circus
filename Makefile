# ─── Config ──────────────────────────────────────────────────────
MODELS_DIR  := $(HOME)/.termite/models/generators
LLAMA_CLI   ?= llama-cli

# Antfly variants
ANTFLY_SRC          ?= $(CURDIR)/antfly
ANTFLY_ZIG_DIR      := $(ANTFLY_SRC)/zig
ANTFLY_LATEST_BIN   := $(ANTFLY_ZIG_DIR)/zig-out/bin/antfly
ANTFLY_STABLE_BIN   ?= $(shell command -v antfly 2>/dev/null)
ZIG_BUILD_FLAGS     := -Doptimize=ReleaseFast -Dskip-openapi=true

# Benchmark settings
BENCH_PROMPT  ?= Explain what a compiler does in two sentences.
BENCH_TOKENS  ?= 32

# Gemma 4 E4B
GEMMA4_E4B_REPO := ggml-org/gemma-4-E4B-it-GGUF
GEMMA4_E4B_FILE := gemma-4-E4B-it-Q4_K_M.gguf
GEMMA4_E4B_PATH  = $(MODELS_DIR)/$(GEMMA4_E4B_REPO)/$(GEMMA4_E4B_FILE)

HAS_ANTFLY_SRC := $(wildcard $(ANTFLY_ZIG_DIR)/build.zig)

.DEFAULT_GOAL := help

help: ## Show this help
	@echo "termite-circus — Termite Benchmark Suite"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ─── Models ──────────────────────────────────────────────────────

download-gemma4-e4b: ## Download Gemma 4 E4B Q4_K_M GGUF (~5 GB)
	go run ./cmd/model-dl -repo $(GEMMA4_E4B_REPO) -file $(GEMMA4_E4B_FILE)

test: ## Run repo tests
	go test ./cmd/...
	./scripts/nightly-bench-test.sh

# ─── Builds ──────────────────────────────────────────────────────

build-antfly-latest: ## Build antfly from antfly/zig with the latest zig
ifneq ($(HAS_ANTFLY_SRC),)
	cd $(ANTFLY_ZIG_DIR) && zig build $(ZIG_BUILD_FLAGS) \
		-Dgit-commit="$$(git -C $(ANTFLY_SRC) rev-parse --short HEAD)"
else
	@echo "antfly source not found at $(ANTFLY_SRC); clone it first or set ANTFLY_SRC=/path/to/antfly"
	@exit 1
endif

build: build-antfly-latest ## Build all available engines

# ─── Benchmarks ──────────────────────────────────────────────────

BENCH_ARGS = \
	-model "$(GEMMA4_E4B_PATH)" \
	-model-dir "$(MODELS_DIR)/$(GEMMA4_E4B_REPO)" \
	-llama-cli "$(LLAMA_CLI)" \
	-prompt "$(BENCH_PROMPT)" \
	-n $(BENCH_TOKENS) \
	-output results/gemma4-e4b.md

ifneq ($(ANTFLY_STABLE_BIN),)
BENCH_ARGS += -antfly-stable "$(ANTFLY_STABLE_BIN)"
endif

ifneq ($(HAS_ANTFLY_SRC),)
BENCH_ARGS += \
	-antfly-latest "$(ANTFLY_LATEST_BIN)" \
	-antfly-latest-dir "$(ANTFLY_SRC)"
endif

bench: build ## Run benchmark (stable brew antfly + latest antfly built from source)
	go run ./cmd/bench $(BENCH_ARGS)
	@echo ""
	@cat results/gemma4-e4b.md
