# ─── Config ──────────────────────────────────────────────────────
MODELS_DIR  := $(HOME)/.termite/models/generators
LLAMA_CLI   ?= llama-cli

# Antfly variants
ANTFLY_SRC          ?= $(CURDIR)/antfly
ANTFLY_ZIG_DIR      := $(ANTFLY_SRC)/zig
ANTFLY_LATEST_BIN   := $(ANTFLY_ZIG_DIR)/zig-out/bin/antfly
ANTFLY_STABLE_BIN   ?= $(shell command -v antfly 2>/dev/null)
ANTFLY_BREW_FORMULA ?= antflydb/taps/antfly
ZIG_BUILD_FLAGS     := -Doptimize=ReleaseFast -Dedition=full

# Benchmark settings
BENCH_PROMPT  ?= Explain what a compiler does in two sentences.
BENCH_TOKENS  ?= 32

# Gemma 4 E4B (small dense, ~5 GB)
GEMMA4_E4B_REPO := ggml-org/gemma-4-E4B-it-GGUF
GEMMA4_E4B_FILE := gemma-4-E4B-it-Q4_K_M.gguf
GEMMA4_E4B_PATH  = $(MODELS_DIR)/$(GEMMA4_E4B_REPO)/$(GEMMA4_E4B_FILE)

# Qwen 3.6 35B-A3B (MoE, 3B active, ~21 GB)
QWEN36_A3B_REPO := unsloth/Qwen3.6-35B-A3B-GGUF
QWEN36_A3B_FILE := Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
QWEN36_A3B_PATH  = $(MODELS_DIR)/$(QWEN36_A3B_REPO)/$(QWEN36_A3B_FILE)

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

download-qwen3.6-a3b: ## Download Qwen 3.6 35B-A3B UD-Q4_K_M GGUF (~21 GB)
	go run ./cmd/model-dl -repo $(QWEN36_A3B_REPO) -file $(QWEN36_A3B_FILE)

test: ## Run repo tests
	go test ./cmd/...
	./scripts/nightly-bench-test.sh

# ─── Builds ──────────────────────────────────────────────────────

update-stable: ## brew upgrade the stable antfly so it tracks the latest published release
	@brew update --quiet >/dev/null 2>&1 || true
	@brew upgrade $(ANTFLY_BREW_FORMULA) 2>&1 | grep -v "already installed" || true
	@echo "stable antfly: $$($(ANTFLY_STABLE_BIN) --version 2>/dev/null | head -1)"

update-antfly: ## Fetch antfly main and fast-forward $(ANTFLY_SRC)
ifneq ($(HAS_ANTFLY_SRC),)
	@git -C $(ANTFLY_SRC) fetch --quiet --tags --prune origin
	@git -C $(ANTFLY_SRC) checkout --quiet --force origin/main
	@echo "antfly @ $$(git -C $(ANTFLY_SRC) log -1 --format='%h %ci %s')"
else
	@echo "antfly source not found at $(ANTFLY_SRC); clone https://github.com/antflydb/antfly into $(ANTFLY_SRC) or set ANTFLY_SRC=/path/to/antfly"
	@exit 1
endif

build-antfly-latest: update-antfly ## Build antfly from antfly/zig with the latest zig
	cd $(ANTFLY_ZIG_DIR) && zig build $(ZIG_BUILD_FLAGS) \
		-Dantfly-version="$$(git -C $(ANTFLY_SRC) rev-parse --short HEAD)"

build: update-stable build-antfly-latest ## Refresh stable + build latest antfly

# ─── Benchmarks ──────────────────────────────────────────────────

# $(call bench_args,<model-path>,<model-dir>,<output-md>)
define bench_args
	-model "$(1)" \
	-model-dir "$(2)" \
	-llama-cli "$(LLAMA_CLI)" \
	-prompt "$(BENCH_PROMPT)" \
	-n $(BENCH_TOKENS) \
	-output $(3)
endef

EXTRA_BENCH_ARGS :=
ifneq ($(ANTFLY_STABLE_BIN),)
EXTRA_BENCH_ARGS += -antfly-stable "$(ANTFLY_STABLE_BIN)"
endif
ifneq ($(HAS_ANTFLY_SRC),)
EXTRA_BENCH_ARGS += \
	-antfly-latest "$(ANTFLY_LATEST_BIN)" \
	-antfly-latest-dir "$(ANTFLY_SRC)"
endif

bench-gemma4-e4b: build ## Bench llama.cpp + antfly variants on Gemma 4 E4B
	go run ./cmd/bench \
		$(call bench_args,$(GEMMA4_E4B_PATH),$(MODELS_DIR)/$(GEMMA4_E4B_REPO),results/gemma4-e4b.md) \
		$(EXTRA_BENCH_ARGS)
	@echo ""
	@cat results/gemma4-e4b.md

bench-qwen3.6-a3b: build ## Bench llama.cpp + antfly variants on Qwen 3.6 35B-A3B
	go run ./cmd/bench \
		$(call bench_args,$(QWEN36_A3B_PATH),$(MODELS_DIR)/$(QWEN36_A3B_REPO),results/qwen3.6-a3b.md) \
		$(EXTRA_BENCH_ARGS)
	@echo ""
	@cat results/qwen3.6-a3b.md

bench: bench-gemma4-e4b bench-qwen3.6-a3b ## Run all benchmarks
