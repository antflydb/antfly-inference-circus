# ─── Config ──────────────────────────────────────────────────────
MODELS_DIR  := $(HOME)/.antfly/inference/models/generators
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

# Gemma 4 E4B QAT q4_0 (Google quantization-aware checkpoint, ~5 GB)
GEMMA4_E4B_QAT_REPO := google/gemma-4-E4B-it-qat-q4_0-gguf
GEMMA4_E4B_QAT_FILE := gemma-4-E4B_q4_0-it.gguf
GEMMA4_E4B_QAT_PATH  = $(MODELS_DIR)/$(GEMMA4_E4B_QAT_REPO)/$(GEMMA4_E4B_QAT_FILE)

# Gemma 4 E4B MTP assistant drafter (safetensors, pulled via antfly)
GEMMA4_E4B_MTP_ASSISTANT     := google/gemma-4-E4B-it-assistant
GEMMA4_E4B_MTP_ASSISTANT_DIR := $(HOME)/.antfly/inference/models/$(GEMMA4_E4B_MTP_ASSISTANT)

# Server-based comparison engines
OLLAMA_BIN         ?= ollama
OLLAMA_QAT_MODEL   ?= gemma4-e4b-qat-q4_0
VLLM_METAL_BIN     ?= $(CURDIR)/.tools/vllm-metal/bin/vllm-metal
VLLM_QAT_MLX_MODEL ?= mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit

HAS_ANTFLY_SRC := $(wildcard $(ANTFLY_ZIG_DIR)/build.zig)

.DEFAULT_GOAL := help

help: ## Show this help
	@echo "termite-circus — Antfly Inference Benchmark Suite"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ─── Models ──────────────────────────────────────────────────────

download-gemma4-e4b: ## Download Gemma 4 E4B Q4_K_M GGUF (~5 GB)
	go run ./cmd/model-dl -repo $(GEMMA4_E4B_REPO) -file $(GEMMA4_E4B_FILE)

download-qwen3.6-a3b: ## Download Qwen 3.6 35B-A3B UD-Q4_K_M GGUF (~21 GB)
	go run ./cmd/model-dl -repo $(QWEN36_A3B_REPO) -file $(QWEN36_A3B_FILE)

download-gemma4-e4b-qat: ## Download Gemma 4 E4B QAT q4_0 GGUF (~5 GB)
	go run ./cmd/model-dl -repo $(GEMMA4_E4B_QAT_REPO) -file $(GEMMA4_E4B_QAT_FILE)

download-gemma4-e4b-mtp-assistant: ## Pull the Gemma 4 E4B MTP assistant drafter via antfly
	$(ANTFLY_STABLE_BIN) inference pull $(GEMMA4_E4B_MTP_ASSISTANT)

setup-ollama-gemma4-qat: ## Import the QAT q4_0 GGUF into the ollama store (ollama server must be running)
	@mkdir -p .tools
	printf 'FROM %s\n' "$(GEMMA4_E4B_QAT_PATH)" > .tools/Modelfile.gemma4-qat
	$(OLLAMA_BIN) create $(OLLAMA_QAT_MODEL) -f .tools/Modelfile.gemma4-qat

setup-vllm-metal: ## Install vllm-metal into .tools/vllm-metal (needs uv; arm64 python 3.12)
	uv venv --python 3.12 .tools/vllm-metal
	VIRTUAL_ENV=$(CURDIR)/.tools/vllm-metal uv pip install vllm-metal

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

# $(call bench_args,<model-path>,<model-dir>,<output-md>,<backend-budget-mb>)
define bench_args
	-model "$(1)" \
	-model-dir "$(2)" \
	-llama-cli "$(LLAMA_CLI)" \
	-prompt "$(BENCH_PROMPT)" \
	-n $(BENCH_TOKENS) \
	-backend-budget-mb $(4) \
	-output $(3)
endef

# 36 GB unified memory leaves ~28 GB for the inference backend after OS + Go bench harness.
GEMMA4_E4B_BUDGET_MB := 20480
QWEN36_A3B_BUDGET_MB := 28672

EXTRA_BENCH_ARGS :=
ifneq ($(ANTFLY_STABLE_BIN),)
EXTRA_BENCH_ARGS += -antfly-stable "$(ANTFLY_STABLE_BIN)"
endif
ifneq ($(HAS_ANTFLY_SRC),)
EXTRA_BENCH_ARGS += \
	-antfly-latest "$(ANTFLY_LATEST_BIN)" \
	-antfly-latest-dir "$(ANTFLY_SRC)"
endif

# QAT + MTP settings: single cooled canonical samples per engine.
BENCH_QAT_TOKEN_COUNTS       ?= 128
BENCH_QAT_COOLDOWN_SECONDS   ?= 60
BENCH_QAT_RUN_TIMEOUT        ?= 600
BENCH_QAT_SPECULATIVE_K      ?= 2
BENCH_QAT_SPECULATION_POLICY ?= force

# No `build` dependency: this target benchmarks pinned antfly binaries
# (ANTFLY_STABLE_BIN / EXTRA_BENCH_ARGS) and must not trigger a zig build.
bench-gemma4-e4b-qat-mtp: ## Cooled single-sample Gemma 4 E4B QAT + MTP bench vs llama.cpp/ollama/vllm
	go run ./cmd/bench \
		-model "$(GEMMA4_E4B_QAT_PATH)" \
		-model-dir "$(MODELS_DIR)/$(GEMMA4_E4B_QAT_REPO)" \
		-llama-cli "$(LLAMA_CLI)" \
		-prompt "$(BENCH_PROMPT)" \
		-token-counts "$(BENCH_QAT_TOKEN_COUNTS)" \
		-cooldown-seconds $(BENCH_QAT_COOLDOWN_SECONDS) \
		-warmup-tokens 8 \
		-run-timeout-seconds $(BENCH_QAT_RUN_TIMEOUT) \
		-ignore-eos \
		-backend-budget-mb $(GEMMA4_E4B_BUDGET_MB) \
		-antfly-draft-model "$(GEMMA4_E4B_MTP_ASSISTANT_DIR)" \
		-antfly-speculative-k $(BENCH_QAT_SPECULATIVE_K) \
		-antfly-speculation-policy $(BENCH_QAT_SPECULATION_POLICY) \
		-ollama-bin "$(OLLAMA_BIN)" \
		-ollama-model "$(OLLAMA_QAT_MODEL)" \
		-vllm-bin "$(wildcard $(VLLM_METAL_BIN))" \
		-vllm-model "$(VLLM_QAT_MLX_MODEL)" \
		-output results/gemma4-e4b-qat-mtp.md \
		$(EXTRA_BENCH_ARGS)
	@echo ""
	@cat results/gemma4-e4b-qat-mtp.md

bench-gemma4-e4b: build ## Bench llama.cpp + antfly variants on Gemma 4 E4B
	go run ./cmd/bench \
		$(call bench_args,$(GEMMA4_E4B_PATH),$(MODELS_DIR)/$(GEMMA4_E4B_REPO),results/gemma4-e4b.md,$(GEMMA4_E4B_BUDGET_MB)) \
		$(EXTRA_BENCH_ARGS)
	@echo ""
	@cat results/gemma4-e4b.md

bench-qwen3.6-a3b: build ## Bench llama.cpp + antfly variants on Qwen 3.6 35B-A3B
	go run ./cmd/bench \
		$(call bench_args,$(QWEN36_A3B_PATH),$(MODELS_DIR)/$(QWEN36_A3B_REPO),results/qwen3.6-a3b.md,$(QWEN36_A3B_BUDGET_MB)) \
		$(EXTRA_BENCH_ARGS)
	@echo ""
	@cat results/qwen3.6-a3b.md

bench: bench-gemma4-e4b bench-qwen3.6-a3b ## Run all benchmarks
