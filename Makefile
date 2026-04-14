# ─── Config ──────────────────────────────────────────────────────
MODELS_DIR  := $(HOME)/.termite/models/generators
LLAMA_CLI   ?= llama-cli

# Termite builds
TERMITE_ZIG_CLEAN   := $(CURDIR)/termite-zig-clean
TERMITE_ZIG_EXP     := $(CURDIR)/termite-zig-experimentation
TERMITE_GO_DIR      := $(CURDIR)/termite-go
ZIG_BUILD_FLAGS     := -Doptimize=ReleaseFast -Donnx=false -Dpjrt=false -Dcoreml=false

# Benchmark settings
BENCH_PROMPT  ?= Explain what a compiler does in two sentences.
BENCH_TOKENS  ?= 32

# Gemma 4 E4B
GEMMA4_E4B_REPO := ggml-org/gemma-4-E4B-it-GGUF
GEMMA4_E4B_FILE := gemma-4-E4B-it-Q4_K_M.gguf
GEMMA4_E4B_PATH  = $(MODELS_DIR)/$(GEMMA4_E4B_REPO)/$(GEMMA4_E4B_FILE)

.DEFAULT_GOAL := help

help: ## Show this help
	@echo "termite-circus — Termite Benchmark Suite"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ─── Models ──────────────────────────────────────────────────────

download-gemma4-e4b: ## Download Gemma 4 E4B Q4_K_M GGUF (~5 GB)
	go run ./cmd/model-dl -repo $(GEMMA4_E4B_REPO) -file $(GEMMA4_E4B_FILE)

# ─── Builds ──────────────────────────────────────────────────────

build-zig-clean: ## Build termite-zig-clean
	cd $(TERMITE_ZIG_CLEAN) && zig build $(ZIG_BUILD_FLAGS) \
		-Dgit-commit="$$(git rev-parse --short HEAD)"

build-zig-exp: ## Build termite-zig-experimentation
	cd $(TERMITE_ZIG_EXP) && zig build $(ZIG_BUILD_FLAGS) \
		-Dgit-commit="$$(git rev-parse --short HEAD)"

build-go: ## Build termite-go
	cd $(TERMITE_GO_DIR) && GOEXPERIMENT=simd go build -o bin/termite ./pkg/termite/cmd

build: build-zig-clean build-go ## Build clean zig + go

build-all: build build-zig-exp ## Build all three

# ─── Benchmarks ──────────────────────────────────────────────────

bench: build ## Benchmark: llama.cpp + termite-zig-clean + termite-go
	go run ./cmd/bench \
		-model "$(GEMMA4_E4B_PATH)" \
		-model-dir "$(MODELS_DIR)/$(GEMMA4_E4B_REPO)" \
		-zig-clean "$(TERMITE_ZIG_CLEAN)/zig-out/bin/termite" \
		-zig-clean-dir "$(TERMITE_ZIG_CLEAN)" \
		-go-bin "$(TERMITE_GO_DIR)/bin/termite" \
		-go-dir "$(TERMITE_GO_DIR)" \
		-llama-cli "$(LLAMA_CLI)" \
		-prompt "$(BENCH_PROMPT)" \
		-n $(BENCH_TOKENS) \
		-output results/gemma4-e4b.md
	@echo ""
	@cat results/gemma4-e4b.md

bench-all: build-all ## Benchmark: llama.cpp + all three termite engines
	go run ./cmd/bench \
		-model "$(GEMMA4_E4B_PATH)" \
		-model-dir "$(MODELS_DIR)/$(GEMMA4_E4B_REPO)" \
		-zig-clean "$(TERMITE_ZIG_CLEAN)/zig-out/bin/termite" \
		-zig-clean-dir "$(TERMITE_ZIG_CLEAN)" \
		-zig-exp "$(TERMITE_ZIG_EXP)/zig-out/bin/termite" \
		-zig-exp-dir "$(TERMITE_ZIG_EXP)" \
		-go-bin "$(TERMITE_GO_DIR)/bin/termite" \
		-go-dir "$(TERMITE_GO_DIR)" \
		-llama-cli "$(LLAMA_CLI)" \
		-prompt "$(BENCH_PROMPT)" \
		-n $(BENCH_TOKENS) \
		-output results/gemma4-e4b.md
	@echo ""
	@cat results/gemma4-e4b.md
