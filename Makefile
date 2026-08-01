PYTHON           ?= python3
REFERENCE_PYTHON ?= $(CURDIR)/.venv/bin/python
ANTFLY_BIN       ?=
ANTFLY_SUBJECT_ID ?=
MODELS_DIR       ?= $(HOME)/.antfly/inference/models
PROFILE          ?= smoke
CATEGORY         ?=
RUN_ID           ?=
OUTPUT           ?=
RESULT           ?=

CIRCUS := $(PYTHON) scripts/circus.py
COMMON_ARGS := --profile $(PROFILE) --models-dir "$(MODELS_DIR)" --python "$(REFERENCE_PYTHON)"
ifneq ($(strip $(ANTFLY_BIN)),)
COMMON_ARGS += --antfly "$(ANTFLY_BIN)"
endif
ifneq ($(strip $(CATEGORY)),)
COMMON_ARGS += --category "$(CATEGORY)"
endif

.DEFAULT_GOAL := help

.PHONY: help fixtures prereqs models reference-models setup-tools smoke full category validate test

help: ## Show commands without running a benchmark
	@echo "termite-circus — supplied-binary Antfly inference benchmarks"
	@echo
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo 'Examples:'
	@echo '  make prereqs PROFILE=smoke ANTFLY_BIN=/absolute/path/to/antfly'
	@echo '  make smoke ANTFLY_BIN=/absolute/path/to/antfly'
	@echo '  make full ANTFLY_BIN=/absolute/path/to/antfly QUIET_HOST_ACK=1'
	@echo '  make category CATEGORY=extraction ANTFLY_BIN=/absolute/path/to/antfly'

fixtures: ## Generate deterministic OCR, multimodal, and speech fixtures under results/
	$(PYTHON) scripts/make_fixtures.py

prereqs: ## Report missing binary, model, fixture, and provider prerequisites
	$(CIRCUS) prereqs $(COMMON_ARGS)

models: ## Explicitly pull the selected profile's Antfly models (networked)
	$(CIRCUS) models $(COMMON_ARGS)

reference-models: ## Explicitly cache selected Hugging Face reference models (networked)
	$(REFERENCE_PYTHON) scripts/cache_reference_models.py --profile "$(PROFILE)" $(if $(CATEGORY),--category "$(CATEGORY)",)

setup-tools: ## Explicitly create the pinned local reference-provider environment
	uv sync --project "$(CURDIR)" --python 3.12

smoke: PROFILE=smoke
smoke: fixtures ## Run one supplied Antfly binary across every working inference role
	$(CIRCUS) prereqs $(COMMON_ARGS)
	$(CIRCUS) run $(COMMON_ARGS) $(if $(RUN_ID),--run-id "$(RUN_ID)",) $(if $(OUTPUT),--output "$(OUTPUT)",) $(if $(ANTFLY_SUBJECT_ID),--subject-id "$(ANTFLY_SUBJECT_ID)",)

full: PROFILE=full
full: fixtures ## Run the full local-provider matrix with three fresh lifecycles
	$(CIRCUS) prereqs $(COMMON_ARGS)
	$(CIRCUS) run $(COMMON_ARGS) $(if $(RUN_ID),--run-id "$(RUN_ID)",) $(if $(OUTPUT),--output "$(OUTPUT)",) $(if $(ANTFLY_SUBJECT_ID),--subject-id "$(ANTFLY_SUBJECT_ID)",)

category: fixtures ## Run one category; set CATEGORY and optionally PROFILE=full
	@test -n "$(CATEGORY)" || { echo 'CATEGORY is required' >&2; exit 2; }
	$(CIRCUS) prereqs $(COMMON_ARGS)
	$(CIRCUS) run $(COMMON_ARGS) $(if $(RUN_ID),--run-id "$(RUN_ID)",) $(if $(OUTPUT),--output "$(OUTPUT)",) $(if $(ANTFLY_SUBJECT_ID),--subject-id "$(ANTFLY_SUBJECT_ID)",)

validate: ## Validate a campaign bundle; set RESULT=results/campaigns/<id>
	@test -n "$(RESULT)" || { echo 'RESULT is required' >&2; exit 2; }
	$(CIRCUS) validate "$(RESULT)"

test: ## Run deterministic harness tests without models or an Antfly binary
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) -m py_compile scripts/circus.py scripts/make_fixtures.py scripts/cache_reference_models.py providers/reference.py
