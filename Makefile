.PHONY: install install-minimal test test-serve bench bench-full bench-chart serve lint parity-all ci-test clean help

P ?= moderncolbert

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install vllm-factory with all deps
	pip install -e ".[gliner]"
	pip install vllm
	@python -m vllm_factory.compat.doctor

install-minimal: ## Install without GLiNER deps
	pip install -e .
	pip install vllm
	@python -m vllm_factory.compat.doctor

install-legacy: ## Install pinned to vLLM 0.15.1 with pooling patch
	pip install -e ".[gliner]"
	pip install "vllm==0.15.1"
	python -m forge.patches.pooling_extra_kwargs

patch: ## Apply legacy vLLM 0.15.x pooling patch (only needed if doctor says so)
	python -m forge.patches.pooling_extra_kwargs

doctor: ## Run environment diagnostics
	python -m vllm_factory.compat.doctor

test: ## Run offline parity test for a plugin (P=name)
	python plugins/$(P)/parity_test.py

test-serve: ## Run end-to-end serve parity test (P=name or all)
	@if [ "$(P)" = "all" ]; then \
		python scripts/serve_parity_test.py; \
	else \
		python scripts/serve_parity_test.py --plugin $(P); \
	fi

test-all: ## Run end-to-end serve parity test for all 12 plugins
	python scripts/serve_parity_test.py

bench: ## Run per-plugin benchmark (P=name)
	python plugins/$(P)/benchmark.py

bench-full: ## Run unified benchmark suite (all registered plugins + charts)
	python -m bench run --all
	python -m bench chart --results bench/results/ --output bench/charts/

bench-chart: ## Generate charts from existing results
	python -m bench chart --results bench/results/ --output bench/charts/

bench-one: ## Run unified benchmark for one plugin (P=name)
	python -m bench run --plugin $(P)
	python -m bench chart --results bench/results/ --output bench/charts/

serve: ## Serve a plugin with IOProcessor (P=name, PORT=8000)
	@echo "Serving $(P) on port $${PORT:-8000}..."
	@IO_PLUGIN=$$(python -c "import importlib.metadata; eps = importlib.metadata.entry_points(group='vllm.io_processor_plugins'); matches = [e for e in eps if '$(P)' in e.name]; print(matches[0].name if matches else '')" 2>/dev/null); \
	MODEL=$$(python -c "from plugins.$(P) import MODEL_NAME; print(MODEL_NAME)" 2>/dev/null || echo ""); \
	if [ -z "$$MODEL" ]; then echo "Could not detect model for $(P). Use vllm serve manually."; exit 1; fi; \
	echo "  Model: $$MODEL"; \
	echo "  IOProcessor: $$IO_PLUGIN"; \
	vllm serve $$MODEL --runner pooling --trust-remote-code --enforce-eager \
		--no-enable-prefix-caching --no-enable-chunked-prefill \
		--io-processor-plugin $$IO_PLUGIN --port $${PORT:-8000}

lint: ## Run ruff linter
	ruff check forge/ plugins/ kernels/ poolers/ vllm_factory/ --select E,F,I,W --ignore E501

ci-test: ## Run local CI-equivalent checks
	ruff check forge/ plugins/ kernels/ poolers/ vllm_factory/ --select E,F,I,W --ignore E501
	python -m compileall forge models poolers kernels plugins vllm_factory

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
