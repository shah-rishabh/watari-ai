.DEFAULT_GOAL := help
.PHONY: help install check lint format typecheck test evals run serve chat docker clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync deps (dev extras) and install pre-commit hooks
	uv sync --extra dev
	uv run pre-commit install

check: lint format-check typecheck test ## Run the full CI gate locally

lint: ## Lint with ruff
	uv run ruff check .

format: ## Format with ruff
	uv run ruff format .

format-check: ## Check formatting without changing files
	uv run ruff format --check .

typecheck: ## Strict type check with pyright
	uv run pyright

test: ## Run unit tests
	uv run pytest tests/unit -q

evals: ## Run the full eval suites (needs a local model)
	uv run watari evals run --suite all

run: serve ## Alias for `serve`

serve: ## Start the FastAPI server
	uv run watari serve

chat: ## Start the interactive chat REPL
	uv run watari chat

docker: ## Build the Docker image
	docker build -t watari:local .

clean: ## Remove caches and build artifacts
	rm -rf .ruff_cache .pytest_cache dist build eval-results
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
