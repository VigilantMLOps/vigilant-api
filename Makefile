.PHONY: help up down logs dev db-init db-reset db-status seed seed-snapshot db-dump db-restore migration init-baseline test

-include .env
export

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up: ## Start all services (postgres, clickhouse, api) in the background
	docker compose up --build

down: ## Stop and remove all containers
	docker compose down

logs: ## Tail logs for all services (or SERVICE=api for one)
	docker compose logs -f $(SERVICE)

dev: ## Start the FastAPI backend with hot reload (requires running databases)
	poetry run uvicorn main:app --reload

seed: ## Seed the DB: evaluate-data → evaluate-model → evaluate-drift (×3 batches). Use ARGS="--skip <stage>" to skip stages
	poetry run python scripts/seed.py $(ARGS)

seed-snapshot: seed db-dump ## Seed locally, then snapshot the volumes for transfer to another host

db-dump: ## Snapshot the postgres+clickhouse volumes to a dir (stops services briefly). Usage: make db-dump [DIR=./snapshots]
	bash scripts/db_snapshot.sh dump $(DIR)

db-restore: ## Restore postgres+clickhouse volumes from a snapshot. Usage: make db-restore [DIR=./snapshots] [FORCE=1]
	bash scripts/db_snapshot.sh restore $(DIR)

migration: ## Create a new migration. Usage: make migration NAME=add_new_table
	poetry run python scripts/migration.py $(NAME)

db-init: ## Apply any pending migrations (idempotent)
	poetry run python -m core.db_manager init

db-reset: ## Drop all tables and re-apply all migrations from scratch
	poetry run python -m core.db_manager reset

db-status: ## Show applied migration history and pending versions
	poetry run python -m core.db_manager status

init-baseline: ## Compute feature baselines from a training file. Usage: make init-baseline INPUT=/path/to/training.csv
	@test -n "$(INPUT)" || (echo "ERROR: INPUT is required. Usage: make init-baseline INPUT=/path/to/data.csv" && exit 1)
	poetry run python scripts/init_baseline.py --input $(INPUT)

test: ## Run unit and integration tests
	poetry install --with dev --quiet && poetry run pytest