# Developer Guide

## Prerequisites

- Python >=3.11
- uv (package manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Playwright browsers: `uv run playwright install chromium`
- API keys in `.env`: `TICKETMASTER_API_KEY`, `LASTFM_API_KEY`

## Build Commands

All commands via Makefile:

| Command | What It Does |
|---------|-------------|
| `make check` | Lint (ruff) + typecheck (mypy --strict) + test (pytest) |
| `make test` | Run pytest |
| `make test-cov` | Run pytest with coverage report (HTML output in htmlcov/) |
| `make lint` | Run ruff check + format check |
| `make typecheck` | Run mypy with strict mode + pydantic plugin |
| `make format` | Auto-fix lint issues + auto-format with ruff |
| `make pipeline` | Run full data collection pipeline (ingest + collect via `scripts/run_pipeline.py`) |
| `make clean` | Remove build artifacts, caches, coverage files |

### Script Commands

```bash
# Data collection
python scripts/collect_listings.py --artist "Bruno Mars" --max-events 3
python scripts/ingest_events.py --event-types concert --cities "Las Vegas"
python scripts/monitor_popular.py  # Hourly monitor (15 popular artists)

# Model training
python scripts/train_model.py --model lightgbm --version v29
python scripts/train_model.py --from-study lightgbm_aggressive --version v30

# Hyperparameter tuning
python scripts/tune_model.py --n-trials 50

# Predictions
python scripts/predict.py --artist "BTS" --city "Tampa" --all-zones

# Preprocessing
python scripts/preprocess_data.py
```

## Workflow

### Before Implementation
1. Read `docs/ARCHITECTURE.md` for layer diagram and dependency rules
2. Identify which layer your change affects
3. Check `.claude/rules/agents.md` consultation matrix for required agent reviews

### During Implementation
1. Follow dependency rules (no upward imports)
2. New feature extractors: implement `FeatureExtractor` ABC with `fit()` / `extract()` / `feature_names`
3. Schema changes: update Pydantic model AND `parquet_schema()` classmethod
4. ML changes: preserve split-before-fit invariant in `trainer.py`

### After Implementation
1. `make lint` — ruff check passes
2. `make typecheck` — mypy strict passes
3. `make test` — all pytest tests pass
4. Run review-orchestrator for quality gate

## Conventions

### Code Style
- **Formatter**: ruff (rules: E, W, F, I, B, C4, UP, ARG, SIM)
- **Type checker**: mypy strict mode with pydantic plugin
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Imports**: isort via ruff (I rules), absolute imports preferred

### Commits
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- Include scope when clear: `feat(features): add venue capacity bucket`

### Testing
- Framework: pytest with pytest-asyncio (asyncio_mode = "auto")
- Coverage: pytest-cov with branch coverage
- Test files: `tests/test_<module>.py`
- Fixtures: `tests/conftest.py` provides sample events from `tests/fixtures/`
- Pattern: in-memory DataFrames, no Parquet I/O in unit tests

### Data
- Storage: Hive-partitioned Parquet (`year=YYYY/month=MM/day=DD/`)
- Deduplication: MD5 hash of composite keys
- Models: `data/models/lightgbm_v{N}.joblib` + `_metrics.json`
