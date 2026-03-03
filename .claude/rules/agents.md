# Agent Quick Reference

## Consultation Matrix

| Changed Path | Required Agents | Priority |
|-------------|----------------|----------|
| `ml/training/**` | leakage-guardian, code-critic | MANDATORY |
| `ml/features/**` | leakage-guardian, code-critic | MANDATORY |
| `schemas/**` | schema-validator, code-critic | MANDATORY |
| `storage/**` | schema-validator, data-quality-checker | MANDATORY |
| `scrapers/**` | data-quality-checker, code-critic | MANDATORY |
| `ingestion/**` | data-quality-checker, code-critic | MANDATORY |
| `docs/**` | doc-critic | MANDATORY |
| `tests/**` | test-critic | MANDATORY |
| Any source file | code-critic | MANDATORY |
| Before merge | review-orchestrator | MANDATORY |

## Agent Inventory

| Agent | Tier | Model | Domain |
|-------|------|-------|--------|
| review-orchestrator | 0 | opus | All — final validation gate |
| leakage-guardian | 1 | opus | ML — data leakage prevention |
| schema-validator | 2 | sonnet | Data — Pydantic/Parquet schema safety |
| data-quality-checker | 2 | sonnet | Data — scraper output, completeness |
| code-critic | 3 | sonnet | All — code quality (4-dimension rubric) |
| doc-critic | 3 | sonnet | All — documentation quality |
| test-critic | 3 | sonnet | All — test quality (5-dimension rubric) |

## Invocation Order

1. Tier 1 (safety): leakage-guardian — if BLOCKING, stop
2. Tier 2 (domain): schema-validator, data-quality-checker
3. Tier 3 (quality): code-critic, test-critic, doc-critic
4. Tier 0 (orchestrator): review-orchestrator consolidates all findings
