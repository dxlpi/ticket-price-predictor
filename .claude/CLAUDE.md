# ticket-price-predictor

ML system for predicting secondary-market ticket prices at the seat-zone level. See root `CLAUDE.md` for detailed project structure and patterns.

Good code guides readers naturally — structure reveals intent without requiring explanation.

**Build Commands**:
```bash
make check    # lint + typecheck + test (full validation)
make test     # pytest only
make format   # auto-fix lint + format
make lint     # ruff check
make typecheck # mypy strict
```

**Key Documentation**:
- `docs/ARCHITECTURE.md` — Layer diagram, dependency rules, modification policy
- `docs/DEV_GUIDE.md` — Build commands, workflow, conventions
- `docs/model-card.md` — Model performance, limitations, training data
- `docs/data-dictionary.md` — Schema definitions, field types, storage layout

Rules in `.claude/rules/` are auto-loaded. Domain-specific rules activate based on file paths being edited via `paths:` frontmatter.

## Workflow

**Before**: Read `docs/ARCHITECTURE.md` and `docs/DEV_GUIDE.md`. Identify required agent consultations from matrix in `.claude/rules/agents.md`.

**During**: Invoke domain agents per consultation matrix. Follow source tree policy and layer dependency rules.

**After Implementation** (strict order, fail-fast by cost):

**Scope gate**: Steps 1-4 apply only when source-affecting files are modified (source code, build config, dependencies). Non-source changes (docs, agent definitions, config prose) skip entirely.

1. **Lint** — `make lint`
2. **Review Gate** — run review-orchestrator. BLOCKING items must pass before build.
3. **Build** — `make typecheck`
4. **Test** — `make test`. All tests must pass before declaring complete.
