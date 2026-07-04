# Agent-Readiness Evals

A lightweight harness that measures how reliably an automated coding agent can work in
this repo. Each **task** is a verifiable acceptance check; the headline metric is the
**pass-rate**. Run it after any agent-driven change to catch regressions in the repo's
agent-facing contracts.

```bash
python evals/run_evals.py          # run all tasks, refresh agent-results.json
make evals                          # same, via Makefile
```

## What it checks (v0)

Tasks are declared in [`tasks.json`](tasks.json) as `(id, category, description, command)`,
where `command` is an argv array run from the repo root. Current suite:

| Task | Guards against |
|------|----------------|
| `doc_paths_resolve` | Hallucinated / stale file paths in context docs |
| `layer_invariant_ml_isolation` | Lower layers importing `ml/` (leakage-risk coupling) |
| `preprocessing_module_compiles` | Syntax breakage in the preprocessing pipeline |
| `issue_log_fully_indexed` | Decision-log entries missing from `docs/issues/README.md` |
| `memory_index_points_to_stores` | `MEMORY.md` losing its link to the decision store |
| `architecture_has_mermaid` | Losing the visual dependency/data-flow diagram |

## Output

[`run_evals.py`](run_evals.py) writes [`agent-results.json`](agent-results.json) — a
timestamped snapshot with per-task results and the pass-rate. Keep it committed as a
baseline; a drop below `threshold` (default 1.0) exits non-zero so it can gate CI.

## Extending

This is a structural + guard eval. The JSON schema already carries per-task results, so a
natural next step is a **task-replay** eval: record real agent tasks (prompt → produced
diff), then score whether each diff passes the repo's checks — turning pass-rate into a
true agent task-success metric wired to session/agent logs.
