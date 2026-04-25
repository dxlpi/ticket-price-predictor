# Git Worktree + Shared Venv Import Isolation
Promoted: 2026-03-23 | Updated: 2026-03-23

## Rule
Git worktrees that share a venv import from the *installed* package location (via `.pth` file), not the worktree's `src/`. Changes in the worktree are silently ignored by tests unless you update the `.pth` file or copy new files to the installed src.

## Why
All test failures in a worktree may pass locally in main — because tests run against the stale installed source, not the worktree edits. A v33 regression investigation wasted cycles before discovering this.

## Pattern
Fix option A — update .pth to point at worktree:
```bash
# Find installed location
python -c "import ticket_price_predictor; print(ticket_price_predictor.__file__)"
# Update .pth file at:
# .venv/lib/python3.12/site-packages/_ticket_price_predictor.pth
# Change its content to: /path/to/worktree/src
# REVERT after merging
```

Fix option B — copy new files to installed src:
```bash
cp src/new_module.py ../main-worktree/src/ticket_price_predictor/new_module.py
```

Fix option C (preferred for agents) — use `uv pip install -e .` inside the worktree to create an isolated venv.
