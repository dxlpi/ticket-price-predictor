# Agent Guardrail Hooks

Automated, non-bypassable checks that run **before** the agent performs risky actions.
Wired in [`.claude/settings.json`](../settings.json) under `hooks.PreToolUse`; the AI
cannot skip them because the harness — not the model — executes them.

| # | Hook | Fires on | Effect |
|---|------|----------|--------|
| ① | **Pre-commit gate** — [`pre-commit-gate.sh`](pre-commit-gate.sh) | `Bash` matching `git commit` | Runs `make check` (lint + typecheck + test). **Blocks the commit** if any stage fails. |
| ② | **Bias-free PR review** — agent hook (inline) | `Bash` matching `git push` | An independent sub-agent reviews `git diff @{u}..HEAD` against the project's non-negotiables (leakage, layer invariant, secrets, tests) and **blocks the push** on any critical finding. Separate reviewer = no author bias. |
| ③ | **Test-first guard** — [`test-first-guard.sh`](test-first-guard.sh) | `Write` / `Edit` to `src/ticket_price_predictor/**.py` | **Blocks the write** unless a matching test exists (`tests/test_<module>.py`). Forces test-before-code. |
| ④ | **Incident deny-list** — [`incident-guard.sh`](incident-guard.sh) + [`incident-denylist.json`](incident-denylist.json) | `Bash` / `Write` / `Edit` | Matches the call against patterns distilled from past production incidents and **blocks** on a hit, citing the incident ID. Acts as an always-on senior reviewer. |

## Contract

Each command hook reads the tool-call JSON on **stdin** and, to block, prints:

```json
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"..."}}
```

Silent (`exit 0`, no output) = allow / defer to normal permission flow. Scripts resolve
the repo via `$CLAUDE_PROJECT_DIR` (falling back to `git rev-parse`).

## Extending the incident deny-list

`incident-denylist.json` is data-driven — add a rule, no code change:

```json
{
  "id": "INC-013",
  "scope": "bash",              // "bash" (matches the command) or "content" (matches file path + written content)
  "pattern": "regex",           // POSIX ERE, case-insensitive (grep -iE). Use [(] for a literal paren; avoid \\b.
  "message": "why this is blocked — reference the incident"
}
```

Patterns are passed to `grep -iE` verbatim (raw via `jq -r`, so backslashes survive —
do **not** use `@tsv`/`@csv`, which double them). Validate after editing:

```bash
jq empty .claude/hooks/incident-denylist.json
```

## Managing / disabling

- Review or toggle any hook interactively via the `/hooks` menu.
- Emergency global off: set `"disableAllHooks": true` in settings, or run with `--no-hooks`.
- A deliberate one-off exception should be made by a **human**, not by weakening a rule.
