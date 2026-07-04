#!/usr/bin/env bash
# Hook 1 — Pre-commit quality gate.
#
# Fires on PreToolUse(Bash) filtered to `git commit`. Runs the full quality
# gate (make check = lint + typecheck + test) and BLOCKS the commit if any
# stage fails. This stops the agent from committing code it has not validated.
#
# Contract: reads the tool-call JSON on stdin, emits a PreToolUse decision on
# stdout. Silent (exit 0, no output) = defer to normal permission flow.
set -uo pipefail

input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')

# Only gate real commits. Ignore other git verbs and no-op commit queries.
case "$cmd" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac
case "$cmd" in
  *"--dry-run"*) exit 0 ;;
esac

root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$root" 2>/dev/null || exit 0

out=$(make check 2>&1)
status=$?

if [ "$status" -ne 0 ]; then
  tail=$(printf '%s' "$out" | tail -n 30)
  reason=$(printf 'Pre-commit gate FAILED — `make check` (lint + typecheck + test) did not pass. Fix the following before committing:\n\n%s' "$tail")
  jq -n --arg r "$reason" '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'
  exit 0
fi

# Passed — surface a note but defer the commit itself to the normal flow.
jq -n '{systemMessage:"Pre-commit gate passed (lint + typecheck + test)."}'
exit 0
