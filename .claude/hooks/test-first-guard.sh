#!/usr/bin/env bash
# Hook 3 — Test-first guard.
#
# Fires on PreToolUse(Write|Edit). If the target is a Python source module under
# src/ticket_price_predictor/ that has NO corresponding test, the write is
# BLOCKED — forcing a test to exist (write it first) before the source lands.
#
# Convention (see .claude/rules/conventions.md): tests/test_<module>.py.
set -uo pipefail

input=$(cat)
fp=$(printf '%s' "$input" | jq -r '.tool_input.file_path // ""')
[ -z "$fp" ] && exit 0

root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

# Only guard Python source modules in the package.
case "$fp" in
  *"/src/ticket_price_predictor/"*.py) ;;
  *) exit 0 ;;
esac

base=$(basename "$fp")
# Exempt package plumbing that carries no independently testable logic.
case "$base" in
  __init__.py|__main__.py|_version.py|conftest.py) exit 0 ;;
esac

module="${base%.py}"

# Primary: the conventional test file exists.
[ -f "$root/tests/test_${module}.py" ] && exit 0

# Fallback: some existing test references this module by name (word-bounded),
# so already-tested modules with non-standard test filenames are not blocked.
if grep -rqiE "(^|[^a-zA-Z0-9_])${module}([^a-zA-Z0-9_]|$)" "$root/tests" 2>/dev/null; then
  exit 0
fi

reason=$(printf 'Test-first guard: no test found for module "%s".\nCreate tests/test_%s.py (or a test that exercises %s) BEFORE writing this source file.\nProject convention: tests/test_<module>.py — every changed source file must have a corresponding test.' "$base" "$module" "$module")
jq -n --arg r "$reason" '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'
exit 0
