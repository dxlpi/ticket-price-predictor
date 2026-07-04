#!/usr/bin/env bash
# Hook 4 — Incident deny-list ("senior engineer" guard).
#
# Fires on PreToolUse(Bash|Write|Edit). Matches the tool call against
# .claude/hooks/incident-denylist.json — a data-driven list of patterns
# distilled from past production incidents — and BLOCKS on a hit, citing the
# incident id so the reason is auditable. Extend the JSON to add new rules; no
# code change needed.
set -uo pipefail

input=$(cat)
root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
denylist="$root/.claude/hooks/incident-denylist.json"
[ -f "$denylist" ] || exit 0

tool=$(printf '%s' "$input" | jq -r '.tool_name // ""')

case "$tool" in
  Bash)
    haystack=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')
    scope="bash"
    # A commit MESSAGE is data, not an executed command — text that merely
    # describes a risky op (e.g. writing "--no-verify" or "curl | sh" in the
    # message) must not self-trip a rule. For `git commit`, strip the heredoc
    # BODY and any -m "..." message before matching. Everything else — including
    # commands chained after the heredoc terminator — is preserved and scanned.
    if printf '%s' "$haystack" | grep -qE 'git[[:space:]]+commit'; then
      opener=$(printf '%s' "$haystack" | grep -oE "<<-?[[:space:]]*[\"']?[A-Za-z_][A-Za-z0-9_]*" | head -1)
      if [ -n "$opener" ]; then
        delim=$(printf '%s' "$opener" | sed -E "s/^<<-?[[:space:]]*[\"']?//")
        # Drop lines from the opener's body up to (and incl.) the terminator;
        # keep the opener line and resume scanning after the terminator.
        haystack=$(printf '%s' "$haystack" | awk -v d="$delim" '
          inb == 1 { if ($0 ~ ("^[[:space:]]*" d "[[:space:]]*$")) inb=0; next }
          { print }
          $0 ~ ("<<-?[[:space:]]*[\"\047]?" d) { inb=1 }
        ')
      fi
      # Remove only the quoted -m message (bounded to its quotes — never touches
      # quoted args of other, chained commands).
      haystack=$(printf '%s' "$haystack" | sed -E "s/-m[[:space:]]+\"[^\"]*\"//g; s/-m[[:space:]]+'[^']*'//g")
    fi
    ;;
  Write|Edit|MultiEdit)
    fp=$(printf '%s' "$input" | jq -r '.tool_input.file_path // ""')
    body=$(printf '%s' "$input" | jq -r '[.tool_input.content, .tool_input.new_string] | map(select(. != null)) | join("\n")')
    haystack=$(printf '%s\n%s' "$fp" "$body")
    scope="content"
    ;;
  *) exit 0 ;;
esac

# Emit each rule as three raw lines (id, pattern, message). `jq -r` does NOT
# escape backslashes (unlike @tsv/@csv), so regex patterns like `\.parquet`
# survive intact. The while loop runs in the current shell (here-string, not a
# pipe) so an inner `exit` actually terminates the hook and the deny reaches
# Claude Code.
rows=$(jq -r --arg scope "$scope" '.[] | select(.scope == $scope) | .id, .pattern, .message' "$denylist")

while IFS= read -r id && IFS= read -r pattern && IFS= read -r msg; do
  [ -z "$pattern" ] && continue
  if printf '%s' "$haystack" | grep -iqE -- "$pattern"; then
    reason=$(printf '[%s] BLOCKED by incident deny-list: %s\n\n(matched rule: /%s/)\nThis guardrail exists because of a past incident. If this is a deliberate exception, a human must make the change.' "$id" "$msg" "$pattern")
    jq -n --arg r "$reason" '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'
    exit 0
  fi
done <<< "$rows"

exit 0
