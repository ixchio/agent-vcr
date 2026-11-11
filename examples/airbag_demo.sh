#!/usr/bin/env bash
# Agent VCR airbag demo — no agent, API key or account needed.
#
# Simulates what happens when a coding agent runs `rm -rf` on your project:
# the hook payload below is exactly what Claude Code sends before a Bash call.
#
#   pip install ai-agent-vcr
#   bash examples/airbag_demo.sh
set -euo pipefail

command -v vcr >/dev/null || { echo "Install first: pip install ai-agent-vcr"; exit 1; }

WORK="$(mktemp -d)"
export AGENT_VCR_AIRBAG_HOME="$WORK/airbag-store"   # keep the demo self-contained
PROJECT="$WORK/myapp"
trap 'rm -rf "$WORK"' EXIT

step() { printf '\n\033[1m▶ %s\033[0m\n' "$*"; }

step "A project with an uncommitted fix and a .env file"
mkdir -p "$PROJECT/src" && cd "$PROJECT"
git init -q -b main && git config user.email demo@example.com && git config user.name demo
printf 'def total(xs):\n    return sum(xs)\n' > src/billing.py
printf 'STRIPE_KEY=sk_test_123\n' > .env && printf '.env\n' > .gitignore
git add -A && git commit -qm "billing v1"
printf 'def total(xs):\n    return round(sum(xs), 2)  # uncommitted fix\n' > src/billing.py
git log --oneline && git status --short

step "Arm the airbag for Claude Code"
vcr airbag install claude --project-dir "$PROJECT"
HOOK="$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["hooks"]["PreToolUse"][0]["hooks"][0]["command"])' .claude/settings.json)"

step "The agent decides to 'clean up' and runs: rm -rf $PROJECT"
printf '{"session_id":"demo","cwd":"%s","hook_event_name":"PreToolUse","tool_name":"Bash","tool_use_id":"t1","tool_input":{"command":"rm -rf %s"}}' \
  "$PROJECT" "$PROJECT" | sh -c "$HOOK"
echo
cd "$WORK" && rm -rf "$PROJECT"
ls "$PROJECT" 2>&1 || true

step "vcr undo"
vcr undo --project-dir "$PROJECT"

step "Everything is back — code, .env and git history"
cd "$PROJECT"
cat src/billing.py
echo ".env: $(cat .env)"
git log --oneline
git status --short

step "And a command a snapshot cannot undo is stopped for a human"
vcr airbag check "psql -h prod-db.internal -c 'DROP TABLE invoices'"
