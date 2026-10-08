#!/usr/bin/env bash
# Copy the settings in your local .env into GitHub Actions secrets and variables,
# so the weekly workflow (.github/workflows/weekly_litreview.yml) can run.
#
# Requirements: the GitHub CLI (`brew install gh`) signed in with `gh auth login`.
# Usage:        scripts/setup_github_secrets.sh [owner/repo]
#
# Values are piped to `gh` on stdin and never printed. Empty values are skipped.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="${1:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
ENV_FILE="${ENV_FILE:-.env}"

if [ ! -f "$ENV_FILE" ]; then
  echo "No $ENV_FILE found — copy .env.example to .env and fill it in first." >&2
  exit 1
fi

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

# Credentials and personal data → encrypted secrets
SECRETS=(
  ANTHROPIC_API_KEY LANGSMITH_API_KEY
  SEMANTIC_SCHOLAR_API_KEY OPENALEX_API_KEY OPENALEX_EMAIL
  ELICIT_API_KEY SCITE_API_KEY SCHOLARCY_API_KEY PERPLEXITY_API_KEY CORE_API_KEY
  NCBI_EMAIL NCBI_API_KEY
  LITREVIEW_GOOGLE_DOC_ID PROPOSAL_GOOGLE_DOC_ID AI_FAILURES_GOOGLE_DOC_ID
  SMTP_USER SMTP_PASSWORD SENDGRID_API_KEY EMAIL_FROM EMAIL_TO EMAIL_CC
)
# Non-sensitive tuning → plain repository variables (visible in Settings)
VARIABLES=(
  ANTHROPIC_MODEL ANTHROPIC_FALLBACK_MODEL ANTHROPIC_EFFORT
  EMAIL_BACKEND SMTP_HOST SMTP_PORT MAX_PAPERS_PER_RUN MIN_RELEVANCE_SCORE LLM_CONCURRENCY
)
# DAYS_LOOKBACK is deliberately not copied: the workflow uses 21 days for weekly runs.

env_value() {
  "$PYTHON" - "$ENV_FILE" "$1" <<'PY'
import sys
from dotenv import dotenv_values
print(dotenv_values(sys.argv[1]).get(sys.argv[2]) or "", end="")
PY
}

echo "Configuring $REPO from $ENV_FILE"
for name in "${SECRETS[@]}"; do
  value="$(env_value "$name")"
  if [ -n "$value" ]; then
    printf '%s' "$value" | gh secret set "$name" --repo "$REPO" >/dev/null
    echo "  secret   $name"
  fi
done
for name in "${VARIABLES[@]}"; do
  value="$(env_value "$name")"
  if [ -n "$value" ]; then
    gh variable set "$name" --repo "$REPO" --body "$value" >/dev/null
    echo "  variable $name=$value"
  fi
done

SA_FILE="$(env_value GOOGLE_SERVICE_ACCOUNT_FILE)"
SA_FILE="${SA_FILE:-config/google_service_account.json}"
if [ -f "$SA_FILE" ]; then
  gh secret set GOOGLE_SERVICE_ACCOUNT_JSON --repo "$REPO" < "$SA_FILE" >/dev/null
  echo "  secret   GOOGLE_SERVICE_ACCOUNT_JSON (from $SA_FILE)"
else
  echo "  (no service account file at $SA_FILE — Google Docs sync stays off in GitHub Actions)"
fi

if [ -z "$(env_value ANTHROPIC_API_KEY)" ]; then
  echo "WARNING: ANTHROPIC_API_KEY is empty — the workflow will stop at its first step." >&2
fi
if [ -z "$(env_value OPENALEX_API_KEY)" ]; then
  echo "Note: no OPENALEX_API_KEY — OpenAlex rejects keyless requests from GitHub's shared IPs (keys are free)."
fi
echo "Done. Start a test run with:  gh workflow run weekly_litreview.yml --repo $REPO -f dry_run=true"
