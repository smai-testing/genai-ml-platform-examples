#!/usr/bin/env bash
#
# check-naming.sh
#
# Naming-consistency linter for the GenAI/ML Standardization workshop.
#
# Enforces a SINGLE source of truth for the workshop project name:
#   * The one canonical literal is CANONICAL (default: bank-marketing-prediction).
#   * Notebooks and code must resolve the name from the environment, e.g.
#         PROJECT_NAME = os.environ.get('PROJECT_NAME', 'bank-marketing-prediction')
#     so the ONLY place the literal is allowed to appear is as that fallback.
#   * No other project-name literal may appear anywhere in source
#     (e.g. 'smai-project-a', 'fraud-detection-monitoring').
#   * A hardcoded  PROJECT_NAME = '<literal>'  assignment (i.e. NOT read from the
#     environment) is a failure even if the literal is the canonical one.
#
# It deliberately ignores:
#   * Jupyter notebook cell OUTPUTS (stale execution logs), linting only the
#     "source" arrays of code/markdown cells.
#   * Files explicitly marked dead (Do_Not_use_*).
#   * This script itself and the .git directory.
#
# Exit status: 0 = clean, 1 = violations found.
#
# Usage:
#   scripts/check-naming.sh                 # from repo root
#   CANONICAL=my-project scripts/check-naming.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CANONICAL="${CANONICAL:-bank-marketing-prediction}"

# Stale/legacy project-name literals that must never appear in source.
FORBIDDEN_LITERALS=(
  "smai-project-a"
  "smai-project-b"
  "fraud-detection-monitoring"
)

red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[1;33m%s\033[0m\n' "$*"; }

# Files to scan: tracked source only. Skip the dead template, this script,
# and vendored/venv paths. (Read into an array in a bash 3.2-compatible way —
# macOS ships bash 3.2, which has no `mapfile`.)
FILES=()
while IFS= read -r _line; do
  [ -n "$_line" ] && FILES+=("$_line")
done < <(
  git ls-files -- \
    '*.ipynb' '*.py' '*.yaml' '*.yml' '*.md' '*.sh' 2>/dev/null \
  | grep -vE 'Do_Not_use|scripts/check-naming\.sh$' || true
)

if [ "${#FILES[@]}" -eq 0 ]; then
  yellow "No source files found to scan (are you in a git repo?)."
  exit 0
fi

violations=0

# For notebooks, extract only the "source" lines (strip cell outputs) with a
# tiny python helper; for everything else, cat the file. Emits "path:content"
# so grep line context is preserved by our own numbering below.
scan_source() {
  local f="$1"
  case "$f" in
    *.ipynb)
      python3 - "$f" <<'PY'
import json, sys
path = sys.argv[1]
try:
    nb = json.load(open(path))
except Exception as e:
    # Unparseable notebook is itself a problem.
    print(f"::PARSE_ERROR:: {e}")
    sys.exit(0)
for ci, cell in enumerate(nb.get("cells", [])):
    for li, line in enumerate(cell.get("source", [])):
        # Fake a 1-based "cell.line" locator so failures are findable.
        print(f"[cell{ci}] {line.rstrip(chr(10))}")
PY
      ;;
    *)
      cat "$f"
      ;;
  esac
}

echo "Naming linter — canonical value: '${CANONICAL}'"
echo "Scanning ${#FILES[@]} tracked source files..."
echo

for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  src="$(scan_source "$f")"

  if grep -q "::PARSE_ERROR::" <<<"$src"; then
    red "FAIL  $f  (notebook JSON did not parse)"
    violations=$((violations+1))
    continue
  fi

  # 1) Forbidden legacy literals anywhere in source.
  for lit in "${FORBIDDEN_LITERALS[@]}"; do
    if hits="$(grep -nF "$lit" <<<"$src")"; then
      red "FAIL  $f  contains forbidden project-name literal '$lit':"
      sed 's/^/        /' <<<"$hits"
      violations=$((violations+1))
    fi
  done

  # 2) Hardcoded PROJECT_NAME / _project_name assignment that is NOT read from
  #    the environment. The env patterns are allowed:
  #      Python:  os.environ.get('PROJECT_NAME', ...)  /  os.getenv(...)
  #      Shell:   PROJECT_NAME="${PROJECT_NAME:-default}"  (parameter expansion)
  #    Match:  PROJECT_NAME = 'literal'   or   _project_name = "literal"
  #    but NOT a quoted variable expansion like  PROJECT_NAME='${PROJECT_NAME}'
  #    (which appears inside shell strings / error messages, not as a hardcode).
  if hits="$(grep -nE "(^|[^A-Za-z_])_?[Pp][Rr][Oo][Jj][Ee][Cc][Tt]_?[Nn][Aa][Mm][Ee] *= *['\"][^\$]" <<<"$src" \
              | grep -vE "os\.environ\.get|os\.getenv|\\\$\{[A-Za-z_]+:-" || true)"; then
    if [ -n "$hits" ]; then
      red "FAIL  $f  hardcodes a project name instead of reading the environment:"
      sed 's/^/        /' <<<"$hits"
      red "      Use: PROJECT_NAME = os.environ.get('PROJECT_NAME', '${CANONICAL}')"
      violations=$((violations+1))
    fi
  fi
done

echo
if [ "$violations" -eq 0 ]; then
  green "OK — naming is consistent. Single source of truth: '${CANONICAL}' (env-overridable)."
  exit 0
else
  red "FOUND ${violations} naming violation(s). See above."
  exit 1
fi
