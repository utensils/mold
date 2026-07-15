#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
workflow="$repo_root/.github/workflows/ci.yml"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

extract_coverage_job() {
  awk '
    /^  coverage:$/ { in_coverage = 1; next }
    in_coverage && /^  [[:alnum:]_-]+:$/ { exit }
    in_coverage { print }
  ' "$1"
}

coverage_job="$(extract_coverage_job "$workflow")"

cleanup_line="$(grep -m1 -nF -- '- name: Free runner disk space' <<< "$coverage_job" | cut -d: -f1 || true)"
coverage_line="$(grep -m1 -nF -- '- name: Generate coverage' <<< "$coverage_job" | cut -d: -f1 || true)"

[[ -n "$cleanup_line" ]] || fail "coverage job does not free runner disk space"
[[ -n "$coverage_line" ]] || fail "coverage job does not generate coverage"
((cleanup_line < coverage_line)) || fail "runner disk cleanup must happen before coverage generation"

grep -Fq '/usr/local/lib/android' <<< "$coverage_job" \
  || fail "coverage cleanup does not remove the preinstalled Android SDK"
grep -Fq 'df -h' <<< "$coverage_job" \
  || fail "coverage cleanup does not report disk capacity"

fixture="$(mktemp)"
trap 'rm -f "$fixture"' EXIT
{
  cat "$workflow"
  printf '%s\n' '  future-job:' '    steps:' '      - name: Future job sentinel'
} > "$fixture"
if extract_coverage_job "$fixture" | grep -Fq 'Future job sentinel'; then
  fail "coverage job extraction includes subsequent workflow jobs"
fi

echo "CI coverage disk guard checks passed"
