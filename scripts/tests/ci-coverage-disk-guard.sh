#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
workflow="$repo_root/.github/workflows/ci.yml"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

coverage_job="$(sed -n '/^  coverage:/,$p' "$workflow")"

cleanup_line="$(grep -nF -- '- name: Free runner disk space' <<< "$coverage_job" | cut -d: -f1 || true)"
coverage_line="$(grep -nF -- '- name: Generate coverage' <<< "$coverage_job" | cut -d: -f1 || true)"

[[ -n "$cleanup_line" ]] || fail "coverage job does not free runner disk space"
[[ -n "$coverage_line" ]] || fail "coverage job does not generate coverage"
((cleanup_line < coverage_line)) || fail "runner disk cleanup must happen before coverage generation"

grep -Fq '/usr/local/lib/android' <<< "$coverage_job" \
  || fail "coverage cleanup does not remove the preinstalled Android SDK"
grep -Fq 'df -h' <<< "$coverage_job" \
  || fail "coverage cleanup does not report disk capacity"

echo "CI coverage disk guard checks passed"
