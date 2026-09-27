#!/usr/bin/env bash
# Contract for scripts/capture-qwen21-metal-uat.sh (#1770): the plan covers
# every checklist item, escalates memory (2K presets last, FP8 expected to be
# refused on Metal), and the memory guard cannot be dropped silently.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/capture-qwen21-metal-uat.sh"

fail() {
  echo "qwen21-metal-uat contract: $*" >&2
  exit 1
}

bash -n "$script" || fail "syntax error"
plan="$(QWEN21_UAT_PLAN=1 bash "$script")"
ids="$(cut -f1 <<<"$plan")"

for id in tier-q4 tier-q8 tier-int8-conv tier-fp8 t2i-bf16-v032 t2i-bf16-attn-math \
  turbo-bf16 webp-still transparent-toggle alpha-reference opaque-reference turbo-lora \
  refs-1 refs-3 refs-10 2k-square-q4 2k-square-bf16 2k-wide-bf16; do
  grep -qx "$id" <<<"$ids" || fail "plan lacks $id"
done

[[ "$(awk -F'\t' '$1 == "tier-fp8" {print $3}' <<<"$plan")" == refused ]] \
  || fail "FP8 must be expected refused on Metal (no F8 cast kernel)"
grep -q -- '--format webp' <<<"$(awk -F'\t' '$1 == "webp-still"' <<<"$plan")" \
  || fail "webp-still must request WebP"
grep -q -- '--transparent' <<<"$(awk -F'\t' '$1 == "transparent-toggle"' <<<"$plan")" \
  || fail "transparent-toggle must send --transparent"

# Memory escalation: nothing at 2K may run before the last 1024² case.
first_2k="$(grep -n '^2k-' <<<"$plan" | head -1 | cut -d: -f1)"
last_other="$(grep -vn '^2k-' <<<"$plan" | tail -1 | cut -d: -f1)"
((first_2k > last_other)) || fail "2K presets must run last"

# Selecting cases narrows the plan.
[[ "$(QWEN21_UAT_PLAN=1 bash "$script" tier-q4 refs-3 | cut -f1 | tr '\n' ' ')" == "tier-q4 refs-3 " ]] \
  || fail "named cases must select exactly those cases, in matrix order"

# The guard and the no-server rule are load-bearing.
grep -q 'memory_pressure' "$script" || fail "memory watchdog removed"
grep -q -- '--local' "$script" || fail "renders must stay --local"
grep -q 'for busy in cargo rustc xcodebuild swift-frontend' "$script" || fail "busy-process preflight removed"

echo "qwen21-metal-uat contract: ok"
