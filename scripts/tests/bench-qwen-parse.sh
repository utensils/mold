#!/usr/bin/env bash
#
# Contract test for scripts/bench-qwen.sh.
#
# Feeds canned `mold run` stderr transcripts (ANSI colours and progress bar
# carriage returns included, exactly as the CLI emits them) through the
# harness's parse/row functions and asserts the extracted JSON fields, drives
# check_gates over canned NDJSON rows, and runs the whole matrix against a stub
# mold binary so the plan and the run must agree on row cardinality. Nothing
# here needs a GPU, a model, or a real mold binary.
#
# Usage: scripts/tests/bench-qwen-parse.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
harness="$repo_root/scripts/bench-qwen.sh"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

failures=0

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

assert_json() {
  local label="$1" json="$2" filter="$3"
  if jq -e "$filter" <<<"$json" >/dev/null; then
    return 0
  fi
  fail "$label: expected \`$filter\` on $json"
}

assert_eq() {
  local label="$1" actual="$2" expected="$3"
  [[ "$actual" == "$expected" ]] || fail "$label: expected '$expected', got '$actual'"
}

command -v jq >/dev/null 2>&1 || {
  echo "error: jq is required" >&2
  exit 2
}
[[ -x "$harness" ]] || {
  echo "error: harness not executable: $harness" >&2
  exit 2
}

# Sourcing must not run the matrix.
# shellcheck source=../bench-qwen.sh disable=SC1091
source "$harness"

esc=$'\033'
cr=$'\r'

# --- cold CFG-on run: every stage present -----------------------------------
cat > "$tmp/cold.log" <<EOF
Denoising...${cr}   ${cr}
  ${esc}[32m✓${esc}[0m Loading Qwen2.5 text encoder (4 shards, GPU) ${esc}[2m[35.1s]${esc}[0m
  ${esc}[32m✓${esc}[0m Loading Qwen-Image transformer (quantized) ${esc}[2m[10.4s]${esc}[0m
  ${esc}[32m✓${esc}[0m Denoising (20 steps) ${esc}[2m[72.2s]${esc}[0m
  ${esc}[32m✓${esc}[0m VAE decode ${esc}[2m[0.7s]${esc}[0m
${esc}[32m✓${esc}[0m Done — ${esc}[1mqwen-image-2512:q4${esc}[0m in 119.8s (seed: 42)
EOF

cold="$(parse_transcript "$tmp/cold.log")"
assert_json "cold total_s" "$cold" '.total_s == 119.8'
assert_json "cold denoise_s" "$cold" '.denoise_s == 72.2'
assert_json "cold te_load_s" "$cold" '.te_load_s == 35.1'
assert_json "cold te_reload_s" "$cold" '.te_reload_s == null'
assert_json "cold transformer_load_s" "$cold" '.transformer_load_s == 10.4'
assert_json "cold transformer_reload_s" "$cold" '.transformer_reload_s == null'
assert_json "cold vae_s" "$cold" '.vae_s == 0.7'
assert_json "cold steps" "$cold" '.steps == 20'
assert_json "cold cache_hit" "$cold" '.cache_hit == false'

row="$(bench_row "qwen-image-2512:q4" 1024 1024 20 4.0 ok local matrix 0 false \
  "$cold" abc1234 "mold 0.1.0" "NVIDIA GeForce RTX 4090" "2026-08-14T00:00:00Z")"
assert_json "row model" "$row" '.model == "qwen-image-2512:q4"'
assert_json "row width" "$row" '.width == 1024'
assert_json "row height" "$row" '.height == 1024'
assert_json "row steps" "$row" '.steps == 20'
assert_json "row guidance" "$row" '.guidance == 4.0'
assert_json "row cfg" "$row" '.cfg == true'
assert_json "row status" "$row" '.status == "ok"'
assert_json "row mode" "$row" '.mode == "local"'
assert_json "row role" "$row" '.role == "matrix"'
assert_json "row repeat" "$row" '.repeat == 0'
assert_json "row total_s" "$row" '.total_s == 119.8'
assert_json "row s_per_step" "$row" '.s_per_step == 3.61'
assert_json "row te_load_s" "$row" '.te_load_s == 35.1'
assert_json "row transformer_load_s" "$row" '.transformer_load_s == 10.4'
assert_json "row vae_s" "$row" '.vae_s == 0.7'
assert_json "row warm" "$row" '.warm == false'
assert_json "row git_sha" "$row" '.git_sha == "abc1234"'
assert_json "row mold_version" "$row" '.mold_version == "mold 0.1.0"'
assert_json "row gpu_name" "$row" '.gpu_name == "NVIDIA GeForce RTX 4090"'
assert_json "row timestamp" "$row" '.timestamp == "2026-08-14T00:00:00Z"'
assert_json "row schema" "$row" '
  (keys_unsorted | sort) == ([
    "cache_hit","cfg","denoise_s","git_sha","gpu_name","guidance","height","mode",
    "model","mold_version","repeat","role","s_per_step","status","steps","te_load_s",
    "te_reload_s","timestamp","total_s","transformer_load_s","transformer_reload_s",
    "vae_s","warm","width"
  ] | sort)'

# --- warm server request: cache hit, no text-encoder stage at all ------------
cat > "$tmp/warm.log" <<EOF
  ${esc}[32m✓${esc}[0m prompt conditioning ${esc}[96m[cache hit]${esc}[0m
  ${esc}[32m✓${esc}[0m Denoising (20 steps) [71.4s]
  ${esc}[32m✓${esc}[0m VAE decode [0.6s]
✓ Done — qwen-image-2512:q4 in 84.2s (seed: 42)
EOF

warm="$(parse_transcript "$tmp/warm.log")"
assert_json "warm total_s" "$warm" '.total_s == 84.2'
assert_json "warm te_load_s null" "$warm" '.te_load_s == null'
assert_json "warm te_reload_s null" "$warm" '.te_reload_s == null'
assert_json "warm cache_hit" "$warm" '.cache_hit == true'
assert_json "warm denoise_s" "$warm" '.denoise_s == 71.4'

warm_row="$(bench_row "qwen-image-2512:q4" 1024 1024 20 4.0 ok server probe_warm 1 true \
  "$warm" abc1234 v gpu ts)"
assert_json "warm row warm flag" "$warm_row" '.warm == true'
assert_json "warm row mode" "$warm_row" '.mode == "server"'
assert_json "warm row s_per_step" "$warm_row" '.s_per_step == 3.57'
assert_json "warm row cache_hit" "$warm_row" '.cache_hit == true'

# --- reload: the residency stages are named "Reloading"/"Unparking" ----------
# The pipeline's in-process stage names carry no "text encoder" and no bare
# "Loading" (crates/mold-inference/src/qwen_image/pipeline.rs). A parser that
# only looks for a load stage reports a 3.2 s reload as "no reload happened".
cat > "$tmp/reload.log" <<EOF
  ${esc}[32m✓${esc}[0m Reloading Qwen2.5 encoder [3.2s]
  ${esc}[32m✓${esc}[0m Denoising (20 steps) [70.9s]
  ${esc}[32m✓${esc}[0m VAE decode [0.6s]
✓ Done — qwen-image-2512:q4 in 76.0s (seed: 42)
EOF

reload="$(parse_transcript "$tmp/reload.log")"
assert_json "reload te_reload_s" "$reload" '.te_reload_s == 3.2'
assert_json "reload te_load_s stays null" "$reload" '.te_load_s == null'

cat > "$tmp/unpark.log" <<'EOF'
  ✓ Unparking Qwen2.5 encoder (CPU→GPU) [1.1s]
  ✓ Denoising (20 steps) [70.9s]
✓ Done — qwen-image-2512:q4 in 73.0s (seed: 42)
EOF
unpark="$(parse_transcript "$tmp/unpark.log")"
assert_json "unpark te_reload_s" "$unpark" '.te_reload_s == 1.1'

# --- a cold load and a transformer reload in one run -------------------------
# "Reloading Qwen-Image transformer" contains "loading Qwen-Image transformer",
# so an unanchored pattern reports the reload as the cold load.
cat > "$tmp/both.log" <<'EOF'
  ✓ Loading Qwen2.5 text encoder (4 shards, GPU) [35.1s]
  ✓ Loading Qwen-Image transformer (quantized) [10.4s]
  ✓ Denoising (20 steps) [72.2s]
  ✓ VAE decode [0.7s]
  ✓ Reloading Qwen-Image transformer [4.9s]
✓ Done — qwen-image-2512:q4 in 119.8s (seed: 42)
EOF
both="$(parse_transcript "$tmp/both.log")"
assert_json "both transformer_load_s is the cold load" "$both" '.transformer_load_s == 10.4'
assert_json "both transformer_reload_s" "$both" '.transformer_reload_s == 4.9'
assert_json "both te_load_s" "$both" '.te_load_s == 35.1'

# --- CFG-off run -------------------------------------------------------------
nocfg_row="$(bench_row "qwen-image-2512:q4" 1024 1024 20 1.0 ok local matrix 0 false \
  "$cold" abc1234 v gpu ts)"
assert_json "cfg off" "$nocfg_row" '.cfg == false'

# --- distilled run: steps come from the transcript, not the request ----------
cat > "$tmp/distilled.log" <<'EOF'
  ✓ Loading Qwen2.5 text encoder (4 shards, GPU) [4.9s]
  ✓ Loading Qwen-Image transformer (quantized) [3.1s]
  ✓ Denoising (8 steps) [12.0s]
  ✓ VAE decode [0.6s]
✓ Done — qwen-image-lightning:fp8 in 21.4s (seed: 42)
EOF

distilled="$(parse_transcript "$tmp/distilled.log")"
assert_json "distilled steps" "$distilled" '.steps == 8'
distilled_row="$(bench_row "qwen-image-lightning:fp8" 1024 1024 default 1.0 ok local matrix 0 false \
  "$distilled" abc1234 v gpu ts)"
assert_json "distilled row steps" "$distilled_row" '.steps == 8'
assert_json "distilled row s_per_step" "$distilled_row" '.s_per_step == 1.5'
assert_json "distilled row total_s" "$distilled_row" '.total_s == 21.4'

# --- failed run (OOM): no Done line, nothing invented ------------------------
cat > "$tmp/oom.log" <<'EOF'
  ✓ Loading Qwen2.5 text encoder (4 shards, GPU) [34.8s]
Error: CUDA error: out of memory
EOF

oom="$(parse_transcript "$tmp/oom.log")"
assert_json "oom total_s" "$oom" '.total_s == null'
assert_json "oom denoise_s" "$oom" '.denoise_s == null'
assert_json "oom steps" "$oom" '.steps == null'
assert_json "oom te_load_s" "$oom" '.te_load_s == 34.8'

oom_row="$(bench_row "qwen-image-2512:q8" 1328 1328 20 4.0 oom_or_error local matrix 0 false \
  "$oom" abc1234 v gpu ts)"
assert_json "oom row status" "$oom_row" '.status == "oom_or_error"'
assert_json "oom row steps falls back" "$oom_row" '.steps == 20'
assert_json "oom row s_per_step" "$oom_row" '.s_per_step == null'

# --- skipped model -----------------------------------------------------------
missing="$(parse_transcript /dev/null)"
missing_row="$(bench_row "qwen-image-flash:q4" 1024 1024 default 1.0 model_missing local matrix 0 false \
  "$missing" abc1234 v gpu ts)"
assert_json "missing row status" "$missing_row" '.status == "model_missing"'
assert_json "missing row steps" "$missing_row" '.steps == null'
assert_json "missing row total_s" "$missing_row" '.total_s == null'

# --- git_sha provenance ------------------------------------------------------
# A binary outside any git work tree must not be stamped with the current
# directory's HEAD: the sha exists to attribute a measurement to a revision.
mkdir -p "$tmp/notarepo/bin"
printf '#!/usr/bin/env bash\nexit 0\n' > "$tmp/notarepo/bin/mold"
chmod +x "$tmp/notarepo/bin/mold"
assert_eq "git_sha outside a work tree" \
  "$(detect_git_sha "$tmp/notarepo/bin/mold")" "unknown"
assert_eq "git_sha for a bare PATH name that resolves nowhere" \
  "$(detect_git_sha "definitely-not-a-real-command-9d1f")" "unknown"
if command -v git >/dev/null 2>&1 && git -C "$repo_root" rev-parse HEAD >/dev/null 2>&1; then
  in_tree="$(detect_git_sha "$repo_root/scripts/bench-qwen.sh")"
  [[ "$in_tree" =~ ^[0-9a-f]+(-dirty)?$ ]] \
    || fail "git_sha inside the work tree should be a short sha, got '$in_tree'"
fi

# --- check_gates over canned rows --------------------------------------------
# check_gates is the harness's decision authority and the milestone's stated
# completion signal, so it is asserted directly rather than through a run.
# check_gates reads BENCH_STEPS, rows_file, skip_distilled, reload_probe and
# resets its own failure/skip accumulators, all harness globals.
# shellcheck disable=SC2034
run_gates() {
  local rows_json="$1"
  BENCH_STEPS=20
  rows_file="$tmp/gate-rows.ndjson"
  jq -c '.[]' <<<"$rows_json" > "$rows_file"
  gate_failures=()
  gate_skips=()
  check_gates 2>"$tmp/gate-stderr" && printf 'pass' || printf 'fail'
}

matrix_row() {
  # matrix_row <model> <width> <cfg-guidance> <status> <total_s> [steps]
  jq -cn --arg model "$1" --argjson width "$2" --argjson guidance "$3" \
    --arg status "$4" --argjson total "$5" --argjson steps "${6:-20}" '
    {model: $model, width: $width, height: $width, steps: $steps,
     guidance: $guidance, cfg: ($guidance > 1), status: $status, mode: "local",
     role: "matrix", repeat: 0, total_s: $total, denoise_s: null,
     s_per_step: null, te_load_s: null, te_reload_s: null,
     transformer_load_s: null, transformer_reload_s: null, vae_s: null,
     cache_hit: false, warm: false, git_sha: "abc", mold_version: "v",
     gpu_name: "gpu", timestamp: "ts"}'
}

probe_row() {
  # probe_row <status> <te_reload_s|null> <te_load_s|null>
  jq -cn --arg status "$1" --argjson reload "$2" --argjson load "$3" '
    {model: "qwen-image-2512:q4", width: 1024, height: 1024, steps: 20,
     guidance: 4.0, cfg: true, status: $status, mode: "server",
     role: "probe_reload", repeat: 2, total_s: 84.0, denoise_s: null,
     s_per_step: null, te_load_s: $load, te_reload_s: $reload,
     transformer_load_s: null, transformer_reload_s: null, vae_s: null,
     cache_hit: false, warm: true, git_sha: "abc", mold_version: "v",
     gpu_name: "gpu", timestamp: "ts"}'
}

distilled_probe_row() {
  # distilled_probe_row <model> <role> <status> <total_s|null>
  jq -cn --arg model "$1" --arg role "$2" --arg status "$3" --argjson total "$4" '
    {model: $model, width: 1024, height: 1024, steps: 8, guidance: 1.0,
     cfg: false, status: $status, mode: "server", role: $role,
     repeat: (if $role == "distilled_warm" then 1 else 0 end),
     total_s: $total, denoise_s: null, s_per_step: null, te_load_s: null,
     te_reload_s: null, transformer_load_s: null, transformer_reload_s: null,
     vae_s: null, cache_hit: ($role == "distilled_warm"),
     warm: ($role == "distilled_warm"), git_sha: "abc", mold_version: "v",
     gpu_name: "gpu", timestamp: "ts"}'
}

passing_rows="$(jq -cn \
  --argjson a "$(matrix_row "qwen-image-lightning:fp8" 1024 1.0 ok 61.4 8)" \
  --argjson b "$(matrix_row "qwen-image-2512:q8" 1328 4.0 ok 150.0)" \
  --argjson c "$(matrix_row "qwen-image-2512:q4" 1328 4.0 ok 134.0)" \
  --argjson d "$(probe_row ok 3.2 null)" \
  --argjson e "$(distilled_probe_row "qwen-image-lightning:fp8" distilled_cold ok 61.4)" \
  --argjson f "$(distilled_probe_row "qwen-image-lightning:fp8" distilled_warm ok 18.2)" \
  '[$a, $b, $c, $d, $e, $f]')"

# shellcheck disable=SC2034  # all three are harness globals check_gates reads
skip_distilled=0
# shellcheck disable=SC2034
reload_probe=1
# shellcheck disable=SC2034
probe_distilled=1
assert_eq "gates pass on a fully green run" "$(run_gates "$passing_rows")" "pass"

# (a) is a WARM budget: a cold matrix row that blows the 25 s budget is not
# evidence against it — that row pays a 20 GB checkpoint load.
assert_json "the green fixture's cold distilled row is over the warm budget" \
  "$passing_rows" 'any(.[]; .role == "matrix" and (.model | test("lightning")) and .total_s > 25)'

# (a) a warm distilled request over budget fails and names itself
slow_distilled="$(jq -c '(.[] | select(.role == "distilled_warm")).total_s = 31.0' <<<"$passing_rows")"
assert_eq "gate (a) fails on a slow warm distilled row" "$(run_gates "$slow_distilled")" "fail"
grep -q 'FAIL (a).*qwen-image-lightning:fp8 total_s=31' "$tmp/gate-stderr" \
  || fail "gate (a) failure should name the model and its total_s"

# (a) an installed distilled model that ERRORED is a failure, not a discard —
# the gate promises every installed distilled model finishes in budget.
failed_distilled="$(jq -c '(.[] | select(.role == "distilled_warm")) |= (.status = "oom_or_error" | .total_s = null)' <<<"$passing_rows")"
assert_eq "gate (a) fails on an errored warm distilled row" "$(run_gates "$failed_distilled")" "fail"

# (a) reads ONLY the warm row: the cold probe request pays the weight load and
# must not be able to fail a warm budget.
slow_cold="$(jq -c '(.[] | select(.role == "distilled_cold")).total_s = 90.0' <<<"$passing_rows")"
assert_eq "gate (a) ignores the cold distilled probe row" "$(run_gates "$slow_cold")" "pass"

# (a) is skipped, never failed, when its opt-in flag excluded the rows
no_distilled="$(jq -c '[.[] | select(.model | test("lightning|flash") | not)]' <<<"$passing_rows")"
skip_distilled=1
assert_eq "gate (a) skipped with --skip-distilled" "$(run_gates "$no_distilled")" "pass"
grep -q 'gate skipped: (a)' "$tmp/gate-stderr" \
  || fail "gate (a) should report itself skipped under --skip-distilled"
# shellcheck disable=SC2034
skip_distilled=0

# (a) is likewise skipped when the probe that produces its only measurement was
# never asked for — a cold matrix row is not a substitute.
probe_distilled=0
assert_eq "gate (a) skipped without --probe-distilled" "$(run_gates "$passing_rows")" "pass"
grep -q 'gate skipped: (a).*--probe-distilled' "$tmp/gate-stderr" \
  || fail "gate (a) should name --probe-distilled as what would evaluate it"
# shellcheck disable=SC2034
probe_distilled=1

assert_eq "gate (a) fails when warm distilled rows are simply absent" \
  "$(run_gates "$no_distilled")" "fail"

# (b) an OOM q8 row fails
oom_q8="$(jq -c '(.[] | select(.model == "qwen-image-2512:q8")).status = "oom_or_error"' <<<"$passing_rows")"
assert_eq "gate (b) fails on an OOM q8 row" "$(run_gates "$oom_q8")" "fail"
grep -q 'FAIL (b)' "$tmp/gate-stderr" || fail "gate (b) should fail on an OOM q8 row"

# (c) is measured on cold rows: it must not require a warm flag no run can set
assert_eq "gate (c) passes without any warm row" "$(run_gates "$passing_rows")" "pass"
# 140 s, recalibrated from the milestone's 148.4 s end state. Both sides of the
# boundary are pinned so a re-tightening has to be deliberate.
at_budget="$(jq -c '(.[] | select(.model == "qwen-image-2512:q4" and .width == 1328)).total_s = 140.0' <<<"$passing_rows")"
assert_eq "gate (c) passes exactly at 140" "$(run_gates "$at_budget")" "pass"
slow_1328="$(jq -c '(.[] | select(.model == "qwen-image-2512:q4" and .width == 1328)).total_s = 148.4' <<<"$passing_rows")"
assert_eq "gate (c) fails over budget" "$(run_gates "$slow_1328")" "fail"
grep -q 'FAIL (c)' "$tmp/gate-stderr" || fail "gate (c) should fail over budget"
# a row at another step count is not evidence for the 20-step gate
other_steps="$(jq -c '(.[] | select(.model == "qwen-image-2512:q4" and .width == 1328)).steps = 8' <<<"$passing_rows")"
assert_eq "gate (c) ignores a non-20-step row" "$(run_gates "$other_steps")" "fail"

# (d) reload budget, from the reload stage, the load stage, or a resident
# encoder that ran neither
slow_reload="$(jq -c '(.[] | select(.role == "probe_reload")).te_reload_s = 9.0' <<<"$passing_rows")"
assert_eq "gate (d) fails on a slow reload" "$(run_gates "$slow_reload")" "fail"
grep -q 'FAIL (d)' "$tmp/gate-stderr" || fail "gate (d) should fail on a slow reload"
load_instead="$(jq -c '(.[] | select(.role == "probe_reload")) |= (.te_reload_s = null | .te_load_s = 8.0)' <<<"$passing_rows")"
assert_eq "gate (d) falls back to the load stage" "$(run_gates "$load_instead")" "fail"
resident="$(jq -c '(.[] | select(.role == "probe_reload")) |= (.te_reload_s = null | .te_load_s = null)' <<<"$passing_rows")"
assert_eq "gate (d) passes when the encoder stayed resident" "$(run_gates "$resident")" "pass"
failed_probe="$(jq -c '(.[] | select(.role == "probe_reload")).status = "not_run"' <<<"$passing_rows")"
assert_eq "gate (d) fails when the probe did not run" "$(run_gates "$failed_probe")" "fail"
reload_probe=0
no_probe="$(jq -c '[.[] | select(.role != "probe_reload")]' <<<"$passing_rows")"
assert_eq "gate (d) skipped without --reload-probe" "$(run_gates "$no_probe")" "pass"
grep -q 'gate skipped: (d)' "$tmp/gate-stderr" \
  || fail "gate (d) should report itself skipped without --reload-probe"
# shellcheck disable=SC2034
reload_probe=1

# --- --dry-run smoke: prints the plan, runs no model -------------------------
plan="$(MOLD_BIN=/nonexistent/mold "$harness" --dry-run 2>/dev/null)"
assert_json "plan is an array" "$plan" 'type == "array" and length > 0'
assert_json "plan status" "$plan" 'all(.[]; .status == "planned")'
assert_json "plan q4 1328 cfg" "$plan" 'any(.[]; .model == "qwen-image-2512:q4" and .width == 1328 and .cfg == true)'
assert_json "plan q4 1024 no-cfg" "$plan" 'any(.[]; .model == "qwen-image-2512:q4" and .width == 1024 and .cfg == false)'
assert_json "plan q8 1328" "$plan" 'any(.[]; .model == "qwen-image-2512:q8" and .width == 1328)'
assert_json "plan distilled lightning" "$plan" 'any(.[]; .model == "qwen-image-lightning:fp8")'
assert_json "plan is cold-only" "$plan" 'all(.[]; .warm == false and .mode == "local")'
assert_json "plan seed prompt matrix is 20 steps for q4" "$plan" '
  all(.[] | select(.model | startswith("qwen-image-2512")); .steps == 20)'

repeats_plan="$(MOLD_BIN=/nonexistent/mold "$harness" --dry-run --repeats 3 2>/dev/null)"
if ! jq -e --argjson base "$plan" '(length) == (($base | length) * 3)' <<<"$repeats_plan" >/dev/null; then
  fail "--repeats 3 should plan three times as many rows"
fi

plan_probe="$(MOLD_BIN=/nonexistent/mold "$harness" --dry-run --reload-probe 2>/dev/null)"
if ! jq -e --argjson base "$plan" '(length) == (($base | length) + 3)' <<<"$plan_probe" >/dev/null; then
  fail "the reload probe should add exactly three server rows to the plan"
fi
assert_json "probe roles" "$plan_probe" '
  [.[] | select(.mode == "server") | .role] == ["probe_cold","probe_warm","probe_reload"]'
assert_json "probe cold row is not warm" "$plan_probe" '
  all(.[] | select(.role == "probe_cold"); .warm == false)'
assert_json "probe warm rows are warm" "$plan_probe" '
  all(.[] | select(.role == "probe_warm" or .role == "probe_reload"); .warm == true)'

# --probe-distilled implies the probe server and adds one cold/warm pair per
# distilled model. Gate (a)'s only measurement is the warm half of each pair.
plan_distilled_probe="$(MOLD_BIN=/nonexistent/mold "$harness" --dry-run --probe-distilled 2>/dev/null)"
if ! jq -e --argjson base "$plan_probe" '(length) == (($base | length) + 4)' <<<"$plan_distilled_probe" >/dev/null; then
  fail "--probe-distilled should add a cold/warm pair for each of the two distilled models"
fi
assert_json "--probe-distilled implies the reload probe" "$plan_distilled_probe" '
  any(.[]; .role == "probe_reload")'
assert_json "distilled probe rows are server rows" "$plan_distilled_probe" '
  all(.[] | select(.role == "distilled_cold" or .role == "distilled_warm"); .mode == "server")'
assert_json "exactly one warm row per distilled model" "$plan_distilled_probe" '
  ([.[] | select(.role == "distilled_warm") | .model] | sort)
  == ["qwen-image-flash:q4", "qwen-image-lightning:fp8"]'
assert_json "the distilled warm row is flagged warm and the cold one is not" "$plan_distilled_probe" '
  all(.[] | select(.role == "distilled_warm"); .warm == true)
  and all(.[] | select(.role == "distilled_cold"); .warm == false)'

# --- plan vs real run: identical row cardinality against a stub binary --------
# A plan-vs-result diff is only meaningful if a missing model or an early
# failure still fills its rows in.
stub="$tmp/stub-mold"
cat > "$stub" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  --version) echo "mold 0.0.0-stub"; exit 0 ;;
  info)
    case "$2" in
      qwen-image-lightning:fp8 | qwen-image-flash:q4)
        echo "Status: Not installed"
        exit 1
        ;;
      *)
        echo "Status: Installed"
        exit 0
        ;;
    esac
    ;;
  run)
    # q8 always OOMs, exactly as it does on a 24 GB card today.
    if [[ "$*" == *qwen-image-2512:q8* ]]; then
      echo "  ✓ Loading Qwen2.5 text encoder (4 shards, GPU) [34.8s]" >&2
      echo "Error: CUDA error: out of memory" >&2
      exit 1
    fi
    {
      echo "  ✓ Loading Qwen2.5 text encoder (4 shards, GPU) [35.1s]"
      echo "  ✓ Loading Qwen-Image transformer (quantized) [10.4s]"
      echo "  ✓ Denoising (20 steps) [72.2s]"
      echo "  ✓ VAE decode [0.7s]"
      echo "✓ Done — stub in 119.8s (seed: 42)"
    } >&2
    exit 0
    ;;
esac
exit 0
STUB
chmod +x "$stub"

stub_plan="$("$harness" --mold-bin "$stub" --dry-run --repeats 2 2>/dev/null)"
stub_run="$("$harness" --mold-bin "$stub" --out-dir "$tmp/stub-out" --repeats 2 2>/dev/null)"
plan_shape="$(jq -c '[.[] | {model, width, guidance, repeat}] | sort' <<<"$stub_plan")"
run_shape="$(jq -c '[.[] | {model, width, guidance, repeat}] | sort' <<<"$stub_run")"
assert_eq "the run records exactly the planned rows" "$run_shape" "$plan_shape"
assert_json "a missing distilled model still fills its repeats" "$stub_run" '
  ([.[] | select(.model == "qwen-image-flash:q4")] | length) == 2
  and all(.[] | select(.model == "qwen-image-flash:q4"); .status == "model_missing")'
assert_json "a failed config records the repeats it never ran" "$stub_run" '
  ([.[] | select(.model == "qwen-image-2512:q8") | .status] | sort)
  == ["not_run","oom_or_error"]'
assert_json "the stub run records the binary version" "$stub_run" '
  all(.[]; .mold_version == "mold 0.0.0-stub")'
assert_json "matrix rows are never labelled warm" "$stub_run" '
  all(.[]; .warm == false)'
[[ -s "$tmp/stub-out/rows.ndjson" ]] || fail "rows.ndjson should be written as the run goes"
assert_eq "rows.ndjson holds one row per line" \
  "$(wc -l < "$tmp/stub-out/rows.ndjson" | tr -d ' ')" \
  "$(jq 'length' <<<"$stub_run")"

# --gates on the stub run must fail loudly rather than crash
if "$harness" --mold-bin "$stub" --out-dir "$tmp/stub-gates" --gates >/dev/null 2>"$tmp/stub-gate-stderr"; then
  fail "--gates should fail on a run where the milestone work has not landed"
fi
grep -q 'FAIL (b)' "$tmp/stub-gate-stderr" || fail "stub --gates should fail gate (b)"
grep -q 'gate skipped: (d)' "$tmp/stub-gate-stderr" \
  || fail "stub --gates should skip gate (d) without --reload-probe"
if "$harness" --mold-bin "$stub" --out-dir "$tmp/stub-gates2" --gates --skip-distilled \
  >/dev/null 2>"$tmp/stub-gate-stderr2"; then
  fail "--gates should still fail gate (b) with --skip-distilled"
fi
if grep -q 'FAIL (a)' "$tmp/stub-gate-stderr2"; then
  fail "--skip-distilled must not turn gate (a) into a failure"
fi
grep -q 'gate skipped: (a)' "$tmp/stub-gate-stderr2" \
  || fail "--skip-distilled should report gate (a) as skipped"

if [[ "$failures" -ne 0 ]]; then
  echo "bench-qwen-parse: $failures assertion(s) failed" >&2
  exit 1
fi

echo "bench-qwen-parse: ok"
