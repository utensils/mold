#!/usr/bin/env bash
#
# Contract test for scripts/bench-qwen21.sh.
#
# Feeds canned `mold run` stderr transcripts (ANSI colours and carriage
# returns included, exactly as the CLI emits them) with Qwen Image 2.1's stage
# names through the harness's parser and row builder, checks the median
# summary and the gates over canned rows, and runs the whole matrix against a
# stub mold binary so the plan and the run must agree on row cardinality.
# Nothing here needs a GPU, a model, a server, or a real mold binary.
#
# Usage: scripts/tests/bench-qwen21-parse.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
harness="$repo_root/scripts/bench-qwen21.sh"

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

command -v jq >/dev/null 2>&1 || {
  echo "error: jq is required" >&2
  exit 2
}
[[ -x "$harness" ]] || {
  echo "error: harness not executable: $harness" >&2
  exit 2
}

# Sourcing must not run the matrix.
# shellcheck source=../bench-qwen21.sh disable=SC1091
source "$harness"

esc=$'\033'
cr=$'\r'

# --- cold server request: every Qwen Image 2.1 stage present ----------------
cat > "$tmp/cold.log" <<EOF
Denoising...${cr}   ${cr}
  ${esc}[32m✓${esc}[0m Loading Qwen Image 2.1 transformer (2 shards) ${esc}[2m[3.8s]${esc}[0m
  ${esc}[32m✓${esc}[0m Loading Qwen Image 2.1 VAE (GPU) ${esc}[2m[0.2s]${esc}[0m
  ${esc}[32m✓${esc}[0m Loading Qwen3-VL text encoder (4 shards, GPU) ${esc}[2m[3.3s]${esc}[0m
  ${esc}[32m✓${esc}[0m Encoding prompt (Qwen3-VL) ${esc}[2m[0.1s]${esc}[0m
  ${esc}[32m✓${esc}[0m Denoising (40 steps) ${esc}[2m[38.9s]${esc}[0m
  ${esc}[32m✓${esc}[0m VAE decode ${esc}[2m[1.1s]${esc}[0m
${esc}[32m✓${esc}[0m Done — ${esc}[1mqwen-image-2.1:bf16${esc}[0m in 40.2s (seed: 210001)
EOF

cold="$(parse_transcript21 "$tmp/cold.log")"
assert_json "cold total_s" "$cold" '.total_s == 40.2'
assert_json "cold denoise_s" "$cold" '.denoise_s == 38.9'
assert_json "cold encode_s" "$cold" '.encode_s == 0.1'
assert_json "cold te_load_s" "$cold" '.te_load_s == 3.3'
assert_json "cold transformer_load_s" "$cold" '.transformer_load_s == 3.8'
assert_json "cold vae_s" "$cold" '.vae_s == 1.1'
assert_json "cold steps" "$cold" '.steps == 40'
assert_json "cold cache_hit" "$cold" '.cache_hit == false'

row="$(bench21_row t2i-1024 qwen-image-2.1:bf16 1024 1024 1 0 timed 2 ok "$cold" \
  abc1234 "mold 0.32.0" "NVIDIA L40S" "2026-09-26T00:00:00Z")"
assert_json "row case" "$row" '.case == "t2i-1024"'
assert_json "row model" "$row" '.model == "qwen-image-2.1:bf16"'
assert_json "row steps" "$row" '.steps == 40'
assert_json "row negative" "$row" '.negative_prompt == false'
assert_json "row s_per_step" "$row" '.s_per_step == 0.972'
assert_json "row role/repeat" "$row" '.role == "timed" and .repeat == 2'
assert_json "row schema" "$row" '
  (keys_unsorted | sort) == ([
    "cache_hit","case","denoise_s","encode_s","git_sha","gpu_name","guidance",
    "height","model","mold_version","negative_prompt","repeat","role","s_per_step",
    "status","steps","te_load_s","timestamp","total_s","transformer_load_s","vae_s","width"
  ] | sort)'

# --- warm request: cached conditioning, no load stages at all ---------------
cat > "$tmp/warm.log" <<EOF
  ${esc}[32m✓${esc}[0m prompt conditioning ${esc}[96m[cache hit]${esc}[0m
  ${esc}[32m✓${esc}[0m Denoising (40 steps) [21.3s]
  ${esc}[32m✓${esc}[0m VAE decode [0.4s]
✓ Done — qwen-image-2.1:bf16 in 21.9s (seed: 210001)
EOF
warm="$(parse_transcript21 "$tmp/warm.log")"
assert_json "warm te_load_s null" "$warm" '.te_load_s == null'
assert_json "warm transformer_load_s null" "$warm" '.transformer_load_s == null'
assert_json "warm cache_hit" "$warm" '.cache_hit == true'
assert_json "warm denoise_s" "$warm" '.denoise_s == 21.3'

# --- a failed request: nothing inferred --------------------------------------
printf 'error: CUDA out of memory\n' > "$tmp/fail.log"
failed="$(parse_transcript21 "$tmp/fail.log")"
assert_json "failed timings null" "$failed" '.total_s == null and .denoise_s == null and .vae_s == null'

# --- matrix shape -------------------------------------------------------------
cases="$(bench21_cases bf16)"
[[ "$(wc -l <<<"$cases")" -eq 4 ]] || fail "bf16 matrix must have 4 cases, got: $cases"
cases="$(bench21_cases "bf16, int8-conv,q4")"
[[ "$(wc -l <<<"$cases")" -eq 6 ]] || fail "two extra tiers must add two cases, got: $cases"
grep -q '^tier-int8-conv-1024|qwen-image-2.1:int8-conv|1024|1024|1|0|$' <<<"$cases" \
  || fail "tier rows are 1024² g1 with no gate: $cases"
grep -q '^cfg-1344x768|qwen-image-2.1:bf16|1344|768|4|1|42$' <<<"$cases" \
  || fail "the guided row carries the negative prompt and the 42 s gate: $cases"

# --- summary medians and gates over canned rows ------------------------------
rows="$tmp/rows.ndjson"
: > "$rows"
add_row() { # case width height guidance negative role repeat status denoise
  bench21_row "$1" qwen-image-2.1:bf16 "$2" "$3" "$4" "$5" "$6" "$7" "$8" \
    "{\"denoise_s\": $9, \"total_s\": $9, \"steps\": 40}" sha v gpu ts >> "$rows"
}
add_row t2i-1024 1024 1024 1 0 warmup 0 ok 30.0
add_row t2i-1024 1024 1024 1 0 timed 1 ok 21.0
add_row t2i-1024 1024 1024 1 0 timed 2 ok 23.0
add_row t2i-1024 1024 1024 1 0 timed 3 ok 20.0
add_row cfg-1344x768 1344 768 4 1 timed 1 ok 43.0
add_row cfg-1344x768 1344 768 4 1 timed 2 ok 44.0
add_row t2i-2048 2048 2048 1 0 timed 1 error null
add_row t2i-2752x1536 2752 1536 1 0 timed 1 ok 80.0
summary="$(bench21_summary "$rows")"
assert_json "median excludes warmup" "$summary" '.[] | select(.case == "t2i-1024") | .denoise_s_median == 21.0 and .timed_ok == 3'
assert_json "even median" "$summary" '.[] | select(.case == "cfg-1344x768") | .denoise_s_median == 43.5'
assert_json "failed rows are not medians" "$summary" '.[] | select(.case == "t2i-2048") | .denoise_s_median == null and .timed_total == 1'
gate_log="$tmp/gates.log"
if bench21_check_gates "$summary" 2> "$gate_log"; then
  fail "gates must fail on a slow guided row and a missing 2048 row"
fi
grep -q 'GATE FAIL: cfg-1344x768 median denoise 43.5s > 42s' "$gate_log" || fail "guided gate: $(cat "$gate_log")"
grep -q 'GATE FAIL: t2i-2048 has no successful timed row' "$gate_log" || fail "missing gate: $(cat "$gate_log")"
grep -q 'gate ok: t2i-1024' "$gate_log" || fail "1024 gate should pass: $(cat "$gate_log")"
grep -q 'gate ok: t2i-2752x1536' "$gate_log" || fail "2752 gate should pass: $(cat "$gate_log")"

# --- dry run and a stubbed real run agree on row cardinality -----------------
plan="$("$harness" --dry-run --repeats 2 --tiers bf16,q8 --out-dir "$tmp/plan")"
assert_json "plan cardinality" "$plan" 'length == 15'
assert_json "plan rows are planned" "$plan" 'all(.status == "planned")'
assert_json "plan warmups" "$plan" '[.[] | select(.role == "warmup")] | length == 5'

stub="$tmp/mold"
cat > "$stub" <<'STUB'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then echo "mold 0.0.0-stub"; exit 0; fi
# A 2752 request "fails" so the run records an error row instead of a gap.
for arg in "$@"; do [[ "$arg" == "2752" ]] && { echo "error: out of memory" >&2; exit 1; }; done
printf '  ✓ Denoising (40 steps) [20.5s]\n  ✓ VAE decode [0.3s]\n✓ Done — qwen-image-2.1:bf16 in 21.0s (seed: 1)\n' >&2
STUB
chmod +x "$stub"
run="$("$harness" --host http://127.0.0.1:1 --mold-bin "$stub" --repeats 2 --tiers bf16,q8 \
  --out-dir "$tmp/run" 2>/dev/null)"
run_rows="$(jq -s '.' "$tmp/run/rows.ndjson")"
assert_json "run cardinality equals plan" "$run_rows" 'length == 15'
assert_json "run errors recorded" "$run_rows" '[.[] | select(.case == "t2i-2752x1536" and .status == "error")] | length == 3'
assert_json "run summary" "$run" '.[] | select(.case == "t2i-1024") | .denoise_s_median == 20.5 and .timed_ok == 2'
if "$harness" --host http://127.0.0.1:1 --mold-bin "$stub" --repeats 1 --gates \
  --out-dir "$tmp/gated" >/dev/null 2>&1; then
  fail "--gates must exit non-zero when a gated case has no successful row"
fi

if [[ "$failures" -ne 0 ]]; then
  echo "bench-qwen21-parse: $failures failure(s)" >&2
  exit 1
fi
echo "bench-qwen21-parse: ok"
