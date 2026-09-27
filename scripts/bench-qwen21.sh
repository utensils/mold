#!/usr/bin/env bash
#
# bench-qwen21.sh — Qwen Image 2.1 server end-to-end benchmark.
#
# Runs a fixed request matrix against one RUNNING `mold serve` (normally the
# scratch server on the assigned GPU), parses the CLI's own stage timings out
# of stderr with the transcript parser `scripts/bench-qwen.sh` already owns
# (and `scripts/tests/bench-qwen-parse.sh` pins), appends one JSON row per
# request to <out-dir>/rows.ndjson as it goes, and prints a JSON summary with
# the MEDIAN of each case on stdout at the end.
#
# It is the end-to-end half of the Qwen Image 2.1 CUDA qualification. The
# kernel half — per-forward timings, peak device memory, per-mode ablations —
# is the ignored `official_cuda_mode_benchmark` test in
# crates/mold-inference/src/qwen_image21/transformer/performance_tests/cuda.rs.
# Record both in docs/qualification/qwen-image-2.1-cuda-performance.{md,json}.
#
# USAGE
#   scripts/bench-qwen21.sh --host http://127.0.0.1:7681 [options]
#
#   --host URL        The server to benchmark. Required (except --dry-run).
#   --mold-bin PATH   mold client binary. Default: $MOLD_BIN, else `mold`.
#   --out-dir DIR     Where transcripts, images and rows.ndjson land.
#                     Default: ${TMPDIR:-/tmp}/qwen21-bench-<timestamp>
#   --repeats N       Timed requests per case. Default 3.
#   --seed N          Default 210001.
#   --tiers LIST      Comma-separated tiers; every tier but bf16 adds one
#                     1024² row. Default: bf16.
#   --gates           After running, assert the M5 CUDA gates on the medians
#                     and exit non-zero listing the failures.
#   --dry-run         Print the planned rows as JSON and exit. Runs nothing,
#                     needs no mold binary, no server, no GPU.
#   -h, --help        This help.
#
# MATRIX (40 steps, one warm-up request per case, then --repeats timed ones)
#   qwen-image-2.1:bf16   1024x1024  guidance 1
#   qwen-image-2.1:bf16   1344x768   guidance 4 + negative "blurry, lowres, watermark"
#   qwen-image-2.1:bf16   2048x2048  guidance 1
#   qwen-image-2.1:bf16   2752x1536  guidance 1
#   qwen-image-2.1:<tier> 1024x1024  guidance 1, per extra --tiers entry
#
# WARM, not cold: the first request of each case is a `warmup` row (weights
# load, cuDNN plans its algorithms) and is excluded from the medians. The
# server's prompt-conditioning cache may hit on the repeats; that is the warm
# steady state a user sees, and it touches only the encode stage, not
# `denoise_s`, which is what the gates read.
#
# EXIT GATES (--gates, median denoise_s of the timed rows; L40S BF16 targets)
#   1024x1024 g1            <= 22 s
#   1344x768 g4 + negative  <= 42 s
#   2048x2048 g1            <= 85 s
#   2752x1536 g1            <= 90 s
# A case with no successful timed row fails its gate.
set -euo pipefail

BENCH21_MODEL_BASE="qwen-image-2.1"
BENCH21_STEPS=40
BENCH21_PROMPT='Straight-on editorial photograph of a tiny artisan bakery on a quiet European corner, deep teal facade with three arched windows and a striped awning. A hand-painted sign above the door reads "MOLD & FLOUR" in cream serif capitals. A vintage delivery bicycle rests at the right edge. Sunny spring morning, crisp realistic detail, balanced symmetrical composition.'
BENCH21_FOX_PROMPT='A red fox standing in fresh snow at the edge of a birch forest, winter morning light, detailed fur, wildlife photograph'
BENCH21_NEGATIVE='blurry, lowres, watermark'

bench21_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The transcript parser and row helpers are bench-qwen.sh's; sourcing it does
# not run its matrix (it guards main on BASH_SOURCE).
# shellcheck source=bench-qwen.sh disable=SC1091
source "$bench21_script_dir/bench-qwen.sh"

bench21_die() {
  echo "error: $*" >&2
  exit 2
}

bench21_usage() {
  sed -n '2,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}" | sed -e 's/^#\{0,1\} \{0,1\}//' -e '$d'
}

# One row per line: case|model|width|height|guidance|negative(0/1)|gate_s
bench21_cases() {
  local tiers="$1" tier
  printf '%s\n' "t2i-1024|${BENCH21_MODEL_BASE}:bf16|1024|1024|1|0|22"
  printf '%s\n' "cfg-1344x768|${BENCH21_MODEL_BASE}:bf16|1344|768|4|1|42"
  printf '%s\n' "t2i-2048|${BENCH21_MODEL_BASE}:bf16|2048|2048|1|0|85"
  printf '%s\n' "t2i-2752x1536|${BENCH21_MODEL_BASE}:bf16|2752|1536|1|0|90"
  IFS=',' read -r -a tier_list <<<"$tiers"
  for tier in "${tier_list[@]}"; do
    tier="${tier// /}"
    [[ -n "$tier" && "$tier" != "bf16" ]] || continue
    printf '%s\n' "tier-${tier}-1024|${BENCH21_MODEL_BASE}:${tier}|1024|1024|1|0|"
  done
}

# parse_transcript21 <stderr-log> -> {total_s, denoise_s, encode_s, te_load_s,
#                                     transformer_load_s, vae_s, steps, cache_hit}
# Stage names are Qwen Image 2.1's (crates/mold-inference/src/qwen_image21/
# pipeline.rs): "Loading Qwen3-VL text encoder", "Encoding prompt (Qwen3-VL)",
# "Loading Qwen Image 2.1 transformer", "Denoising (N steps)", "VAE decode".
# Every field is null when the transcript does not contain it.
parse_transcript21() {
  local transcript="$1"
  local total denoise encode te_load transformer vae steps cache_hit
  total="$(transcript_total_seconds "$transcript")"
  denoise="$(stage_seconds "$transcript" 'Denoising' last)"
  encode="$(stage_seconds "$transcript" 'Encoding prompt [(]Qwen3-VL[)]' last)"
  te_load="$(stage_seconds "$transcript" '(^|[^A-Za-z])Loading Qwen3-VL text encoder' first)"
  transformer="$(stage_seconds "$transcript" '(^|[^A-Za-z])Loading Qwen Image 2[.]1 transformer' first)"
  vae="$(stage_seconds "$transcript" 'VAE decode' last)"
  steps="$(transcript_steps "$transcript")"
  cache_hit=false
  if transcript_has_cache_hit "$transcript"; then
    cache_hit=true
  fi
  jq -cn \
    --argjson total_s "$(json_num "$total")" \
    --argjson denoise_s "$(json_num "$denoise")" \
    --argjson encode_s "$(json_num "$encode")" \
    --argjson te_load_s "$(json_num "$te_load")" \
    --argjson transformer_load_s "$(json_num "$transformer")" \
    --argjson vae_s "$(json_num "$vae")" \
    --argjson steps "$(json_num "$steps")" \
    --argjson cache_hit "$cache_hit" \
    '{
      total_s: $total_s,
      denoise_s: $denoise_s,
      encode_s: $encode_s,
      te_load_s: $te_load_s,
      transformer_load_s: $transformer_load_s,
      vae_s: $vae_s,
      steps: $steps,
      cache_hit: $cache_hit
    }'
}

# bench21_row <case> <model> <width> <height> <guidance> <negative 0/1> <role>
#             <repeat> <status> <parsed-json> <git_sha> <mold_version>
#             <gpu_name> <timestamp>
bench21_row() {
  local case_name="$1" model="$2" width="$3" height="$4" guidance="$5" negative="$6"
  local role="$7" repeat="$8" status="$9" parsed="${10}" git_sha="${11}"
  local mold_version="${12}" gpu_name="${13}" ts="${14}"
  jq -cn \
    --arg case "$case_name" \
    --arg model "$model" \
    --arg role "$role" \
    --arg status "$status" \
    --arg git_sha "$git_sha" \
    --arg mold_version "$mold_version" \
    --arg gpu_name "$gpu_name" \
    --arg timestamp "$ts" \
    --argjson width "$width" \
    --argjson height "$height" \
    --argjson guidance "$guidance" \
    --argjson negative "$([[ "$negative" == 1 ]] && echo true || echo false)" \
    --argjson repeat "$repeat" \
    --argjson steps_req "$BENCH21_STEPS" \
    --argjson parsed "$parsed" \
    '
    (($parsed.steps // $steps_req)) as $steps
    | {
      case: $case,
      model: $model,
      width: $width,
      height: $height,
      steps: $steps,
      guidance: $guidance,
      negative_prompt: $negative,
      role: $role,
      repeat: $repeat,
      status: $status,
      total_s: $parsed.total_s,
      denoise_s: $parsed.denoise_s,
      s_per_step: (
        if $parsed.denoise_s != null and $steps != null and $steps > 0
        then (($parsed.denoise_s / $steps) * 1000 | round) / 1000
        else null
        end
      ),
      encode_s: $parsed.encode_s,
      te_load_s: $parsed.te_load_s,
      transformer_load_s: $parsed.transformer_load_s,
      vae_s: $parsed.vae_s,
      cache_hit: $parsed.cache_hit,
      git_sha: $git_sha,
      mold_version: $mold_version,
      gpu_name: $gpu_name,
      timestamp: $timestamp
    }'
}

bench21_record() {
  local row
  row="$(bench21_row "$@" "$git_sha" "$mold_version" "$gpu_name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
  printf '%s\n' "$row" >> "$rows_file"
  jq -r '"\(.case) \(.role)#\(.repeat) \(.status) denoise=\(.denoise_s // "—")s total=\(.total_s // "—")s"' <<<"$row" >&2
}

# Planned rows for a matrix: one warm-up and --repeats timed rows per case,
# in execution order, every one "planned". The real run records exactly these
# rows (status filled in), so the two can be diffed by cardinality.
bench21_emit_plan() {
  local line case_name model width height guidance negative gate i
  while IFS='|' read -r case_name model width height guidance negative gate; do
    bench21_record "$case_name" "$model" "$width" "$height" "$guidance" "$negative" \
      warmup 0 planned '{}'
    for ((i = 1; i <= repeats; i++)); do
      bench21_record "$case_name" "$model" "$width" "$height" "$guidance" "$negative" \
        timed "$i" planned '{}'
    done
  done < <(bench21_cases "$tiers")
}

bench21_run_request() {
  local case_name="$1" model="$2" width="$3" height="$4" guidance="$5" negative="$6"
  local role="$7" repeat="$8"
  local label="${case_name}-${role}-${repeat}"
  local transcript="$out_dir/$label.stderr.log"
  local prompt="$BENCH21_PROMPT" status=ok exit_code=0 parsed
  [[ "$negative" != 1 ]] || prompt="$BENCH21_FOX_PROMPT"
  local args=("run" "$model" "$prompt" "--no-expand"
    "--width" "$width" "--height" "$height" "--steps" "$BENCH21_STEPS"
    "--guidance" "$guidance" "--seed" "$seed" "-o" "$out_dir/$label.png")
  [[ "$negative" != 1 ]] || args+=("--negative-prompt" "$BENCH21_NEGATIVE")
  set +e
  MOLD_HOST="$host" "$mold_bin" "${args[@]}" > "$out_dir/$label.stdout.log" 2> "$transcript"
  exit_code=$?
  set -e
  parsed="$(parse_transcript21 "$transcript")"
  if [[ "$exit_code" -ne 0 ]] || [[ "$(jq -r '.denoise_s' <<<"$parsed")" == "null" ]]; then
    status=error
  fi
  bench21_record "$case_name" "$model" "$width" "$height" "$guidance" "$negative" \
    "$role" "$repeat" "$status" "$parsed"
}

bench21_run_matrix() {
  local case_name model width height guidance negative gate i
  while IFS='|' read -r case_name model width height guidance negative gate; do
    bench21_run_request "$case_name" "$model" "$width" "$height" "$guidance" "$negative" warmup 0
    for ((i = 1; i <= repeats; i++)); do
      bench21_run_request "$case_name" "$model" "$width" "$height" "$guidance" "$negative" timed "$i"
    done
  done < <(bench21_cases "$tiers")
}

# bench21_summary <rows-file> -> JSON: per case, the median of the timed ok rows.
bench21_summary() {
  jq -s '
    def median: sort | if length == 0 then null
      elif length % 2 == 1 then .[length / 2 | floor]
      else (.[length / 2 - 1] + .[length / 2]) / 2 end;
    [ group_by(.case)[]
      | (map(select(.role == "timed" and .status == "ok"))) as $ok
      | {
          case: .[0].case,
          model: .[0].model,
          width: .[0].width,
          height: .[0].height,
          guidance: .[0].guidance,
          negative_prompt: .[0].negative_prompt,
          timed_ok: ($ok | length),
          timed_total: (map(select(.role == "timed")) | length),
          denoise_s_median: ($ok | map(.denoise_s) | median),
          total_s_median: ($ok | map(.total_s) | median),
          s_per_step_median: ($ok | map(.s_per_step) | median),
          vae_s_median: ($ok | map(.vae_s // empty) | median)
        } ]' "$1"
}

# bench21_check_gates <summary-json>: prints each failure, returns non-zero on any.
bench21_check_gates() {
  local summary="$1" failures=0 line case_name model width height guidance negative gate median
  while IFS='|' read -r case_name model width height guidance negative gate; do
    [[ -n "$gate" ]] || continue
    median="$(jq -r --arg c "$case_name" '.[] | select(.case == $c) | .denoise_s_median // "null"' <<<"$summary")"
    if [[ -z "$median" || "$median" == "null" ]]; then
      echo "GATE FAIL: $case_name has no successful timed row" >&2
      failures=$((failures + 1))
    elif ! awk -v m="$median" -v g="$gate" 'BEGIN { exit !(m <= g) }'; then
      echo "GATE FAIL: $case_name median denoise ${median}s > ${gate}s" >&2
      failures=$((failures + 1))
    else
      echo "gate ok: $case_name median denoise ${median}s <= ${gate}s" >&2
    fi
  done < <(bench21_cases "bf16")
  [[ "$failures" -eq 0 ]]
}

bench21_main() {
  mold_bin="${MOLD_BIN:-mold}"
  host=""
  out_dir=""
  repeats=3
  seed=210001
  tiers="bf16"
  gates=0
  dry_run=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --host)
        [[ $# -ge 2 ]] || bench21_die "--host requires a value"
        host="$2"
        shift 2
        ;;
      --mold-bin)
        [[ $# -ge 2 ]] || bench21_die "--mold-bin requires a value"
        mold_bin="$2"
        shift 2
        ;;
      --out-dir)
        [[ $# -ge 2 ]] || bench21_die "--out-dir requires a value"
        out_dir="$2"
        shift 2
        ;;
      --repeats)
        [[ $# -ge 2 ]] || bench21_die "--repeats requires a value"
        repeats="$2"
        shift 2
        ;;
      --seed)
        [[ $# -ge 2 ]] || bench21_die "--seed requires a value"
        seed="$2"
        shift 2
        ;;
      --tiers)
        [[ $# -ge 2 ]] || bench21_die "--tiers requires a value"
        tiers="$2"
        shift 2
        ;;
      --gates)
        gates=1
        shift
        ;;
      --dry-run)
        dry_run=1
        shift
        ;;
      -h | --help)
        bench21_usage
        exit 0
        ;;
      *)
        bench21_die "unknown option: $1"
        ;;
    esac
  done
  [[ "$seed" =~ ^[0-9]+$ ]] || bench21_die "--seed must be an integer"
  [[ "$repeats" =~ ^[0-9]+$ ]] && [[ "$repeats" -ge 1 ]] || bench21_die "--repeats must be >= 1"
  command -v jq >/dev/null 2>&1 || bench21_die "required command not found: jq"
  if [[ -z "$out_dir" ]]; then
    out_dir="${TMPDIR:-/tmp}/qwen21-bench-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  mkdir -p "$out_dir"
  rows_file="$out_dir/rows.ndjson"
  : > "$rows_file"

  if [[ "$dry_run" -eq 1 ]]; then
    git_sha="$(detect_git_sha "${BASH_SOURCE[0]}")"
    mold_version="unknown"
    gpu_name="unknown"
    bench21_emit_plan 2>/dev/null
    jq -s '.' "$rows_file"
    return 0
  fi

  [[ -n "$host" ]] || bench21_die "--host is required (the scratch server to benchmark)"
  if [[ ! -x "$mold_bin" ]]; then
    command -v "$mold_bin" >/dev/null 2>&1 || bench21_die "mold binary not found: $mold_bin"
  fi
  git_sha="$(detect_git_sha "$mold_bin")"
  mold_version="$(detect_mold_version "$mold_bin")"
  gpu_name="$(detect_gpu_name)"
  echo "mold:  $mold_bin ($mold_version, git $git_sha)" >&2
  echo "host:  $host" >&2
  echo "rows:  $rows_file" >&2

  bench21_run_matrix
  local summary
  summary="$(bench21_summary "$rows_file")"
  printf '%s\n' "$summary" > "$out_dir/summary.json"
  printf '%s\n' "$summary"
  if [[ "$gates" -eq 1 ]]; then
    bench21_check_gates "$summary"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  bench21_main "$@"
fi
