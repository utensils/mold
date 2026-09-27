#!/usr/bin/env bash
# Qwen Image 2.1 Metal UAT matrix (#1770): every path PR #1767 added, run on
# Apple Silicon one render at a time under a memory guard.
#
#   scripts/capture-qwen21-metal-uat.sh            # run the whole matrix
#   scripts/capture-qwen21-metal-uat.sh tier-q4    # run named cases only
#   QWEN21_UAT_PLAN=1 scripts/capture-qwen21-metal-uat.sh   # print the plan
#
# The guard exists because a unified-memory Mac that runs out of memory takes
# the whole desktop down with it. Every case runs `--local` with no server, no
# build and no second mold alive; a watchdog samples `memory_pressure` and
# kills the render the moment free memory drops below QWEN21_UAT_MIN_FREE_PCT.
# Cases escalate from the smallest tier to the 2K presets, so a machine that
# cannot hold a case stops before the ones that need more. Admission refusals
# are recorded as results, never bypassed.
#
# Results land in $QWEN21_UAT_OUT/results.json (one row per case) beside each
# case's output file and log. Fold them into
# docs/qualification/qwen-image-2.1-metal-uat.json by hand after review.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mold_home="${MOLD_HOME:-/Volumes/ExternalStorage/mold2}"
bin="${QWEN21_UAT_BIN:-$repo_root/target/dev-fast/mold}"
v032_bin="${QWEN21_UAT_V032_BIN:-}"
lora="${QWEN21_UAT_LORA:-}"
min_free_pct="${QWEN21_UAT_MIN_FREE_PCT:-12}"
out="${QWEN21_UAT_OUT:-$mold_home/output/verification/qwen-image-2.1-metal/$(date -u +%Y%m%dT%H%M%SZ)}"
plan_only="${QWEN21_UAT_PLAN:-0}"

# The v0.32 qualification render (docs/qualification/qwen-image-2.1-metal-uat.json).
teapot="A small red ceramic teapot on a sunlit wooden windowsill, editorial product photograph, soft morning shadows"
v032_sha="0e8fc93c5686a6f1c8107daf91a53e151bedc2bf8379e801612857e08fbc516b"
lantern="A red paper lantern with a gold tassel"
edit="Keep the subject of image 1 and restyle it as a watercolour illustration"

# id | model | expect (render|refused) | extra `mold run` arguments
# Order is the memory escalation: quantized tiers, then bf16 at 1024², then
# references, then the 2K presets last.
cases=(
  "tier-q4|qwen-image-2.1:q4|render|--width 1024 --height 1024 --seed 210001"
  "tier-q8|qwen-image-2.1:q8|render|--width 1024 --height 1024 --seed 210001"
  "tier-int8-conv|qwen-image-2.1:int8-conv|render|--width 1024 --height 1024 --seed 210001"
  "tier-fp8|qwen-image-2.1:fp8|refused|--width 1024 --height 1024 --seed 210001"
  "t2i-bf16-v032|qwen-image-2.1:bf16|render|--width 1024 --height 1024 --steps 40 --guidance 1 --seed 210001"
  "t2i-bf16-attn-math|qwen-image-2.1:bf16|render|--width 1024 --height 1024 --steps 40 --guidance 1 --seed 210001"
  "turbo-bf16|qwen-image-2.1-turbo:bf16|render|--seed 7"
  "webp-still|qwen-image-2.1-turbo:bf16|render|--seed 7 --format webp"
  "transparent-toggle|qwen-image-2.1-turbo:bf16|render|--seed 11 --transparent"
  "alpha-reference|qwen-image-2.1-turbo:bf16|render|--seed 12"
  "opaque-reference|qwen-image-2.1-turbo:bf16|render|--seed 13"
  "turbo-lora|qwen-image-2.1-turbo:bf16|render|--seed 7"
  "refs-1|qwen-image-2.1-turbo:bf16|render|--seed 21"
  "refs-3|qwen-image-2.1-turbo:bf16|render|--seed 22"
  "refs-10|qwen-image-2.1-turbo:bf16|render|--seed 23"
  "2k-square-q4|qwen-image-2.1:q4|render|--width 2048 --height 2048 --seed 31"
  "2k-square-bf16|qwen-image-2.1:bf16|render|--width 2048 --height 2048 --seed 31"
  "2k-wide-bf16|qwen-image-2.1:bf16|render|--width 2752 --height 1536 --seed 32"
)

fail() {
  echo "Qwen Image 2.1 Metal UAT: $*" >&2
  exit 1
}

prompt_for() {
  case "$1" in
    t2i-*) echo "$teapot" ;;
    transparent-toggle) echo "$lantern" ;;
    alpha-reference | opaque-reference | refs-*) echo "$edit" ;;
    *) echo "$teapot" ;;
  esac
}

selected() {
  local id="$1"
  [[ ${#requested[@]} -eq 0 ]] && return 0
  local want
  for want in "${requested[@]}"; do [[ "$want" == "$id" ]] && return 0; done
  return 1
}

requested=("$@")
requested=(${requested[@]+"${requested[@]}"})

if [[ "$plan_only" == 1 ]]; then
  for row in "${cases[@]}"; do
    IFS='|' read -r id model expect args <<<"$row"
    selected "$id" || continue
    printf '%s\t%s\t%s\t%s\n' "$id" "$model" "$expect" "$args"
  done
  exit 0
fi

# ---- preflight: this machine, this binary, nothing else heavy alive ----------
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || fail "Apple Silicon Metal only"
[[ -x "$bin" ]] || fail "no mold binary at $bin (build with --features metal,webp first)"
for command in jq shasum sips memory_pressure; do
  command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
done
for busy in cargo rustc xcodebuild swift-frontend; do
  if pgrep -x "$busy" >/dev/null; then fail "$busy is running; one heavy process at a time"; fi
done
if pgrep -f "mold (serve|run)" >/dev/null; then fail "another mold serve/run is alive; stop it first"; fi
if lsof -iTCP:7680 -sTCP:LISTEN >/dev/null 2>&1; then fail "something is listening on :7680"; fi

free_pct() {
  memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {sub("%", "", $2); print $2}'
}
start_free="$(free_pct)"
((start_free >= min_free_pct + 20)) \
  || fail "only ${start_free}% memory free before starting; close applications first"

mkdir -p "$out/inputs"
export MOLD_HOME="$mold_home"
# Never reach a server: every render is this process on this GPU.
export MOLD_HOST="http://127.0.0.1:9"
results="$out/results.json"
echo '[]' >"$results"

record() {
  jq --argjson row "$1" '. + [$row]' "$results" >"$results.tmp" && mv "$results.tmp" "$results"
}

# Runs one render under the watchdog. Sets: status, exit_code, seconds,
# peak_footprint, min_free.
guarded_run() {
  local log="$1"
  shift
  local started
  started="$(date +%s)"
  /usr/bin/time -l "$@" >"$log" 2>&1 &
  local pid=$!
  min_free=100
  status=""
  while kill -0 "$pid" 2>/dev/null; do
    local now
    now="$(free_pct)"
    [[ -n "$now" ]] && ((now < min_free)) && min_free="$now"
    if [[ -n "$now" ]] && ((now < min_free_pct)); then
      echo "memory guard: ${now}% free < ${min_free_pct}%, killing the render" | tee -a "$log" >&2
      pkill -TERM -P "$pid" 2>/dev/null || true
      sleep 3
      pkill -KILL -P "$pid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
      status="aborted-memory"
      break
    fi
    sleep 2
  done
  set +e
  wait "$pid"
  exit_code=$?
  set -e
  seconds=$(($(date +%s) - started))
  peak_footprint="$(awk '/peak memory footprint/ {print $1}' "$log" | tail -1)"
  [[ -n "$peak_footprint" ]] || peak_footprint=0
}

has_alpha() {
  sips -g hasAlpha "$1" 2>/dev/null | awk '/hasAlpha/ {print $2}'
}

# Ten ~1024² references with differing aspects, derived from the turbo render
# so no external asset is needed; the last one is portrait, so `canvas:
# last-reference` has something to follow.
make_references() {
  local base="$1" i
  for i in $(seq 1 10); do
    local ref="$out/inputs/ref-$i.png"
    cp "$base" "$ref"
    case $((i % 4)) in
      0) sips --flip horizontal "$ref" >/dev/null ;;
      1) sips --rotate 90 "$ref" >/dev/null ;;
      2) sips --resampleHeightWidth 832 1216 "$ref" >/dev/null ;;
      3) sips --flip vertical "$ref" >/dev/null ;;
    esac
  done
  sips --resampleHeightWidth 1216 832 "$out/inputs/ref-10.png" >/dev/null
  sips -s format jpeg "$base" --out "$out/inputs/opaque.jpg" >/dev/null
}

case_args() {
  local id="$1"
  case "$id" in
    alpha-reference) echo "--reference $out/transparent-toggle.png" ;;
    opaque-reference) echo "--reference $out/inputs/opaque.jpg" ;;
    turbo-lora) echo "--lora $lora" ;;
    refs-1) echo "--reference $out/inputs/ref-10.png" ;;
    refs-3) printf -- '--reference %s ' "$out"/inputs/ref-{1,2,10}.png ;;
    refs-10) printf -- '--reference %s ' "$out"/inputs/ref-{1..10}.png ;;
  esac
}

for row in "${cases[@]}"; do
  IFS='|' read -r id model expect args <<<"$row"
  selected "$id" || continue
  ext=png
  [[ "$args" == *"--format webp"* ]] && ext=webp
  output="$out/$id.$ext"
  log="$out/$id.log"
  env_prefix=()
  [[ "$id" == t2i-bf16-attn-math ]] && env_prefix=(env MOLD_ATTN=math)
  if [[ "$id" == turbo-lora && -z "$lora" ]]; then
    record "$(jq -nc --arg id "$id" '{id: $id, status: "skipped", note: "set QWEN21_UAT_LORA to a Qwen Image 2.1 LoRA"}')"
    continue
  fi
  if [[ "$id" == refs-* || "$id" == opaque-reference ]] && [[ ! -f "$out/inputs/ref-10.png" ]]; then
    [[ -f "$out/turbo-bf16.png" ]] || fail "$id needs the turbo-bf16 case to have rendered first"
    make_references "$out/turbo-bf16.png"
  fi
  if [[ "$id" == alpha-reference && ! -f "$out/transparent-toggle.png" ]]; then
    fail "alpha-reference needs the transparent-toggle case to have rendered first"
  fi
  echo "=== $id ($model) — $(date '+%H:%M:%S'), ${start_free}% free at start" >&2
  # shellcheck disable=SC2046,SC2086
  guarded_run "$log" ${env_prefix[@]+"${env_prefix[@]}"} "$bin" run "$model" "$(prompt_for "$id")" \
    $args $(case_args "$id") --local --output "$output"
  if [[ -z "$status" ]]; then
    if [[ "$expect" == refused ]]; then
      if ((exit_code != 0)); then status="pass"; else status="fail"; fi
    elif ((exit_code == 0)) && [[ -s "$output" ]]; then
      status="pass"
    elif grep -qiE 'refus|not enough memory|insufficient memory|exceeds' "$log"; then
      status="refused"
    else
      status="fail"
    fi
  fi
  sha="" alpha="" width="" height=""
  if [[ -s "$output" ]]; then
    sha="$(shasum -a 256 "$output" | awk '{print $1}')"
    alpha="$(has_alpha "$output")"
    width="$(sips -g pixelWidth "$output" 2>/dev/null | awk '/pixelWidth/ {print $2}')"
    height="$(sips -g pixelHeight "$output" 2>/dev/null | awk '/pixelHeight/ {print $2}')"
  fi
  note=""
  case "$id" in
    t2i-bf16-v032)
      if [[ "$sha" == "$v032_sha" ]]; then note="byte-identical to the v0.32 M5 Max qualification render"; else note="differs from the v0.32 M5 Max hash $v032_sha; compare against QWEN21_UAT_V032_BIN on this machine"; fi ;;
    transparent-toggle | alpha-reference) [[ "$alpha" == yes ]] || { [[ "$status" == pass ]] && status="fail"; note="expected real alpha"; } ;;
    opaque-reference) [[ "$alpha" != yes ]] || { [[ "$status" == pass ]] && status="fail"; note="an opaque reference without the toggle must not produce alpha"; } ;;
    webp-still) head -c 12 "$output" 2>/dev/null | grep -q WEBP || { [[ "$status" == pass ]] && status="fail"; note="not a RIFF/WEBP file"; } ;;
    refs-*) note="last reference is 832x1216; with no --width/--height the canvas should follow it" ;;
  esac
  record "$(jq -nc \
    --arg id "$id" --arg model "$model" --arg status "$status" --arg sha "$sha" \
    --arg alpha "$alpha" --arg width "$width" --arg height "$height" --arg note "$note" \
    --argjson exit "$exit_code" --argjson seconds "$seconds" \
    --argjson peak "$peak_footprint" --argjson min_free "$min_free" \
    '{id: $id, model: $model, status: $status, exit_code: $exit, seconds: $seconds,
      peak_footprint_bytes: $peak, min_free_pct: $min_free, sha256: $sha,
      has_alpha: $alpha, width: $width, height: $height, note: $note}')"
  echo "    -> $status (exit $exit_code, ${seconds}s, min ${min_free}% free)" >&2
  [[ "$status" == aborted-memory ]] && fail "stopped after $id hit the memory guard; results in $results"
done

if [[ -n "$v032_bin" && -x "$v032_bin" ]] && selected t2i-bf16-v032-baseline; then
  guarded_run "$out/v032.log" "$v032_bin" run qwen-image-2.1:bf16 "$teapot" \
    --width 1024 --height 1024 --steps 40 --guidance 1 --seed 210001 --local --output "$out/v032.png"
  record "$(jq -nc --arg sha "$(shasum -a 256 "$out/v032.png" 2>/dev/null | awk '{print $1}')" \
    --argjson exit "$exit_code" '{id: "t2i-bf16-v032-baseline", status: (if $exit == 0 then "pass" else "fail" end), sha256: $sha}')"
fi

echo "results: $results" >&2
jq -r '.[] | "\(.id)\t\(.status)"' "$results"
