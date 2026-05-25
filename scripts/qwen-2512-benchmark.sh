#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/qwen-2512-benchmark.sh [options]

Runs repeatable Qwen-Image-2512 timing probes and writes one NDJSON metric row
per phase. The default phases are:
  1. local cold CLI request
  2. server model load request
  3. warm server streaming request

Options:
  --mold-bin PATH     mold binary to use for local runs. Defaults to ./target/release/mold, then mold
  --host URL          Server URL. Defaults to $MOLD_HOST or http://localhost:7680
  --api-key KEY       Optional X-Api-Key value. Defaults to $MOLD_API_KEY
  --model NAME        Defaults to qwen-image-2512:q6
  --prompt TEXT       Defaults to a short deterministic benchmark prompt
  --width N           Defaults to 512
  --height N          Defaults to 512
  --steps N           Defaults to 4
  --seed N            Defaults to 12345
  --out-dir DIR       Defaults to qwen-2512-bench-<timestamp>
  --metrics PATH      Defaults to <out-dir>/metrics.ndjson
  --skip-local        Do not run the local cold CLI phase
  --skip-server       Do not run the server load/warm phases
  --smoke             Enable seeded output sanity checks; fails near-black outputs
  --dry-run           Print planned phases as NDJSON without running mold/curl
  --self-test         Run formatting/parser self-tests without model weights
  -h, --help          Show this help
USAGE
}

host="${MOLD_HOST:-http://localhost:7680}"
api_key="${MOLD_API_KEY:-}"
mold_bin=""
model="qwen-image-2512:q6"
prompt="a small red ceramic teapot on a gray table"
width="512"
height="512"
steps="4"
seed="12345"
out_dir=""
metrics_path=""
skip_local=0
skip_server=0
smoke=0
dry_run=0

die() {
  echo "error: $*" >&2
  exit 2
}

need_arg() {
  [[ $# -ge 2 ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mold-bin)
      need_arg "$@"
      mold_bin="$2"
      shift 2
      ;;
    --host)
      need_arg "$@"
      host="$2"
      shift 2
      ;;
    --api-key)
      need_arg "$@"
      api_key="$2"
      shift 2
      ;;
    --model)
      need_arg "$@"
      model="$2"
      shift 2
      ;;
    --prompt)
      need_arg "$@"
      prompt="$2"
      shift 2
      ;;
    --width)
      need_arg "$@"
      width="$2"
      shift 2
      ;;
    --height)
      need_arg "$@"
      height="$2"
      shift 2
      ;;
    --steps)
      need_arg "$@"
      steps="$2"
      shift 2
      ;;
    --seed)
      need_arg "$@"
      seed="$2"
      shift 2
      ;;
    --out-dir)
      need_arg "$@"
      out_dir="$2"
      shift 2
      ;;
    --metrics)
      need_arg "$@"
      metrics_path="$2"
      shift 2
      ;;
    --skip-local)
      skip_local=1
      shift
      ;;
    --skip-server)
      skip_server=1
      shift
      ;;
    --smoke)
      smoke=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --self-test)
      self_test=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

self_test="${self_test:-0}"

for value_name in width height steps seed; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9]+$ ]] || die "--$value_name must be an integer"
done

if [[ -z "$out_dir" ]]; then
  out_dir="qwen-2512-bench-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$metrics_path" ]]; then
  metrics_path="$out_dir/metrics.ndjson"
fi

if [[ -z "$mold_bin" ]]; then
  if [[ -x ./target/release/mold ]]; then
    mold_bin="./target/release/mold"
  else
    mold_bin="mold"
  fi
fi

now_ms() {
  date +%s%3N
}

file_bytes() {
  if [[ -f "$1" ]]; then
    wc -c < "$1" | tr -d ' '
  else
    printf '0'
  fi
}

json_row() {
  jq -cn \
    --arg phase "$1" \
    --arg mode "$2" \
    --arg model "$model" \
    --arg prompt "$prompt" \
    --arg output_path "$3" \
    --arg log_path "$4" \
    --arg vae_mode "$5" \
    --argjson width "$width" \
    --argjson height "$height" \
    --argjson steps "$steps" \
    --argjson seed "$seed" \
    --argjson start_ms "$6" \
    --argjson end_ms "$7" \
    --argjson duration_ms "$((${7} - ${6}))" \
    --argjson http_code "$8" \
    --argjson output_bytes "$9" \
    --argjson transformer_reload_count "${10}" \
    --argjson encoder_reload_count "${11}" \
    '{
      phase: $phase,
      mode: $mode,
      model: $model,
      prompt: $prompt,
      width: $width,
      height: $height,
      steps: $steps,
      seed: $seed,
      start_ms: $start_ms,
      end_ms: $end_ms,
      duration_ms: $duration_ms,
      http_code: $http_code,
      output_path: $output_path,
      output_bytes: $output_bytes,
      transformer_reload_count: $transformer_reload_count,
      encoder_reload_count: $encoder_reload_count,
      vae_mode: $vae_mode,
      log_path: $log_path
    }'
}

count_transformer_loads() {
  [[ -f "$1" ]] || {
    printf '0'
    return
  }
  grep -Eic 'Loading Qwen-Image transformer|Qwen-Image transformer loaded|Loading Qwen-Image transformer \(quantized\)' "$1" || true
}

count_encoder_loads() {
  [[ -f "$1" ]] || {
    printf '0'
    return
  }
  grep -Eic 'Loading Qwen2\.5 text encoder|Qwen2\.5 text encoder loaded|Freed Qwen2\.5 text encoder' "$1" || true
}

detect_vae_mode() {
  [[ -f "$1" ]] || {
    printf 'unknown'
    return
  }
  if grep -Eiq 'Tiled GPU VAE|tiled GPU decode|MOLD_VAE_TILED|retrying with tiled GPU decode' "$1"; then
    printf 'tiled-gpu'
  elif grep -Eiq 'Loading Qwen-Image VAE \(CPU|retrying on CPU|VAE decode.*CPU' "$1"; then
    printf 'cpu'
  elif grep -Eiq 'Loading Qwen-Image VAE \(GPU|VAE decode' "$1"; then
    printf 'gpu'
  else
    printf 'unknown'
  fi
}

image_sanity_values() {
  local image_path="$1"
  magick "$image_path" -colorspace RGB -format '%[fx:mean] %[fx:maxima]' info:
}

image_is_near_black() {
  local image_path="$1"
  local mean max
  read -r mean max < <(image_sanity_values "$image_path")
  awk -v mean="$mean" -v max="$max" 'BEGIN { exit !(max <= 0.02 && mean <= 0.01) }'
}

validate_smoke_image() {
  local image_path="$1"
  local label="$2"
  [[ "$smoke" -eq 1 ]] || return 0
  [[ -s "$image_path" ]] || die "$label smoke output missing or empty: $image_path"
  command -v magick >/dev/null 2>&1 || die "--smoke requires ImageMagick 'magick'"
  local mean max
  read -r mean max < <(image_sanity_values "$image_path")
  if awk -v mean="$mean" -v max="$max" 'BEGIN { exit !(max <= 0.02 && mean <= 0.01) }'; then
    die "$label smoke output is near-black (mean=$mean max=$max): $image_path"
  fi
  echo "$label smoke sanity ok (mean=$mean max=$max)"
}

validate_qwen_debug_log() {
  local log_path="$1"
  local label="$2"
  [[ "$smoke" -eq 1 ]] || return 0
  [[ -f "$log_path" ]] || return 0
  if grep -Eq 'NaN=[1-9]|[+]Inf=[1-9]|-Inf=[1-9]|near-black' "$log_path"; then
    die "$label smoke log reported Qwen tensor corruption or near-black decode: $log_path"
  fi
}

event_data_lines() {
  awk '
    /^data: / {
      sub(/^data: /, "")
      print
    }
  ' "$1"
}

extract_complete_image() {
  local sse_file="$1"
  local output_path="$2"
  local complete_json
  complete_json="$(awk '
    /^event: / {
      event=$2
      next
    }
    /^data: / && event == "complete" {
      sub(/^data: /, "")
      print
      exit
    }
  ' "$sse_file")"
  [[ -n "$complete_json" ]] || return 1
  jq -r '.image' <<<"$complete_json" | base64 -d > "$output_path"
}

server_request_json() {
  jq -n \
    --arg prompt "$prompt" \
    --arg model "$model" \
    --argjson width "$width" \
    --argjson height "$height" \
    --argjson steps "$steps" \
    --argjson seed "$seed" \
    '{
      prompt: $prompt,
      model: $model,
      width: $width,
      height: $height,
      steps: $steps,
      seed: $seed,
      batch_size: 1,
      output_format: "png"
    }'
}

append_row() {
  json_row "$@" | tee -a "$metrics_path" >/dev/null
}

run_local_cold() {
  local output_path="$out_dir/local-cold.png"
  local log_path="$out_dir/local-cold.stderr.log"
  local start end status

  start="$(now_ms)"
  set +e
  if [[ "$smoke" -eq 1 ]]; then
    MOLD_QWEN_DEBUG=1 "$mold_bin" run "$model" "$prompt" \
      --local \
      --width "$width" \
      --height "$height" \
      --steps "$steps" \
      --seed "$seed" \
      --output "$output_path" \
      2> "$log_path"
  else
    "$mold_bin" run "$model" "$prompt" \
      --local \
      --width "$width" \
      --height "$height" \
      --steps "$steps" \
      --seed "$seed" \
      --output "$output_path" \
      2> "$log_path"
  fi
  status=$?
  set -e
  end="$(now_ms)"

  append_row \
    "local_cold" \
    "cli" \
    "$output_path" \
    "$log_path" \
    "$(detect_vae_mode "$log_path")" \
    "$start" \
    "$end" \
    "$status" \
    "$(file_bytes "$output_path")" \
    "$(count_transformer_loads "$log_path")" \
    "$(count_encoder_loads "$log_path")"

  [[ "$status" -eq 0 ]] || return "$status"
  validate_smoke_image "$output_path" "local_cold"
  validate_qwen_debug_log "$log_path" "local_cold"
}

run_server_load() {
  local log_path="$out_dir/server-load.curl.txt"
  local body_path="$out_dir/server-load.response.json"
  local start end code
  local headers=(-H "Content-Type: application/json")
  [[ -z "$api_key" ]] || headers+=(-H "X-Api-Key: $api_key")

  start="$(now_ms)"
  code="$(curl -sS "${headers[@]}" \
    -X POST "$host/api/models/load" \
    --data-binary "$(jq -cn --arg model "$model" '{model: $model}')" \
    -o "$body_path" \
    -w '%{http_code}' \
    2> "$log_path")"
  end="$(now_ms)"

  append_row \
    "server_load" \
    "http" \
    "$body_path" \
    "$log_path" \
    "unknown" \
    "$start" \
    "$end" \
    "$code" \
    "$(file_bytes "$body_path")" \
    0 \
    0

  [[ "$code" -ge 200 && "$code" -lt 300 ]]
}

run_server_warm() {
  local request_path="$out_dir/server-warm.request.json"
  local sse_path="$out_dir/server-warm.sse"
  local progress_path="$out_dir/server-warm.progress.jsonl"
  local output_path="$out_dir/server-warm.png"
  local curl_log="$out_dir/server-warm.curl.txt"
  local start end code transformer_count encoder_count vae_mode
  local headers=(-H "Content-Type: application/json")
  [[ -z "$api_key" ]] || headers+=(-H "X-Api-Key: $api_key")

  server_request_json > "$request_path"
  start="$(now_ms)"
  code="$(curl -sS -N "${headers[@]}" \
    -X POST "$host/api/generate/stream" \
    --data-binary "@$request_path" \
    -o "$sse_path" \
    -w '%{http_code}' \
    2> "$curl_log")"
  end="$(now_ms)"

  event_data_lines "$sse_path" > "$progress_path" || true
  extract_complete_image "$sse_path" "$output_path" || true
  transformer_count="$(grep -Eic 'Loading Qwen-Image transformer|Qwen-Image transformer loaded' "$progress_path" || true)"
  encoder_count="$(grep -Eic 'Loading Qwen2\.5 text encoder|Freed Qwen2\.5 text encoder' "$progress_path" || true)"
  vae_mode="$(detect_vae_mode "$progress_path")"

  append_row \
    "server_warm_request" \
    "sse" \
    "$output_path" \
    "$progress_path" \
    "$vae_mode" \
    "$start" \
    "$end" \
    "$code" \
    "$(file_bytes "$output_path")" \
    "$transformer_count" \
    "$encoder_count"

  [[ "$code" -ge 200 && "$code" -lt 300 ]] && [[ -s "$output_path" ]]
  validate_smoke_image "$output_path" "server_warm_request"
  validate_qwen_debug_log "$progress_path" "server_warm_request"
}

emit_plan() {
  [[ "$skip_local" -eq 1 ]] || json_row "local_cold" "cli" "$out_dir/local-cold.png" "" "unknown" 0 0 0 0 0 0
  if [[ "$skip_server" -eq 0 ]]; then
    json_row "server_load" "http" "$out_dir/server-load.response.json" "" "unknown" 0 0 0 0 0 0
    json_row "server_warm_request" "sse" "$out_dir/server-warm.png" "" "unknown" 0 0 0 0 0 0
  fi
}

run_self_test() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  cat > "$tmp/sample.sse" <<'EOF'
event: progress
data: {"type":"stage_start","name":"Loading Qwen-Image transformer (quantized)"}

event: progress
data: {"type":"info","message":"Loading Qwen2.5 text encoder (q4 GGUF, GPU)"}

event: progress
data: {"type":"stage_start","name":"VAE decode"}

event: complete
data: {"image":"aGVsbG8=","format":"png","width":1,"height":1,"seed_used":1,"generation_time_ms":2,"model":"qwen-image-2512:q6"}
EOF

  event_data_lines "$tmp/sample.sse" > "$tmp/progress.jsonl"
  extract_complete_image "$tmp/sample.sse" "$tmp/out.bin"
  [[ "$(cat "$tmp/out.bin")" == "hello" ]] || die "self-test complete image extraction failed"
  [[ "$(count_transformer_loads "$tmp/progress.jsonl")" == "1" ]] || die "self-test transformer count failed"
  [[ "$(count_encoder_loads "$tmp/progress.jsonl")" == "1" ]] || die "self-test encoder count failed"
  [[ "$(detect_vae_mode "$tmp/progress.jsonl")" == "gpu" ]] || die "self-test VAE mode failed"
  json_row "self_test" "dry" "$tmp/out.bin" "$tmp/progress.jsonl" "gpu" 1 3 200 5 1 1 | jq -e '.duration_ms == 2 and .transformer_reload_count == 1' >/dev/null
  if command -v magick >/dev/null 2>&1; then
    magick -size 2x2 xc:black "$tmp/black.png"
    magick -size 2x2 xc:white "$tmp/white.png"
    image_is_near_black "$tmp/black.png" || die "self-test near-black detection failed"
    if image_is_near_black "$tmp/white.png"; then
      die "self-test non-black detection failed"
    fi
  fi
  echo "self-test ok"
}

if [[ "$self_test" -eq 1 ]]; then
  for bin in jq base64 awk grep; do
    command -v "$bin" >/dev/null 2>&1 || die "required command not found: $bin"
  done
  run_self_test
  exit 0
fi

if [[ "$dry_run" -eq 1 ]]; then
  command -v jq >/dev/null 2>&1 || die "required command not found: jq"
  emit_plan
  exit 0
fi

for bin in jq curl base64 awk grep; do
  command -v "$bin" >/dev/null 2>&1 || die "required command not found: $bin"
done
[[ "$skip_local" -eq 1 ]] || command -v "$mold_bin" >/dev/null 2>&1 || [[ -x "$mold_bin" ]] || die "mold binary not found: $mold_bin"

mkdir -p "$out_dir"
: > "$metrics_path"

if [[ "$skip_local" -eq 0 ]]; then
  run_local_cold
fi
if [[ "$skip_server" -eq 0 ]]; then
  run_server_load
  run_server_warm
fi

echo "wrote benchmark metrics to $metrics_path"
