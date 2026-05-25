#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/qwen-edit-parity-smoke.sh --image input.png [options]

Runs two independent qwen-image-edit requests against a mold server and records
GPU/resource snapshots before, between, and after the requests. This is a
manual CUDA smoke harness for checking whether request 2 grows or OOMs after an
identical successful request 1.

Options:
  --host URL       Server URL. Defaults to $MOLD_HOST or http://localhost:7680
  --api-key KEY    Optional X-Api-Key value. Defaults to $MOLD_API_KEY
  --image PATH     PNG/JPEG edit input image. Required
  --prompt TEXT    Prompt. Defaults to a small deterministic edit prompt
  --model NAME     Defaults to qwen-image-edit-2511:q4
  --width N        Defaults to 1328
  --height N       Defaults to 1328
  --steps N        Defaults to 20
  --seed N         Defaults to 12345
  --runs N         Defaults to 2
  --out-dir DIR    Defaults to qwen-edit-parity-<timestamp>
  -h, --help       Show this help
USAGE
}

host="${MOLD_HOST:-http://localhost:7680}"
api_key="${MOLD_API_KEY:-}"
image_path=""
prompt="make the subject look cinematic while preserving composition"
model="qwen-image-edit-2511:q4"
width="1328"
height="1328"
steps="20"
seed="12345"
runs="2"
out_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      host="$2"
      shift 2
      ;;
    --api-key)
      api_key="$2"
      shift 2
      ;;
    --image)
      image_path="$2"
      shift 2
      ;;
    --prompt)
      prompt="$2"
      shift 2
      ;;
    --model)
      model="$2"
      shift 2
      ;;
    --width)
      width="$2"
      shift 2
      ;;
    --height)
      height="$2"
      shift 2
      ;;
    --steps)
      steps="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --runs)
      runs="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$image_path" ]]; then
  echo "--image is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$image_path" ]]; then
  echo "image not found: $image_path" >&2
  exit 2
fi

for bin in curl jq base64; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "required command not found: $bin" >&2
    exit 2
  fi
done

if ! [[ "$runs" =~ ^[0-9]+$ ]] || [[ "$runs" -lt 2 ]]; then
  echo "--runs must be an integer >= 2" >&2
  exit 2
fi

if [[ -z "$out_dir" ]]; then
  out_dir="qwen-edit-parity-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$out_dir"

auth_headers=(-H "Content-Type: application/json")
if [[ -n "$api_key" ]]; then
  auth_headers+=(-H "X-Api-Key: $api_key")
fi

b64_file() {
  if base64 --help 2>&1 | grep -q -- "-w"; then
    base64 -w0 "$1"
  else
    base64 "$1" | tr -d '\n'
  fi
}

snapshot() {
  local name="$1"
  local prefix="$out_dir/$name"

  curl -fsS "${auth_headers[@]}" "$host/api/status" \
    -o "$prefix.status.json" || true
  curl -fsS "${auth_headers[@]}" "$host/api/resources" \
    -o "$prefix.resources.json" || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits > "$prefix.nvidia-smi.csv" || true
  fi
}

image_b64="$(b64_file "$image_path")"
request_json="$out_dir/request.json"
jq -n \
  --arg prompt "$prompt" \
  --arg model "$model" \
  --arg image "$image_b64" \
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
    output_format: "png",
    edit_images: [$image]
  }' > "$request_json"

cat > "$out_dir/summary.tsv" <<'EOF'
run	http_code	time_total_seconds	output_bytes	output_path
EOF

snapshot "00-before"

for run in $(seq 1 "$runs"); do
  snapshot "$(printf '%02d-before-run' "$run")"

  output_path="$out_dir/run-$run.png"
  meta_path="$out_dir/run-$run.curl.txt"
  curl -sS "${auth_headers[@]}" \
    -X POST "$host/api/generate" \
    --data-binary "@$request_json" \
    -o "$output_path" \
    -w '%{http_code} %{time_total}' \
    > "$meta_path"
  read -r code time_total < "$meta_path"
  bytes="$(wc -c < "$output_path" | tr -d ' ')"
  printf '%s\t%s\t%s\t%s\t%s\n' "$run" "$code" "$time_total" "$bytes" "$output_path" \
    | tee -a "$out_dir/summary.tsv"

  if [[ "$code" -lt 200 || "$code" -ge 300 ]]; then
    echo "request $run failed with HTTP $code; body saved at $output_path" >&2
    snapshot "$(printf '%02d-after-failure' "$run")"
    exit 1
  fi

  snapshot "$(printf '%02d-after-run' "$run")"
done

snapshot "99-after"
echo "wrote smoke artifacts to $out_dir"
