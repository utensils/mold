#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin" "$tmp/run" "$tmp/state"

cat > "$tmp/bin/magick" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
out="${@: -1}"
printf 'fake image\n' > "$out"
FAKE
chmod +x "$tmp/bin/magick"

cat > "$tmp/bin/curl" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
url="${@: -1}"
case "$url" in
  */api/models)
    cat <<'JSON'
[
  {
    "name": "sdxl-test:q8",
    "family": "sdxl",
    "downloaded": true,
    "default_steps": 1,
    "default_width": 1024,
    "default_height": 1024
  },
  {
    "name": "zimage-test:q8",
    "family": "z-image",
    "downloaded": true,
    "default_steps": 1,
    "default_width": 1024,
    "default_height": 1024
  },
  {
    "name": "cv:shared",
    "family": "flux",
    "downloaded": true,
    "default_steps": 1,
    "default_width": 512,
    "default_height": 512
  },
  {
    "name": "cv:shared",
    "family": "z-image",
    "downloaded": true,
    "default_steps": 1,
    "default_width": 512,
    "default_height": 512
  }
]
JSON
    ;;
  */api/catalog/installed?kind=lora)
    cat <<'JSON'
{
  "entries": [
    {"family": "z-image", "sub_family": "", "primary_path": "/tmp/z1.safetensors"},
    {"family": "z-image", "sub_family": "", "primary_path": "/tmp/z2.safetensors"}
  ]
}
JSON
    ;;
  *)
    echo "unexpected curl URL: $url" >&2
    exit 2
    ;;
esac
FAKE
chmod +x "$tmp/bin/curl"

cat > "$tmp/bin/fake-mold" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail

state_dir="${MOLD_FAKE_STATE:?}"
output=""
width=""
height=""
while (($# > 0)); do
  case "$1" in
    --output)
      shift
      output="$1"
      ;;
    --width)
      shift
      width="$1"
      ;;
    --height)
      shift
      height="$1"
      ;;
  esac
  shift || true
done

if [[ -z "$output" ]]; then
  echo "fake mold did not receive --output" >&2
  exit 2
fi

case "$output" in
  *sdxl-test-q8.base.png) printf '%s %s\n' "$width" "$height" > "$state_dir/sdxl-base-size" ;;
  *sdxl-test-q8.source.png) printf '%s %s\n' "$width" "$height" > "$state_dir/sdxl-source-size" ;;
  *zimage-test-q8.base.png) printf '%s %s\n' "$width" "$height" > "$state_dir/zimage-base-size" ;;
  *zimage-test-q8.source.png) printf '%s %s\n' "$width" "$height" > "$state_dir/zimage-source-size" ;;
  *zimage-test-q8.lora1.png) printf '%s %s\n' "$width" "$height" > "$state_dir/zimage-lora1-size" ;;
  *zimage-test-q8.lora2.png) printf '%s %s\n' "$width" "$height" > "$state_dir/zimage-lora2-size" ;;
esac

printf 'fake output\n' > "$output"
FAKE
chmod +x "$tmp/bin/fake-mold"

PATH="$tmp/bin:$PATH" \
MOLD_BIN="$tmp/bin/fake-mold" \
MOLD_FAKE_STATE="$tmp/state" \
MOLD_REGRESSION_OUT="$tmp/run" \
MOLD_REGRESSION_RUN_ID=family-sizing \
MOLD_REGRESSION_TIMEOUT_IMAGE=10 \
"$repo_root/scripts/regression-matrix.sh" > "$tmp/stdout" 2> "$tmp/stderr"

base_size="$(<"$tmp/state/sdxl-base-size")"
source_size="$(<"$tmp/state/sdxl-source-size")"
zimage_base_size="$(<"$tmp/state/zimage-base-size")"
zimage_source_size="$(<"$tmp/state/zimage-source-size")"

if [[ "$base_size" != "1024 1024" ]]; then
  echo "expected SDXL base case to keep model default 1024x1024, saw $base_size" >&2
  exit 1
fi

if [[ "$source_size" != "768 768" ]]; then
  echo "expected SDXL source case to use 768x768 fit-safe dimensions, saw $source_size" >&2
  echo "--- stdout ---" >&2
  cat "$tmp/stdout" >&2
  echo "--- stderr ---" >&2
  cat "$tmp/stderr" >&2
  exit 1
fi

if [[ "$zimage_base_size" != "768 768" ]]; then
  echo "expected Z-Image base case to use 768x768 fit-safe dimensions, saw $zimage_base_size" >&2
  exit 1
fi

if [[ "$zimage_source_size" != "768 768" ]]; then
  echo "expected Z-Image source case to use 768x768 fit-safe dimensions, saw $zimage_source_size" >&2
  exit 1
fi

if [[ -e "$tmp/state/zimage-lora1-size" || -e "$tmp/state/zimage-lora2-size" ]]; then
  echo "expected Z-Image LoRA cases to be skipped while offload+LoRA is unsupported" >&2
  exit 1
fi

run_dir="$tmp/run/family-sizing"
duplicate_outputs="$(jq -r 'select(.status == "ok") | .output' "$run_dir/results.jsonl" | sort | uniq -d)"
if [[ -n "$duplicate_outputs" ]]; then
  echo "expected family-qualified output paths, saw duplicates:" >&2
  printf '%s\n' "$duplicate_outputs" >&2
  exit 1
fi

if jq -e 'select(.model == "cv:shared")' "$run_dir/results.jsonl" > /dev/null; then
  echo "expected ambiguous duplicate model names to be skipped" >&2
  exit 1
fi

if ! grep -q $'cv:shared\tflux' "$run_dir/ambiguous-models.tsv" || ! grep -q $'cv:shared\tz-image' "$run_dir/ambiguous-models.tsv"; then
  echo "expected ambiguous duplicate model names to be recorded" >&2
  exit 1
fi

echo "regression matrix applies family-specific fit-safe sizing"
