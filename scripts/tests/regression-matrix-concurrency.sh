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
    "name": "flux-schnell:q8",
    "family": "flux",
    "downloaded": true,
    "default_steps": 1,
    "default_width": 64,
    "default_height": 64
  }
]
JSON
    ;;
  */api/catalog/installed?kind=lora)
    cat <<'JSON'
{
  "entries": [
    {"family": "flux", "sub_family": "", "primary_path": "/tmp/first.safetensors"},
    {"family": "flux", "sub_family": "", "primary_path": "/tmp/second.safetensors"}
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
while (($# > 0)); do
  if [[ "$1" == "--output" ]]; then
    shift
    output="$1"
  fi
  shift || true
done

if [[ -z "$output" ]]; then
  echo "fake mold did not receive --output" >&2
  exit 2
fi

(
  flock 9
  current=0
  [[ -f "$state_dir/current" ]] && current="$(<"$state_dir/current")"
  current=$((current + 1))
  printf '%s\n' "$current" > "$state_dir/current"
  max=0
  [[ -f "$state_dir/max" ]] && max="$(<"$state_dir/max")"
  if (( current > max )); then
    printf '%s\n' "$current" > "$state_dir/max"
  fi
) 9>"$state_dir/lock"

sleep 0.25
printf 'fake output\n' > "$output"

(
  flock 9
  current="$(<"$state_dir/current")"
  printf '%s\n' "$((current - 1))" > "$state_dir/current"
) 9>"$state_dir/lock"
FAKE
chmod +x "$tmp/bin/fake-mold"

PATH="$tmp/bin:$PATH" \
MOLD_BIN="$tmp/bin/fake-mold" \
MOLD_FAKE_STATE="$tmp/state" \
MOLD_REGRESSION_OUT="$tmp/run" \
MOLD_REGRESSION_RUN_ID=batch \
MOLD_REGRESSION_BATCH=4 \
MOLD_REGRESSION_TIMEOUT_IMAGE=10 \
"$repo_root/scripts/regression-matrix.sh" > "$tmp/stdout" 2> "$tmp/stderr"

max_concurrency="$(<"$tmp/state/max")"
if (( max_concurrency < 4 )); then
  echo "expected four matrix cases to be queued concurrently, saw $max_concurrency" >&2
  echo "--- stdout ---" >&2
  cat "$tmp/stdout" >&2
  echo "--- stderr ---" >&2
  cat "$tmp/stderr" >&2
  exit 1
fi

echo "regression matrix queued $max_concurrency cases concurrently"
