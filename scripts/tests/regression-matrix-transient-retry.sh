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
    "name": "flux-transient:q8",
    "family": "flux",
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
{"entries":[]}
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

# Fail the first base attempt only. Base and source run concurrently, so a
# shared check-then-create stamp could fail both and make this test flaky.
if [[ "$output" == *.base.png ]] && mkdir "$state_dir/failed-once" 2>/dev/null; then
  echo 'server error: generation error: DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")' >&2
  exit 1
fi

printf 'fake output\n' > "$output"
FAKE
chmod +x "$tmp/bin/fake-mold"

PATH="$tmp/bin:$PATH" \
MOLD_BIN="$tmp/bin/fake-mold" \
MOLD_FAKE_STATE="$tmp/state" \
MOLD_REGRESSION_OUT="$tmp/run" \
MOLD_REGRESSION_RUN_ID=transient-retry \
MOLD_REGRESSION_TIMEOUT_IMAGE=10 \
"$repo_root/scripts/regression-matrix.sh" > "$tmp/stdout" 2> "$tmp/stderr"

run_dir="$tmp/run/transient-retry"
if [[ -s "$run_dir/errors.jsonl" || -s "$run_dir/FAILED_CASES.tsv" ]]; then
  echo "expected transient retry to avoid final aggregate failure" >&2
  exit 1
fi

if [[ "$(jq -r 'select(.status | startswith("retry:")) | [.model, .case] | join(":")' "$run_dir/results.jsonl")" != "flux-transient:q8:base" ]]; then
  echo "expected retry status to be recorded" >&2
  exit 1
fi

ok_count="$(jq -r 'select(.status == "ok") | .model' "$run_dir/results.jsonl" | wc -l)"
if (( ok_count != 2 )); then
  echo "expected retried case to finish ok" >&2
  exit 1
fi

if ! jq -e 'select(.status | startswith("attempt-failed:"))' "$run_dir/attempt-errors.jsonl" > /dev/null; then
  echo "expected failed attempt to be recorded" >&2
  exit 1
fi

echo "regression matrix retries transient CUDA invalid-argument failures"
