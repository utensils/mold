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

cat > "$tmp/bin/timeout" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
state_dir="${MOLD_FAKE_STATE:?}"
timeout_s="$1"
shift
(
  flock 9
  min_timeout="$timeout_s"
  [[ -f "$state_dir/min-timeout" ]] && min_timeout="$(<"$state_dir/min-timeout")"
  if (( timeout_s < min_timeout )); then
    printf '%s\n' "$timeout_s" > "$state_dir/min-timeout"
  elif [[ ! -f "$state_dir/min-timeout" ]]; then
    printf '%s\n' "$timeout_s" > "$state_dir/min-timeout"
  fi
) 9>"$state_dir/timeout-lock"
exec "$@"
FAKE
chmod +x "$tmp/bin/timeout"

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
  started=0
  [[ -f "$state_dir/started" ]] && started="$(<"$state_dir/started")"
  printf '%s\n' "$((started + 1))" > "$state_dir/started"
) 9>"$state_dir/lock"

cleanup() {
  (
    flock 9
    current="$(<"$state_dir/current")"
    printf '%s\n' "$((current - 1))" > "$state_dir/current"
  ) 9>"$state_dir/lock"
}
trap cleanup EXIT

sleep 0.2
if [[ "$output" == *.lora1.png ]]; then
  echo "intentional lora1 failure" >&2
  exit 23
fi

printf 'fake output\n' > "$output"
FAKE
chmod +x "$tmp/bin/fake-mold"

set +e
PATH="$tmp/bin:$PATH" \
MOLD_BIN="$tmp/bin/fake-mold" \
MOLD_FAKE_STATE="$tmp/state" \
MOLD_REGRESSION_OUT="$tmp/run" \
MOLD_REGRESSION_RUN_ID=aggregate \
MOLD_REGRESSION_BATCH=2 \
MOLD_REGRESSION_TIMEOUT_IMAGE=10 \
"$repo_root/scripts/regression-matrix.sh" > "$tmp/stdout" 2> "$tmp/stderr"
status=$?
set -e

if (( status == 0 )); then
  echo "expected regression matrix to fail" >&2
  exit 1
fi

started="$(<"$tmp/state/started")"
if (( started != 4 )); then
  echo "expected all four matrix cases to be queued before failure reporting, saw $started" >&2
  echo "--- stdout ---" >&2
  cat "$tmp/stdout" >&2
  echo "--- stderr ---" >&2
  cat "$tmp/stderr" >&2
  exit 1
fi

errors="$tmp/run/aggregate/errors.jsonl"
failed_cases="$tmp/run/aggregate/FAILED_CASES.tsv"

if [[ ! -s "$errors" ]]; then
  echo "expected aggregate errors file: $errors" >&2
  exit 1
fi

if [[ ! -s "$failed_cases" ]]; then
  echo "expected aggregate failed-cases file: $failed_cases" >&2
  exit 1
fi

jq -e 'select(.status == "failed:23" and .case == "lora1" and (.stderr_tail | contains("intentional lora1 failure")))' "$errors" > /dev/null
grep -F $'flux-schnell:q8	flux	lora1	failed:23' "$failed_cases" > /dev/null

min_timeout="$(<"$tmp/state/min-timeout")"
if (( min_timeout <= 10 )); then
  echo "expected queue-wait allowance to increase per-case timeout above 10s, saw $min_timeout" >&2
  exit 1
fi

echo "regression matrix queued all cases and recorded aggregate failures"
