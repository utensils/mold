#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/bin"
cat >"$scratch/bin/timeout" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
while [[ "${1:-}" == --* ]]; do shift; done
shift # duration
H3_TEST_TIMEOUT_ACTIVE=1 exec "$@"
FAKE
cat >"$scratch/bin/cargo" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
[[ "${RUST_TEST_THREADS:-}" == "${EXPECTED_THREADS:-8}" ]]
[[ -d "$XDG_CONFIG_HOME" && "$XDG_CONFIG_HOME" != "$ORIGINAL_CONFIG" ]]
[[ -z "${MOLD_HOME+x}${MOLD_MODELS_DIR+x}${MOLD_DB_PATH+x}${MOLD_OFFLOAD+x}" ]]
case "$*" in
  'test --locked -p mold-ai-server --lib --features h3-cuda --no-run')
    [[ -z "${H3_TEST_TIMEOUT_ACTIVE+x}" ]]
    printf 'compile\n' >>"$CALL_LOG"
    ;;
  'test --locked -p mold-ai-server --lib --features h3-cuda -- fixture_test')
    [[ "${H3_TEST_TIMEOUT_ACTIVE:-}" == 1 ]]
    printf 'run\n' >>"$CALL_LOG"
    exit "${FIXTURE_EXIT:-0}"
    ;;
  *) exit 99 ;;
esac
FAKE
chmod +x "$scratch/bin/cargo" "$scratch/bin/timeout"
export PATH="$scratch/bin:$PATH" ORIGINAL_CONFIG="$scratch/config" CALL_LOG="$scratch/calls"
export XDG_CONFIG_HOME="$ORIGINAL_CONFIG" MOLD_HOME=/real/store MOLD_MODELS_DIR=/real/models
export MOLD_DB_PATH=/real/store/mold.db MOLD_OFFLOAD=1
env -u RUST_TEST_THREADS bash "$repo_root/scripts/test-h3-cuda-server.sh" fixture_test
[[ "$(cat "$CALL_LOG")" == $'compile\nrun' ]]
: >"$CALL_LOG"
RUST_TEST_THREADS=3 EXPECTED_THREADS=3 bash "$repo_root/scripts/test-h3-cuda-server.sh" fixture_test
[[ "$(cat "$CALL_LOG")" == $'compile\nrun' ]]
: >"$CALL_LOG"
if FIXTURE_EXIT=17 env -u RUST_TEST_THREADS bash "$repo_root/scripts/test-h3-cuda-server.sh" fixture_test; then
  echo 'H3 test wrapper swallowed a test failure' >&2
  exit 1
fi
[[ "$(cat "$CALL_LOG")" == $'compile\nrun' ]]
echo 'H3 CUDA server test isolation contract OK'
