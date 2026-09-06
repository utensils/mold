#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/bin"
cat >"$scratch/bin/cargo" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
[[ "${RUST_TEST_THREADS:-}" == 8 ]]
[[ -d "$XDG_CONFIG_HOME" && "$XDG_CONFIG_HOME" != "$ORIGINAL_CONFIG" ]]
[[ -z "${MOLD_HOME+x}${MOLD_MODELS_DIR+x}${MOLD_DB_PATH+x}${MOLD_OFFLOAD+x}" ]]
[[ "$*" == 'test --locked -p mold-ai-server --lib --features h3-cuda -- fixture_test' ]]
exit "${FIXTURE_EXIT:-0}"
FAKE
chmod +x "$scratch/bin/cargo"
export PATH="$scratch/bin:$PATH" ORIGINAL_CONFIG="$scratch/config"
export XDG_CONFIG_HOME="$ORIGINAL_CONFIG" MOLD_HOME=/real/store MOLD_MODELS_DIR=/real/models
export MOLD_DB_PATH=/real/store/mold.db MOLD_OFFLOAD=1
bash "$repo_root/scripts/test-h3-cuda-server.sh" fixture_test
if FIXTURE_EXIT=17 bash "$repo_root/scripts/test-h3-cuda-server.sh" fixture_test; then
  echo 'H3 test wrapper swallowed a test failure' >&2
  exit 1
fi
echo 'H3 CUDA server test isolation contract OK'
