#!/usr/bin/env bash
# Build the same reviewed Rust protocol into the browser/Tauri distribution.
set -euo pipefail
cd "$(dirname "$0")/.."
version=$(python3 - <<'PY'
import tomllib
with open('Cargo.lock', 'rb') as f:
    print(next(p['version'] for p in tomllib.load(f)['package'] if p['name'] == 'wasm-bindgen'))
PY
)
if ! command -v wasm-bindgen >/dev/null || [[ "$(wasm-bindgen --version)" != "wasm-bindgen $version" ]]; then
  echo "Install the locked generator: cargo +stable install wasm-bindgen-cli --version $version --locked" >&2
  exit 1
fi
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS='--cfg getrandom_backend="wasm_js"'
cargo +"${MOLD_WASM_TOOLCHAIN:-stable}" build --locked -p mold-ai-secure --features wasm --target wasm32-unknown-unknown --release
wasm-bindgen "${CARGO_TARGET_DIR:-target}/wasm32-unknown-unknown/release/mold_secure.wasm" \
  --target web --out-dir studio/api/generated/secure --out-name mold_secure
if [[ "${1:-}" == "--check" ]]; then
  git diff --exit-code -- studio/api/generated/secure
fi
