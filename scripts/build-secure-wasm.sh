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
# Panic locations and generated Rust paths otherwise encode the builder's
# checkout, registry and installed rust-src paths into the shipped WASM.
# Keep bytes stable across canonical Linux checkout and installation paths.
# Keep the default aligned with the secure-protocol CI compiler.
wasm_toolchain="${MOLD_WASM_TOOLCHAIN:-1.98.0}"
if [[ "$(rustc +"$wasm_toolchain" --version)" != "rustc 1.98.0 "* ]]; then
  echo "Browser crypto requires Rust 1.98.0, matching secure-protocol CI." >&2
  exit 1
fi
wasm_sysroot=$(rustc +"$wasm_toolchain" --print sysroot)
wasm_rust_info=$(rustc +"$wasm_toolchain" -vV)
wasm_rust_host=$(printf '%s\n' "$wasm_rust_info" | sed -n 's/^host: //p')
if [[ "$wasm_rust_host" != x86_64-unknown-linux-gnu ]]; then
  echo "Browser crypto artifacts require the canonical x86_64-unknown-linux-gnu builder." >&2
  echo "On x86_64 Linux, install Rust 1.98.0 with target wasm32-unknown-unknown and wasm-bindgen-cli $version (--locked), then run this script." >&2
  echo "Other hosts consume the checked-in artifacts; native Apple builds remain supported. See crates/mold-secure/README.md." >&2
  exit 1
fi
wasm_rust_revision=$(printf '%s\n' "$wasm_rust_info" | sed -n 's/^commit-hash: //p')
wasm_flags=(
  --cfg 'getrandom_backend="wasm_js"'
  "--remap-path-prefix=$PWD=/mold"
  "--remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}=/cargo"
  "--remap-path-prefix=$wasm_sysroot/lib/rustlib/src/rust=/rustc/$wasm_rust_revision"
)
# The encoded form preserves paths containing spaces and takes precedence over
# ambient target flags without depending on the caller's shell quoting.
printf -v CARGO_ENCODED_RUSTFLAGS '%s\x1f' "${wasm_flags[@]}"
export CARGO_ENCODED_RUSTFLAGS="${CARGO_ENCODED_RUSTFLAGS%$'\x1f'}"
cargo +"$wasm_toolchain" build --locked -p mold-ai-secure --features wasm --target wasm32-unknown-unknown --release
wasm-bindgen "${CARGO_TARGET_DIR:-target}/wasm32-unknown-unknown/release/mold_secure.wasm" \
  --target web --out-dir studio/api/generated/secure --out-name mold_secure
if [[ "${1:-}" == "--check" ]]; then
  git diff --exit-code -- studio/api/generated/secure
fi
