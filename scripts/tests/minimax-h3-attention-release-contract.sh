#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

require_text() {
  local file=$1
  local text=$2
  local message=$3
  grep -Fq "$text" "$file" || fail "$message"
}

require_text crates/mold-candle/Cargo.toml \
  'h3-flash-attn-rc = ["dep:candle-flash-attn", "cuda"]' \
  "mold-candle does not expose the isolated H3 kernel feature"
require_text crates/mold-candle/Cargo.toml \
  'flash-attn = ["h3-flash-attn-rc"]' \
  "the existing global developer feature does not share the reviewed H3 primitive"
require_text crates/mold-inference/Cargo.toml \
  'h3-attention-rc = ["cuda", "mold-candle/h3-flash-attn-rc"]' \
  "mold-inference does not expose the synthetic-only H3 qualification path"
require_text crates/mold-inference/Cargo.toml \
  'required-features = ["dev-bins", "h3-attention-rc"]' \
  "the H3 qualification probe is reachable without both opt-in features"

if grep -Eq '^h3-(flash-)?attention-rc[[:space:]]*=' crates/mold-cli/Cargo.toml; then
  fail "mold-ai must not forward the H3 release-candidate feature into a runnable binary"
fi

release_commands="$({
  grep -E 'cargo build --release .*--features' .github/workflows/release.yml Dockerfile
  sed -n '/^[[:space:]]*releaseFeatures =/,/^[[:space:]]*completionFeatures =/p' flake.nix
} || true)"
if grep -Eq '(^|[,[:space:]])(h3-attention-rc|h3-flash-attn-rc|flash-attn)([,[:space:]]|$)' \
  <<< "$release_commands"; then
  fail "a published release feature set compiles an H3/FlashAttention candidate"
fi

require_text .github/workflows/ci.yml \
  'cargo clippy -p mold-ai-candle --features h3-flash-attn-rc --all-targets -- -D warnings' \
  "CI does not compile the exact H3 kernel crate and feature"
require_text .github/workflows/ci.yml \
  'cargo clippy -p mold-ai-inference --features h3-attention-rc,dev-bins --bin h3_attention_qualification -- -D warnings' \
  "CI does not compile the synthetic H3 qualification executable"
flash_filter="$(sed -n '/^            flash:/,/^            website:/p' .github/workflows/ci.yml)"
for path in \
  crates/mold-candle/Cargo.toml \
  crates/mold-candle/src/minimax_h3/attention.rs \
  crates/mold-inference/Cargo.toml \
  crates/mold-inference/src/bin/h3_attention_qualification.rs; do
  grep -Fq "'$path'" <<< "$flash_filter" \
    || fail "FlashAttention CI classifier omits $path"
done

require_text scripts/verify-cuda-release-binary.sh \
  '"$h3_release_exclusion" "$binary"' \
  "CUDA archive verification does not enforce the H3 candidate exclusion"
require_text Dockerfile \
  'RUN scripts/verify-h3-release-exclusion.sh /build/target/release/mold' \
  "the published container does not enforce the H3 candidate exclusion"

scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT
printf 'ordinary-published-mold\n' > "$scratch_dir/ordinary"
scripts/verify-h3-release-exclusion.sh "$scratch_dir/ordinary" >/dev/null
printf 'mold.minimax-h3.attention-rc.kernel-compiled.v1\n' > "$scratch_dir/claimed"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/claimed" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted an H3 release-candidate claim marker"
fi

echo "PASS: MiniMax H3 attention release-candidate contract"
