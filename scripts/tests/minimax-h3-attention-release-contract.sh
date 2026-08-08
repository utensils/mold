#!/usr/bin/env bash
# shellcheck disable=SC2016 # Literal workflow/Nix source is asserted below.
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
  'flash-attn = ["dep:candle-flash-attn", "cuda"]' \
  "the existing global developer feature is not independent from H3 qualification"
if grep -Eq '^flash-attn[[:space:]]*=.*h3-flash-attn-rc' crates/mold-candle/Cargo.toml; then
  fail "the global FlashAttention feature aliases the isolated H3 release candidate"
fi
if grep -Fq 'feature = "h3-flash-attn-rc"' \
  crates/mold-candle/src/minimax_h3/visual_vae.rs; then
  fail "H3 qualification changes Visual VAE attention dispatch"
fi
require_text crates/mold-candle/src/minimax_h3/visual_vae.rs \
  '#[cfg(feature = "flash-attn")]' \
  "Visual VAE no longer keys FlashAttention off the global developer feature"
if ! awk '
  previous == "#[cfg(feature = \"h3-flash-attn-rc\")]" &&
    $0 ~ /^fn flash_attention\(q: &Tensor/ { found = 1 }
  { previous = $0 }
  END { exit(found ? 0 : 1) }
' crates/mold-candle/src/minimax_h3/attention.rs; then
  fail "the isolated H3 feature does not directly guard the production FlashAttention dispatch"
fi
require_text crates/mold-inference/Cargo.toml \
  'h3-attention-rc = ["cuda", "mold-candle/h3-flash-attn-rc"]' \
  "mold-inference does not expose the synthetic-only H3 qualification path"
require_text crates/mold-inference/Cargo.toml \
  'required-features = ["dev-bins", "h3-attention-rc"]' \
  "the H3 qualification probe is reachable without both opt-in features"

if grep -Eq '^h3-(flash-)?attention-rc[[:space:]]*=' crates/mold-cli/Cargo.toml; then
  fail "mold-ai must not forward the H3 release-candidate feature into a runnable binary"
fi

release_feature_sources="$({
  grep -E -- '--features|releaseFeatures[[:space:]]*=|buildFeatures[[:space:]]*=' \
    .github/workflows/release.yml \
    .github/workflows/desktop.yml \
    Dockerfile \
    flake.nix \
    packaging/aur/mold-ai/PKGBUILD \
    packaging/aur/mold-ai-git/PKGBUILD
  sed -n '/^[[:space:]]*releaseFeatures =/,/^[[:space:]]*completionFeatures =/p' flake.nix
} || true)"

contains_forbidden_release_feature() {
  LC_ALL=C tr -cs '[:alnum:]_-' '\n' \
    | grep -Fx \
      -e h3-attention-rc \
      -e h3-flash-attn-rc \
      -e flash-attn \
      >/dev/null
}

for fixture in \
  'cargo build --release --features=h3-attention-rc' \
  'cargo build --release --features "cuda,h3-flash-attn-rc"' \
  'cargo build --release --features "cuda,flash-attn"' \
  'releaseFeatures = "cuda,h3-attention-rc"' \
  'releaseFeatures = [ "cuda" "h3-flash-attn-rc" ]'; do
  contains_forbidden_release_feature <<< "$fixture" \
    || fail "release feature scanner missed fixture: $fixture"
done
for fixture in \
  'cargo build --release --features=cuda,preview' \
  'releaseFeatures = "cuda,preview,discord"'; do
  if contains_forbidden_release_feature <<< "$fixture"; then
    fail "release feature scanner rejected allowed fixture: $fixture"
  fi
done

if contains_forbidden_release_feature <<< "$release_feature_sources"; then
  fail "a published release feature set compiles an H3/FlashAttention candidate"
fi

require_text .github/workflows/ci.yml \
  'cargo clippy -p mold-ai-candle --features h3-flash-attn-rc --all-targets -- -D warnings' \
  "CI does not compile the exact H3 kernel crate and feature"
require_text .github/workflows/ci.yml \
  'cargo clippy -p mold-ai-inference --features h3-attention-rc,dev-bins --bin h3_attention_qualification -- -D warnings' \
  "CI does not compile the synthetic H3 qualification executable"
flash_job="$(awk '
  $0 == "  flash-attn-check:" { selected = 1 }
  selected && $0 ~ /^  [[:alnum:]_-][[:alnum:]_-]*:$/ &&
    $0 != "  flash-attn-check:" { exit }
  selected { print }
' .github/workflows/ci.yml)"
grep -Fq "if: github.event_name == 'push'" <<< "$flash_job" \
  || fail "FlashAttention proof is not deferred from PRs to every main update"

release_filter="$(awk '
  $0 == "            release:" { selected = 1 }
  selected && $0 ~ /^            [[:alnum:]_-][[:alnum:]_-]*:$/ &&
    $0 != "            release:" { exit }
  selected { print }
' .github/workflows/ci.yml)"
for path in \
  crates/mold-candle/src/minimax_h3/mod.rs \
  crates/mold-candle/src/minimax_h3/dit.rs \
  crates/mold-candle/src/minimax_h3/visual_vae.rs; do
  grep -Fq "'$path'" <<< "$release_filter" \
    || fail "release CI classifier omits $path"
done

require_text scripts/verify-cuda-release-binary.sh \
  '"$h3_release_exclusion" "$binary"' \
  "CUDA archive verification does not enforce the H3 candidate exclusion"
require_text Dockerfile \
  'RUN scripts/verify-h3-release-exclusion.sh /build/target/release/mold' \
  "the published container does not enforce the H3 candidate exclusion"
for target in 86 89 100 120; do
  require_text .github/workflows/release.yml \
    "scripts/verify-cuda-release-binary.sh target/release/mold $target" \
    "the native sm$target publication route bypasses H3 exclusion verification"
done
require_text flake.nix \
  '${pkgs.bash}/bin/bash ${./scripts/verify-h3-release-exclusion.sh} "$out/bin/mold"' \
  "Nix CUDA CLI outputs bypass H3 exclusion verification"
require_text flake.nix \
  '${pkgs.bash}/bin/bash ${./scripts/verify-h3-release-exclusion.sh} "$desktop_bin"' \
  "Nix CUDA desktop outputs bypass H3 exclusion verification"
require_text .github/workflows/desktop.yml \
  '../scripts/verify-h3-release-exclusion.sh "$desktop_binary"' \
  "the Linux CUDA AppImage route bypasses H3 exclusion verification"
for pkgbuild in packaging/aur/mold-ai/PKGBUILD packaging/aur/mold-ai-git/PKGBUILD; do
  require_text "$pkgbuild" \
    './scripts/verify-h3-release-exclusion.sh "target/release/${_binname}"' \
    "$pkgbuild bypasses H3 exclusion verification"
done

omitted_marker='mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=omitted'
claim_marker='mold.minimax-h3.attention-rc.kernel-compiled.v1'
private_qwen_support_marker='mold.minimax-h3.private-uat-qwen-support-loader.v1'
compiled_markers=(
  'mold.minimax-h3.attention-release-provenance.v2:h3-rc=compiled:global-flash=omitted'
  'mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=compiled'
  'mold.minimax-h3.attention-release-provenance.v2:h3-rc=compiled:global-flash=compiled'
)

scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT
printf '%s\n' "$omitted_marker" > "$scratch_dir/ordinary"
scripts/verify-h3-release-exclusion.sh "$scratch_dir/ordinary" >/dev/null
printf 'ordinary-published-mold\n' > "$scratch_dir/missing-provenance"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/missing-provenance" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted missing compile-time provenance"
fi
printf '%s\n%s\n' "$omitted_marker" "$claim_marker" > "$scratch_dir/claimed"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/claimed" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted an H3 release-candidate claim marker"
fi
printf '%s\n%s\n' "$omitted_marker" "$private_qwen_support_marker" > "$scratch_dir/private-qwen-support"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private-qwen-support" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the private H3 Qwen support loader"
fi
for index in "${!compiled_markers[@]}"; do
  printf '%s\n%s\n' "$omitted_marker" "${compiled_markers[$index]}" \
    > "$scratch_dir/compiled-$index"
  if scripts/verify-h3-release-exclusion.sh "$scratch_dir/compiled-$index" >/dev/null 2>&1; then
    fail "release exclusion verifier accepted compiled FlashAttention provenance"
  fi
done

echo "PASS: MiniMax H3 attention release-candidate contract"
