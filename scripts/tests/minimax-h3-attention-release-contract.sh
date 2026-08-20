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
require_text crates/mold-candle/src/minimax_h3/attention.rs \
  'pub fn verify_current_release_candidate_dispatch' \
  "private H3 dispatch does not revalidate the current compiled attention candidate"
require_text crates/mold-candle/src/minimax_h3/attention.rs \
  'H3AttentionReleaseCandidateBuild::current()' \
  "attention dispatch trusts a serialized authority without current-build provenance"
require_text crates/mold-inference/Cargo.toml \
  'h3-attention-rc = ["cuda", "mold-candle/h3-flash-attn-rc"]' \
  "mold-inference does not expose the synthetic-only H3 qualification path"
require_text crates/mold-inference/Cargo.toml \
  'required-features = ["dev-bins", "h3-attention-rc"]' \
  "the H3 qualification probe is reachable without both opt-in features"
benchmark_stanza="$(sed -n \
  '/name = "h3_attention_benchmark"/,/required-features =/p' \
  crates/mold-inference/Cargo.toml)"
grep -Fq 'path = "src/bin/h3_attention_benchmark.rs"' <<<"$benchmark_stanza" \
  || fail "the H3 packed-row benchmark binary is not registered"
grep -Fq 'required-features = ["dev-bins", "h3-attention-rc"]' <<<"$benchmark_stanza" \
  || fail "the H3 packed-row benchmark is reachable without both opt-in features"

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

# Since #1164 the bare `h3` feature implies neither CUDA nor the SM89
# attention kernel, so a recipe still written as `--features cuda,h3` builds a
# CUDA H3 binary with no H3 kernel. `crates/mold-server/build_support/
# h3_server_features.rs` refuses that graph at build time; this check refuses
# the recipe at review time, before anyone waits out a release build to learn
# it from a build script.
#
# The scan parses the actual feature list out of each `--features` invocation
# and tests for the exact token `h3`, rather than matching the line. A looser
# pattern matched the test filter `minimax_h3` and the comments that name the
# anti-pattern, which is precisely the kind of false positive that gets a
# guard deleted.
enabled_feature_lists() {
  grep -Eho -- '--features[= ]+"?[A-Za-z0-9_,-]+' "$@" \
    | sed -E 's/^--features[= ]+"?//'
}

h3_recipe_files=(
  .github/workflows/release.yml
  .github/workflows/desktop.yml
  Dockerfile
  packaging/aur/mold-ai/PKGBUILD
  packaging/aur/mold-ai-git/PKGBUILD
  scripts/ci-local.sh
)

while IFS= read -r feature_list; do
  [[ -n "$feature_list" ]] || continue
  while IFS= read -r feature; do
    [[ "$feature" == "h3" ]] \
      && fail "a shipping recipe enables the bare h3 feature ('$feature_list'); SM89 recipes must name h3-cuda"
  done < <(tr ',' '\n' <<< "$feature_list")
done < <(enabled_feature_lists "${h3_recipe_files[@]}")

for fixture in 'cuda,h3,preview' 'h3' 'dev-bins,h3'; do
  grep -qx 'h3' <<< "$(tr ',' '\n' <<< "$fixture")" \
    || fail "bare-h3 scanner missed fixture: $fixture"
done
for fixture in 'h3-cuda,preview' 'dev-bins,h3-private-uat' 'cuda,h3-attention-rc'; do
  if grep -qx 'h3' <<< "$(tr ',' '\n' <<< "$fixture")"; then
    fail "bare-h3 scanner rejected allowed fixture: $fixture"
  fi
done

# Nix composes its feature strings rather than passing `--features`, so its two
# helpers are checked on the literal they yield for SM89.
require_text flake.nix \
  'if computeCap == "89" then "h3-cuda" else "cuda"' \
  "a flake feature helper no longer selects the h3-cuda edge for SM89"
if grep -Eq '"cuda,h3"|,h3"' flake.nix; then
  fail "flake.nix still composes a bare cuda,h3 feature string"
fi

# Positive proof that each SM89 route still names the edge, so deleting an H3
# line from a recipe cannot pass by leaving nothing to match.
for recipe in "${h3_recipe_files[@]}" flake.nix desktop/src-tauri/Cargo.toml; do
  grep -Fq 'h3-cuda' "$recipe" \
    || fail "$recipe no longer names the h3-cuda shipping edge"
done

require_text crates/mold-server/build_support/h3_server_features.rs \
  '"CARGO_FEATURE_H3_CUDA",' \
  "the mold-server build fence does not require the h3-cuda shipping edge"
require_text crates/mold-server/Cargo.toml \
  'h3-cuda = ["h3", "cuda", "mold-inference/h3-cuda"]' \
  "mold-server does not expose the h3-cuda shipping edge"
require_text crates/mold-inference/Cargo.toml \
  'h3-cuda = ["h3", "cuda", "h3-attention-rc"]' \
  "the h3-cuda edge does not carry the SM89 attention kernel"
# The inverse guard: re-coupling a device to the bare feature is what made an
# Apple Silicon H3 build inexpressible in the first place.
if grep -Eq '^h3 = \[[^]]*"cuda"' crates/mold-inference/Cargo.toml \
  crates/mold-server/Cargo.toml crates/mold-cli/Cargo.toml \
  desktop/src-tauri/Cargo.toml; then
  fail "the bare h3 feature re-couples a device, making an Apple Silicon build inexpressible"
fi

omitted_marker='mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=omitted'
claim_marker='mold.minimax-h3.attention-rc.kernel-compiled.v1'
private_qwen_support_marker='mold.minimax-h3.private-uat-qwen-support-loader.v1'
h3_compiled_marker='mold.minimax-h3.attention-release-provenance.v2:h3-rc=compiled:global-flash=omitted'
public_qwen_support_marker='mold.minimax-h3.qwen-support-loader.v1'
global_compiled_markers=(
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
  fail "release verifier accepted an H3 claim without H3 provenance"
fi
printf '%s\n%s\n' "$omitted_marker" "$public_qwen_support_marker" > "$scratch_dir/ordinary-with-support"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/ordinary-with-support" >/dev/null 2>&1; then
  fail "release verifier accepted public H3 Qwen support provenance without H3 provenance"
fi
printf '%s\n%s\n%s\n' "$h3_compiled_marker" "$claim_marker" "$public_qwen_support_marker" > "$scratch_dir/public-h3"
scripts/verify-h3-release-exclusion.sh "$scratch_dir/public-h3" >/dev/null
printf '%s\n' "$h3_compiled_marker" > "$scratch_dir/public-h3-missing-claim"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/public-h3-missing-claim" >/dev/null 2>&1; then
  fail "release verifier accepted H3 provenance without its kernel claim"
fi
printf '%s\n%s\n' "$h3_compiled_marker" "$claim_marker" > "$scratch_dir/public-h3-missing-support"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/public-h3-missing-support" >/dev/null 2>&1; then
  fail "release verifier accepted H3 provenance without public Qwen support provenance"
fi
printf '%s\n%s\n' "$omitted_marker" "$private_qwen_support_marker" > "$scratch_dir/private-qwen-support"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private-qwen-support" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the private H3 Qwen support loader"
fi
for index in "${!global_compiled_markers[@]}"; do
  printf '%s\n%s\n' "$omitted_marker" "${global_compiled_markers[$index]}" \
    > "$scratch_dir/compiled-$index"
  if scripts/verify-h3-release-exclusion.sh "$scratch_dir/compiled-$index" >/dev/null 2>&1; then
    fail "release verifier accepted global FlashAttention provenance"
  fi
done

echo "PASS: MiniMax H3 attention release-candidate contract"
