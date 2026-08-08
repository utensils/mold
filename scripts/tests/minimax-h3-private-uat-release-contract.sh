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
  'h3-private-uat = []' \
  "mold-candle does not keep the private H3 runtime behind its own feature"
require_text crates/mold-inference/Cargo.toml \
  'h3-private-uat = ["mold-candle/h3-private-uat"]' \
  "mold-inference does not narrowly forward the private H3 runtime feature"
require_text crates/mold-inference/Cargo.toml \
  'required-features = ["dev-bins", "h3-private-uat"]' \
  "the private H3 artifact qualifier is reachable without both development features"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  '"mold.minimax-h3.private-uat-artifact-reader.v1"' \
  "the private H3 artifact reader has no release-rejectable claim marker"
require_text crates/mold-inference/src/minimax_h3/private_qwen_support.rs \
  '"mold.minimax-h3.private-uat-qwen-support-loader.v1"' \
  "the private H3 Qwen support loader has no release-rejectable claim marker"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'pub const H3_PRIVATE_HOST_AUTHORITY_SHA256: &str =' \
  "private H3 qualification is not bound to an opaque authorized-host identity"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'pub const H3_PRIVATE_AUTHORIZATION_SCOPE: &str = "private-h3-uat";' \
  "private H3 qualification does not use the generic private authorization scope"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  '"/storage/jamesbrink/mold-uat/minimax-h3/models"' \
  "private H3 qualification is not bound to the reviewed storage root"

if grep -Eq '^h3-private-uat[[:space:]]*=' crates/mold-cli/Cargo.toml; then
  fail "mold-ai forwards the private H3 UAT feature into a runnable product binary"
fi
server_dev_dependencies=$(sed -n '/^\[dev-dependencies\]/,/^\[/p' crates/mold-server/Cargo.toml)
if ! grep -Eq '^mold-inference[[:space:]]*=.*path[[:space:]]*=[[:space:]]*"\.\./mold-inference".*package[[:space:]]*=[[:space:]]*"mold-ai-inference".*features[[:space:]]*=[[:space:]]*\["h3-private-uat"\]' \
  <<<"$server_dev_dependencies"; then
  fail "mold-server tests do not unify the private H3 loader for bridge verification"
fi
server_dependencies=$(sed -n '/^\[dependencies\]/,/^\[/p' crates/mold-server/Cargo.toml)
if grep -Eq '^mold-inference[[:space:]]*=.*h3-private-uat' <<<"$server_dependencies"; then
  fail "mold-ai-server activates the private H3 UAT path through an ordinary dependency"
fi
if sed -n '/^\[features\]/,/^\[/p' crates/mold-server/Cargo.toml \
  | grep -Eq '^h3-private-uat[[:space:]]*='; then
  fail "mold-ai-server forwards the private H3 UAT feature into a runnable product binary"
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
if LC_ALL=C tr -cs '[:alnum:]_-' '\n' <<<"$release_feature_sources" \
  | grep -Fxq h3-private-uat; then
  fail "a published release feature set compiles the private H3 UAT path"
fi

require_text scripts/verify-h3-release-exclusion.sh \
  "private_uat_marker='mold.minimax-h3.private-uat-artifact-reader.v1'" \
  "published binary verification does not reject the private H3 UAT marker"
require_text scripts/verify-h3-release-exclusion.sh \
  "private_qwen_support_marker='mold.minimax-h3.private-uat-qwen-support-loader.v1'" \
  "published binary verification does not reject the private H3 Qwen support marker"
require_text .github/workflows/ci.yml \
  'cargo clippy -p mold-ai-inference --features dev-bins,h3-private-uat --bin h3_artifact_qualification -- -D warnings' \
  "CI does not compile the authorization-bound artifact qualifier"
require_text .github/workflows/ci.yml \
  'cargo test -p mold-ai-inference --lib --features h3-private-uat minimax_h3' \
  "CI does not execute the private H3 foundation tests"
require_text .github/workflows/ci.yml \
  "'crates/mold-inference/src/minimax_h3/private_*.rs'" \
  "private H3 runtime source changes do not trigger release exclusion checks"
require_text .github/workflows/ci.yml \
  "'crates/mold-inference/src/minimax_h3/mod.rs'" \
  "private H3 module-gating changes do not trigger release exclusion checks"
require_text .github/workflows/ci.yml \
  'run: bash scripts/tests/minimax-h3-private-uat-release-contract.sh' \
  "release CI does not run the private H3 exclusion contract"
require_text docs/qualification/minimax-h3.md \
  'h3_artifact_qualification' \
  "the private artifact qualifier has no operator runbook"

scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT
ordinary_marker='mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=omitted'
private_marker='mold.minimax-h3.private-uat-artifact-reader.v1'
private_qwen_support_marker='mold.minimax-h3.private-uat-qwen-support-loader.v1'
printf '%s\n' "$ordinary_marker" >"$scratch_dir/ordinary"
scripts/verify-h3-release-exclusion.sh "$scratch_dir/ordinary" >/dev/null
printf '%s\n%s\n' "$ordinary_marker" "$private_marker" >"$scratch_dir/private"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the private H3 artifact reader"
fi
printf '%s\n%s\n' "$ordinary_marker" "$private_qwen_support_marker" \
  >"$scratch_dir/private-qwen-support"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private-qwen-support" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the private H3 Qwen support loader"
fi

echo "PASS: MiniMax H3 private-UAT release contract"
