#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ci="$repo_root/.github/workflows/ci.yml"
desktop="$repo_root/.github/workflows/desktop.yml"
ios="$repo_root/.github/workflows/ios.yml"

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

require_text "$ci" \
  "cancel-in-progress: \${{ github.event_name == 'pull_request' }}" \
  "root CI does not cancel superseded PR runs"
require_text "$ci" \
  "if: needs.changes.outputs.gpu == 'true'" \
  "backend checks are not routed through the GPU-specific classifier"
require_text "$ci" \
  "cargo clippy -p mold-ai --features cuda,preview,expand,tui,webp,mp4,mdns --all-targets -- -D warnings" \
  "CUDA-gated production code is not linted"
require_text "$ci" \
  "cargo clippy -p mold-ai --features metal,preview,expand,tui,webp,mp4,mdns --all-targets -- -D warnings" \
  "Metal-gated production code is not linted"
require_text "$ci" \
  "cargo clippy -p mold-ai --features flash-attn -- -D warnings" \
  "the flash-attn binary wiring is only typechecked"
flash_filter="$(sed -n '/^            flash:/,/^            website:/p' "$ci")"
if grep -Fq ".github/workflows/ci.yml" <<< "$flash_filter"; then
  fail "unrelated root CI edits still trigger the hour-long FlashAttention kernel build on PRs"
fi
require_text "$ci" \
  "if: github.event_name == 'push' || needs.changes.outputs.flash == 'true'" \
  "FlashAttention is not covered by every main push after narrowing its PR paths"
require_text "$ci" \
  "if: github.event_name == 'push' && needs.changes.outputs.rust == 'true'" \
  "coverage is not deferred to the post-merge main run"
release_filter="$(sed -n '/^            release:/,/^  release:/p' "$ci")"
for manifest in Cargo.toml Cargo.lock desktop/src-tauri/Cargo.toml desktop/src-tauri/Cargo.lock; do
  grep -Fxq "              - '$manifest'" <<< "$release_filter" \
    || fail "release classifier does not run the desktop Candle lock guard for $manifest changes"
done
for workflow in desktop.yml ios.yml; do
  grep -Fq ".github/workflows/$workflow" <<< "$release_filter" \
    || fail "release classifier does not run the routing contract for $workflow changes"
done

nix_filter="$(sed -n '/^            nix:/,/^            release:/p' "$ci")"
if grep -Fq "desktop/**" <<< "$nix_filter"; then
  fail "desktop-only changes still start the mold-web Nix build"
fi

desktop_classifier="$(sed -n '/^  changes:/,/^  desktop-frontend:/p' "$desktop")"
require_text "$desktop" "name: Classify desktop changes" \
  "desktop workflow has no path classifier"
require_text "$desktop" \
  "if: github.event_name != 'pull_request' || needs.changes.outputs.frontend == 'true'" \
  "desktop frontend job is not path-gated on PRs"
require_text "$desktop" \
  "if: github.event_name != 'pull_request' || needs.changes.outputs.rust == 'true'" \
  "desktop native jobs are not path-gated on PRs"
[[ "$(grep -Fc "if: github.event_name != 'pull_request' || needs.changes.outputs.updater == 'true'" "$desktop")" -eq 2 ]] \
  || fail "updater tooling and contract tests are not restricted to updater changes on PRs"
if grep -Fq "desktop/**" <<< "$desktop_classifier"; then
  fail "desktop frontend classifier is broad enough to include native/mobile files"
fi
[[ "$(grep -Fc 'desktop/src/components/**' "$desktop")" -eq 3 ]] \
  || fail "desktop frontend trigger/classifier does not track desktop components"
[[ "$(grep -Fc 'desktop/src-tauri/**' "$desktop")" -eq 3 ]] \
  || fail "desktop native trigger/classifier does not track the Tauri crate"
if grep -Fq "desktop/src/mobile/**" "$desktop"; then
  fail "mobile-only changes are not owned exclusively by the iOS workflow"
fi
[[ "$(grep -Fc 'crates/mold-scheduler/**' "$desktop")" -eq 3 ]] \
  || fail "desktop trigger/classifier does not track its scheduler dependency"

require_text "$ios" \
  "cancel-in-progress: \${{ github.event_name == 'pull_request' }}" \
  "iOS workflow does not cancel superseded PR runs"
require_text "$ios" "name: Classify iOS changes" \
  "iOS workflow has no path classifier"
require_text "$ios" \
  "if: github.event_name != 'pull_request' || needs.changes.outputs.rust == 'true'" \
  "iOS simulator job is not restricted to native changes on PRs"
require_text "$ios" "run: ../scripts/tests/ios-release-assets.sh" \
  "iOS frontend job does not validate the built mobile entry"

echo "PASS: CI routing contract"
