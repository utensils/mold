#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workflow="$repo_root/.github/workflows/desktop.yml"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

grep -Fq \
  "group: desktop-\${{ github.event_name == 'pull_request' && format('pr-{0}', github.event.pull_request.number) || github.run_id }}" \
  "$workflow" \
  || fail "workflow-level concurrency does not isolate PRs while keeping main runs independent"
if grep -Fq "&& 'primary'" <<< "$(sed -n '/^concurrency:/,/^permissions:/p' "$workflow")"; then
  fail "main pushes still share a workflow-level queue that lets Linux block macOS"
fi

workflow_concurrency="$(sed -n '/^concurrency:/,/^permissions:/p' "$workflow")"
grep -Fq "cancel-in-progress: \${{ github.event_name == 'pull_request' }}" <<< "$workflow_concurrency" \
  || fail "workflow-level concurrency does not cancel only superseded PR runs"

# Packaging proof used to be a per-PR macOS debug bundle. #947 made desktop
# pull-request feedback format-only and moved every build to main, so the proof
# is asserted where it now lives — one job per platform, both gated to pushes —
# rather than deleted along with the step that used to carry it.
rust_job="$(sed -n '/^  desktop-rust:/,/^  desktop-linux:/p' "$workflow")"
grep -Fq 'run: cargo test --manifest-path src-tauri/Cargo.toml' <<< "$rust_job" \
  || fail "the native desktop gate no longer runs the test suite"

linux_job="$(sed -n '/^  desktop-linux:/,/^  desktop-nightly:/p' "$workflow")"
grep -Fq 'bunx tauri build --features h3-cuda,cudnn,pulid --bundles appimage --ci -v' <<< "$linux_job" \
  || fail "main pushes have no Linux packaging proof"
grep -Fq "if: github.event_name != 'pull_request'" <<< "$linux_job" \
  || fail "Linux packaging is not reserved for main pushes"

nightly_header="$(sed -n '/^  desktop-nightly:/,/^  publish-desktop-nightly:/p' "$workflow")"
grep -Fq 'uses: ./.github/workflows/desktop-distribution.yml' <<< "$nightly_header" \
  || fail "main pushes have no macOS packaging proof"
grep -Fq 'needs: [changes]' <<< "$nightly_header" \
  || fail "macOS Nightly distribution is not launched immediately after classification"
if grep -Fq 'desktop-linux' <<< "$nightly_header"; then
  fail "macOS Nightly distribution is still blocked on the Linux AppImage job"
fi

publisher_header="$(sed -n '/^  publish-desktop-nightly:/,/^    steps:/p' "$workflow")"
grep -Fq 'needs: [desktop-nightly, desktop-frontend, desktop-rust]' <<< "$publisher_header" \
  || fail "Nightly publication is not gated by distribution, frontend, and Rust validation"
grep -Fq 'group: desktop-nightly-publication' <<< "$publisher_header" \
  || fail "Nightly publication is not serialized"
grep -Fq 'cancel-in-progress: false' <<< "$publisher_header" \
  || fail "Nightly publication cancels an in-progress publisher"

mapfile -t guard_lines < <(
  grep -nF "if scripts/check-desktop-nightly-main-head.sh \"\$GITHUB_SHA\"; then" "$workflow" \
    | cut -d: -f1
)
[[ "${#guard_lines[@]}" -eq 2 ]] \
  || fail "expected two live-main guards, found ${#guard_lines[@]}"

payload_upload_line="$(grep -nF "gh release upload latest \"\$payload\" \"\$signature\" \"\$dmg\"" "$workflow" | cut -d: -f1)"
payload_verified_line="$(grep -nF 'nightly updater payload failed anonymous SHA-256 verification' "$workflow" | cut -d: -f1)"
manifest_upload_line="$(grep -nF "gh release upload latest \"\$manifest\"" "$workflow" | cut -d: -f1)"

[[ "${guard_lines[0]}" -lt "$payload_upload_line" ]] \
  || fail "first live-main guard does not run before payload upload"
[[ "${guard_lines[1]}" -gt "$payload_verified_line" ]] \
  || fail "second live-main guard does not run after payload verification"
[[ "${guard_lines[1]}" -lt "$manifest_upload_line" ]] \
  || fail "second live-main guard does not run immediately before manifest upload"

# Face identity ships in every desktop recipe (#1223).
#
# The embedded This-device server advertises `supports_identity` only when
# `pulid` is compiled, and `studio/lib/identityConditioning.ts` gates the
# identity photo well on that advertisement — absence reads as NO. So a
# desktop recipe that drops the feature does not degrade, it hides the whole
# feature permanently, silently, with nothing in the UI to explain it. Each
# published recipe is asserted by hand because there is no single build
# command to read.
grep -Fq 'pulid = ["mold-core/pulid", "mold-server/pulid"]' "$repo_root/desktop/src-tauri/Cargo.toml" \
  || fail "the desktop crate no longer forwards the pulid feature"
# The range starts at the binding name alone: nixfmt moves `computeCap:` onto
# its own line as soon as the recipe grows past one line, and anchoring on the
# joined form made this guard silently extract nothing and fail on the
# assertion below rather than on the thing it is guarding.
desktop_feature_recipe="$(sed -n '/desktopFeaturesFor =/,/;/p' "$repo_root/flake.nix")"
grep -Fq '"pulid"' <<< "$desktop_feature_recipe" \
  || fail "the Nix desktop feature recipe no longer builds face identity"
grep -Fq 'buildFeatures = desktopFeaturesFor computeCap;' "$repo_root/flake.nix" \
  || fail "the Nix desktop packages no longer use the shared desktop feature recipe"
# Every `cargo tauri` invocation in the devshell — dev, build, and the signed
# `desktop-release` distribution path a maintainer runs by hand — must take the
# shared recipe. `desktop-release` hard-coded `--features metal` and would have
# produced a signed DMG with face identity missing, while the CI-built nightly
# from the same tree had it.
while IFS= read -r invocation; do
  grep -Fq -- '--features ${desktopFeatures}' <<< "$invocation" \
    || fail "a devshell cargo tauri command does not use the shared desktop feature recipe: $invocation"
done < <(grep -F 'cargo tauri ' "$repo_root/flake.nix")
grep -Fq 'pkgs.protobuf' "$repo_root/flake.nix" \
  || fail "the Nix build has no protoc for candle-onnx"
distribution="$repo_root/.github/workflows/desktop-distribution.yml"
grep -Fq 'bunx tauri build --features metal,pulid --bundles app --ci --config' "$distribution" \
  || fail "the signed macOS desktop build no longer ships face identity"
grep -Fq 'brew install minisign protobuf' "$distribution" \
  || fail "the signed macOS desktop build has no protoc for candle-onnx"

echo "PASS: desktop-nightly-race-guards"
