#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workflow="$repo_root/.github/workflows/desktop.yml"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

grep -Fq \
  "group: desktop-\${{ github.event_name == 'pull_request' && format('pr-{0}', github.event.pull_request.number) || github.event_name == 'push' && github.run_attempt == 1 && 'primary' || github.run_id }}" \
  "$workflow" \
  || fail "workflow-level concurrency does not isolate PRs, reruns, and primary pushes"

workflow_concurrency="$(sed -n '/^concurrency:/,/^permissions:/p' "$workflow")"
grep -Fq "cancel-in-progress: \${{ github.event_name == 'pull_request' }}" <<< "$workflow_concurrency" \
  || fail "workflow-level concurrency does not cancel only superseded PR runs"

rust_job="$(sed -n '/^  desktop-rust:/,/^  desktop-nightly:/p' "$workflow")"
grep -Fq 'run: bunx tauri build --debug --bundles app --ci' <<< "$rust_job" \
  || fail "bundle-sensitive macOS PRs have no packaging proof"
grep -Fq "if: github.event_name == 'pull_request' && needs.changes.outputs.bundle == 'true'" <<< "$rust_job" \
  || fail "debug bundle is not restricted to bundle-sensitive PRs"
grep -Fq 'run: cargo test --manifest-path src-tauri/Cargo.toml' <<< "$rust_job" \
  || fail "macOS PR gate no longer runs the native test suite"

nightly_header="$(sed -n '/^  desktop-nightly:/,/^  publish-desktop-nightly:/p' "$workflow")"
grep -Fq 'needs: [desktop-frontend, desktop-rust]' <<< "$nightly_header" \
  || fail "macOS Nightly distribution does not start after its own frontend and Rust gates"
if grep -Fq 'desktop-linux' <<< "$nightly_header"; then
  fail "macOS Nightly distribution is still blocked on the Linux AppImage job"
fi

publisher_header="$(sed -n '/^  publish-desktop-nightly:/,/^    steps:/p' "$workflow")"
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

echo "PASS: desktop-nightly-race-guards"
