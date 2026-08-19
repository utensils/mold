#!/usr/bin/env bash
# Tests for scripts/release/sync-release-pr.sh: changelog promotion, link-ref
# rewriting, desktop version sync, and idempotency — against a fixture tree.
set -euo pipefail

script="$(cd "$(dirname "$0")/../release" && pwd)/sync-release-pr.sh"
render_script="$(cd "$(dirname "$0")/../release" && pwd)/render-release-pr-body.sh"
workflow="$(cd "$(dirname "$0")/../.." && pwd)/.github/workflows/release-plz.yml"
ci_workflow="$(cd "$(dirname "$0")/../.." && pwd)/.github/workflows/ci.yml"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fail() { echo "FAIL: $1" >&2; exit 1; }

mkdir -p "$tmp/desktop/src-tauri" "$tmp/apps/mobile/src-tauri"

cat > "$tmp/Cargo.toml" <<'EOF'
[workspace]
members = ["crates/x"]

[workspace.package]
version = "0.15.0"
edition = "2021"
EOF

cat > "$tmp/CHANGELOG.md" <<'EOF'
# Changelog

## [Unreleased]

### Added

- A new thing.

## [0.14.0] - 2026-07-04

### Added

- Old thing.

[Unreleased]: https://github.com/utensils/mold/compare/v0.14.0...HEAD
[0.14.0]: https://github.com/utensils/mold/compare/v0.13.1...v0.14.0
EOF

# Per-PR changelog fragments: assembled into the promoted section, then deleted.
# README.md is documentation, never a fragment.
mkdir -p "$tmp/changelog.d"
cat > "$tmp/changelog.d/README.md" <<'EOF'
# fragments live here
EOF
cat > "$tmp/changelog.d/alpha-feature.md" <<'EOF'
- **Alpha fragment.** Landed as a fragment.
EOF
cat > "$tmp/changelog.d/beta-fix.md" <<'EOF'
- **Beta fragment.** Also a fragment,
  with a continuation line.
EOF

cat > "$tmp/desktop/src-tauri/Cargo.toml" <<'EOF'
[package]
name = "mold-desktop"
version = "0.14.0"
edition = "2021"

[dependencies]
mold-core = { path = "../../crates/mold-core", package = "mold-ai-core", version = "0.14.0" }
mold-server = { path = "../../crates/mold-server", package = "mold-ai-server", version = "0.14.0", features = [
  "preview",
] }
serde = { version = "1", features = ["derive"] }
EOF

cat > "$tmp/desktop/src-tauri/Cargo.lock" <<'EOF'
version = 4

[[package]]
name = "mold-ai-core"
version = "0.14.0"

[[package]]
name = "mold-desktop"
version = "0.14.0"
dependencies = [
 "serde",
]

[[package]]
name = "serde"
version = "1.0.0"
EOF

cat > "$tmp/desktop/package.json" <<'EOF'
{
  "name": "mold-desktop",
  "version": "0.14.0",
  "private": true
}
EOF

cat > "$tmp/apps/mobile/src-tauri/Cargo.toml" <<'EOF'
[package]
name = "mold-mobile"
version = "0.14.0"
edition = "2024"
EOF

cat > "$tmp/apps/mobile/src-tauri/Cargo.lock" <<'EOF'
version = 4

[[package]]
name = "mold-mobile"
version = "0.14.0"
EOF

cat > "$tmp/apps/mobile/src-tauri/tauri.conf.json" <<'EOF'
{
  "productName": "Mold",
  "version": "0.14.0",
  "identifier": "com.utensils.mold"
}
EOF

"$script" "$tmp" > /dev/null

grep -q '^## \[Unreleased\]$' "$tmp/CHANGELOG.md" || fail "[Unreleased] heading missing after promotion"
grep -q "^## \[0.15.0\] - $(date -u +%Y-%m-%d)$" "$tmp/CHANGELOG.md" || fail "promoted version heading missing"
grep -q '^\[Unreleased\]: https://github.com/utensils/mold/compare/v0.15.0...HEAD$' "$tmp/CHANGELOG.md" || fail "Unreleased link ref not rewritten"
grep -q '^\[0.15.0\]: https://github.com/utensils/mold/compare/v0.14.0...v0.15.0$' "$tmp/CHANGELOG.md" || fail "0.15.0 link ref missing"
# The unreleased content must now sit under the 0.15.0 heading.
awk '/^## \[0.15.0\]/{s=1} s && /A new thing/{found=1} END{exit !found}' "$tmp/CHANGELOG.md" || fail "unreleased content not under promoted heading"
# [Unreleased] section must be empty (next heading follows immediately).
awk '/^## \[Unreleased\]$/{getline; getline; exit ($0 ~ /^## \[0.15.0\]/) ? 0 : 1}' "$tmp/CHANGELOG.md" || fail "[Unreleased] section not empty"
# Fragments are assembled under the promoted heading (above any inline
# [Unreleased] entries), consumed, and the directory README survives.
awk '/^## \[0.15.0\]/{s=1} s && /Alpha fragment/{found=1} END{exit !found}' "$tmp/CHANGELOG.md" || fail "alpha fragment not under promoted heading"
awk '/^## \[0.15.0\]/{s=1} s && /with a continuation line/{found=1} END{exit !found}' "$tmp/CHANGELOG.md" || fail "multi-line fragment body lost"
awk '/Beta fragment/{b=NR} /A new thing/{n=NR} END{exit (b && n && b < n) ? 0 : 1}' "$tmp/CHANGELOG.md" || fail "fragments must precede inline [Unreleased] entries"
awk '/^## \[0.14.0\]/{s=1} s && /fragment/{found=1} END{exit found}' "$tmp/CHANGELOG.md" || fail "fragments leaked into an older release section"
test ! -e "$tmp/changelog.d/alpha-feature.md" || fail "assembled fragment not deleted"
test ! -e "$tmp/changelog.d/beta-fix.md" || fail "assembled fragment not deleted"
test -e "$tmp/changelog.d/README.md" || fail "changelog.d README was consumed as a fragment"
test "$(grep -c 'Alpha fragment' "$tmp/CHANGELOG.md")" -eq 1 || fail "fragment duplicated"

grep -q '^version = "0.15.0"$' "$tmp/desktop/src-tauri/Cargo.toml" || fail "desktop Cargo.toml version not synced"
grep -q '^edition = "2021"$' "$tmp/desktop/src-tauri/Cargo.toml" || fail "desktop Cargo.toml collateral damage"
grep -q 'package = "mold-ai-core", version = "0.15.0"' "$tmp/desktop/src-tauri/Cargo.toml" || fail "mold-ai-core dep requirement not synced"
grep -q 'package = "mold-ai-server", version = "0.15.0", features' "$tmp/desktop/src-tauri/Cargo.toml" || fail "mold-ai-server dep requirement not synced (inline attrs lost?)"
grep -q 'serde = { version = "1", features' "$tmp/desktop/src-tauri/Cargo.toml" || fail "serde dep requirement was touched"
grep -q '"version": "0.15.0"' "$tmp/desktop/package.json" || fail "desktop package.json version not synced"
awk '/^name = "mold-desktop"$/{getline; exit ($0 == "version = \"0.15.0\"") ? 0 : 1}' "$tmp/desktop/src-tauri/Cargo.lock" || fail "Cargo.lock mold-desktop version not synced"
awk '/^name = "mold-ai-core"$/{getline; exit ($0 == "version = \"0.15.0\"") ? 0 : 1}' "$tmp/desktop/src-tauri/Cargo.lock" || fail "Cargo.lock mold-ai-core version not synced"
awk '/^name = "serde"$/{getline; exit ($0 == "version = \"1.0.0\"") ? 0 : 1}' "$tmp/desktop/src-tauri/Cargo.lock" || fail "Cargo.lock touched serde"
grep -q '^version = "0.15.0"$' "$tmp/apps/mobile/src-tauri/Cargo.toml" || fail "mobile Cargo.toml version not synced"
awk '/^name = "mold-mobile"$/{getline; exit ($0 == "version = \"0.15.0\"") ? 0 : 1}' "$tmp/apps/mobile/src-tauri/Cargo.lock" || fail "mobile Cargo.lock version not synced"
grep -q '"version": "0.15.0"' "$tmp/apps/mobile/src-tauri/tauri.conf.json" || fail "mobile tauri config version not synced"

# Idempotency: second run must not change anything.
cp -R "$tmp" "$tmp.before"
"$script" "$tmp" > /dev/null
diff -r "$tmp.before" "$tmp" > /dev/null || fail "second run was not a no-op"
rm -rf "$tmp.before"

echo "PASS: sync-release-pr"

# The promoted release notes must also replace release-plz's empty changelog
# placeholder in the PR body without disturbing its package summary/footer.
cat > "$tmp/pr-body.md" <<'EOF'
## 🤖 New release

* `mold-ai`: 0.14.0 -> 0.15.0

<details><summary><i><b>Changelog</b></i></summary><p>


</p></details>

---
This PR was generated with release-plz.
EOF

"$render_script" "$tmp/CHANGELOG.md" 0.15.0 "$tmp/pr-body.md" > "$tmp/rendered-body.md"

# The literal backticks are Markdown, not shell.
# shellcheck disable=SC2016
grep -q '^\* `mold-ai`: 0.14.0 -> 0.15.0$' "$tmp/rendered-body.md" || fail "PR package summary was lost"
grep -q '^- A new thing\.$' "$tmp/rendered-body.md" || fail "promoted changelog entry missing from PR body"
grep -q '^This PR was generated with release-plz\.$' "$tmp/rendered-body.md" || fail "PR footer was lost"
test "$(grep -c 'A new thing' "$tmp/rendered-body.md")" -eq 1 || fail "PR changelog entry duplicated"

# A first release has no following version heading. Link-reference definitions
# still belong to the changelog file, not the release notes embedded in the PR.
cat > "$tmp/first-release-changelog.md" <<'EOF'
# Changelog

## [Unreleased]

## [0.1.0] - 2026-07-13

### Added

- First release.

[Unreleased]: https://github.com/utensils/mold/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/utensils/mold/releases/tag/v0.1.0
EOF
"$render_script" "$tmp/first-release-changelog.md" 0.1.0 "$tmp/pr-body.md" > "$tmp/first-release-body.md"
grep -q '^- First release\.$' "$tmp/first-release-body.md" || fail "first-release notes missing from PR body"
if grep -q '^\[Unreleased\]:' "$tmp/first-release-body.md"; then
  fail "changelog link references leaked into first-release PR body"
fi

# The workflow must operate on release-plz's exact output and keep every
# follow-up commit attributable to the GitHub App bot. A human-attributed
# second commit makes release-plz close/recreate the PR on the next push.
grep -q 'id: release-plz' "$workflow" || fail "release-plz step has no id"
grep -q 'steps.release-plz.outputs.pr' "$workflow" || fail "workflow does not use the exact release-plz PR output"
grep -q '302347651+release-plz-mold\[bot\]@users.noreply.github.com' "$workflow" || fail "follow-up commit is not attributed to the app bot"
grep -q 'render-release-pr-body.sh' "$workflow" || fail "workflow does not populate the PR changelog body"
# The dollar sign is part of the workflow source being asserted.
# shellcheck disable=SC2016
grep -Fq 'if [ -z "$version" ]; then' "$workflow" || fail "workflow does not validate the workspace version"
grep -Fq 'actions: read' "$workflow" || fail "release tag job cannot inspect candidate workflows"
grep -Fq 'Detect an unpublished release candidate' "$workflow" || fail "release tag job does not distinguish ordinary main pushes"
grep -Fq 'Wait for exact Apple candidate delivery' "$workflow" || fail "release tag job does not wait for Apple candidates"
grep -Fq 'Desktop|push|Publish nightly update' "$workflow" || fail "release tag job does not gate on macOS publication"
grep -Fq 'iOS TestFlight|workflow_run|Build, upload, and validate' "$workflow" || fail "release tag job does not gate on TestFlight delivery"
# These dollar signs are literal workflow source assertions.
# shellcheck disable=SC2016
grep -Fq -- '--commit "$GITHUB_SHA"' "$workflow" || fail "release tag job does not bind candidates to the exact SHA"
# shellcheck disable=SC2016
grep -Fq 'select(.name == \"$job_name\")' "$workflow" || fail "release tag job gates on whole workflows instead of Apple delivery jobs"
grep -q 'bash scripts/tests/release-sync-pr.sh' "$ci_workflow" || fail "CI does not exercise release PR synchronization"
# The bot commit deletes consumed changelog.d fragments, so the trusted
# release-PR file-scope check must accept D for exactly that path.
grep -Fq 'changelog.d/*.md' "$ci_workflow" || fail "trusted release PR scope does not allow changelog.d fragment deletions"

echo "PASS: release PR body sync"
