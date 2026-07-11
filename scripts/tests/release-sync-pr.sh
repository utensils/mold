#!/usr/bin/env bash
# Tests for scripts/release/sync-release-pr.sh: changelog promotion, link-ref
# rewriting, desktop version sync, and idempotency — against a fixture tree.
set -euo pipefail

script="$(cd "$(dirname "$0")/../release" && pwd)/sync-release-pr.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fail() { echo "FAIL: $1" >&2; exit 1; }

mkdir -p "$tmp/desktop/src-tauri"

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

cat > "$tmp/desktop/src-tauri/Cargo.toml" <<'EOF'
[package]
name = "mold-desktop"
version = "0.14.0"
edition = "2021"

[dependencies]
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

"$script" "$tmp" > /dev/null

grep -q '^## \[Unreleased\]$' "$tmp/CHANGELOG.md" || fail "[Unreleased] heading missing after promotion"
grep -q "^## \[0.15.0\] - $(date -u +%Y-%m-%d)$" "$tmp/CHANGELOG.md" || fail "promoted version heading missing"
grep -q '^\[Unreleased\]: https://github.com/utensils/mold/compare/v0.15.0...HEAD$' "$tmp/CHANGELOG.md" || fail "Unreleased link ref not rewritten"
grep -q '^\[0.15.0\]: https://github.com/utensils/mold/compare/v0.14.0...v0.15.0$' "$tmp/CHANGELOG.md" || fail "0.15.0 link ref missing"
# The unreleased content must now sit under the 0.15.0 heading.
awk '/^## \[0.15.0\]/{s=1} s && /A new thing/{found=1} END{exit !found}' "$tmp/CHANGELOG.md" || fail "unreleased content not under promoted heading"
# [Unreleased] section must be empty (next heading follows immediately).
awk '/^## \[Unreleased\]$/{getline; getline; exit ($0 ~ /^## \[0.15.0\]/) ? 0 : 1}' "$tmp/CHANGELOG.md" || fail "[Unreleased] section not empty"

grep -q '^version = "0.15.0"$' "$tmp/desktop/src-tauri/Cargo.toml" || fail "desktop Cargo.toml version not synced"
grep -q '^edition = "2021"$' "$tmp/desktop/src-tauri/Cargo.toml" || fail "desktop Cargo.toml collateral damage"
grep -q '"version": "0.15.0"' "$tmp/desktop/package.json" || fail "desktop package.json version not synced"
awk '/^name = "mold-desktop"$/{getline; exit ($0 == "version = \"0.15.0\"") ? 0 : 1}' "$tmp/desktop/src-tauri/Cargo.lock" || fail "Cargo.lock mold-desktop version not synced"
awk '/^name = "mold-ai-core"$/{getline; exit ($0 == "version = \"0.14.0\"") ? 0 : 1}' "$tmp/desktop/src-tauri/Cargo.lock" || fail "Cargo.lock touched an unrelated package"
awk '/^name = "serde"$/{getline; exit ($0 == "version = \"1.0.0\"") ? 0 : 1}' "$tmp/desktop/src-tauri/Cargo.lock" || fail "Cargo.lock touched serde"

# Idempotency: second run must not change anything.
cp -R "$tmp" "$tmp.before"
"$script" "$tmp" > /dev/null
diff -r "$tmp.before" "$tmp" > /dev/null || fail "second run was not a no-op"
rm -rf "$tmp.before"

echo "PASS: sync-release-pr"
