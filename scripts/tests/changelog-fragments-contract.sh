#!/usr/bin/env bash
# Tests for scripts/release/check-changelog-fragments.sh against a throwaway
# git repo, plus the CI wiring that runs it on pull requests.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/release/check-changelog-fragments.sh"
ci="$repo_root/.github/workflows/ci.yml"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fail() { echo "FAIL: $1" >&2; exit 1; }

cd "$tmp"
git init -q -b main
git config user.email t@example.com; git config user.name t
mkdir -p crates/x/src changelog.d
printf '# Changelog\n\n## [Unreleased]\n\n- Existing note.\n\n## [0.1.0] - 2026-01-01\n\n- Old.\n' > CHANGELOG.md
printf '# docs\n' > changelog.d/README.md
printf 'fn main() {}\n' > crates/x/src/lib.rs
git add -A && git commit -qm base
base=$(git rev-parse HEAD)

expect_ok()   { "$script" "$base" "$(git rev-parse HEAD)" >/dev/null 2>&1 || fail "$1"; }
expect_fail() { if "$script" "$base" "$(git rev-parse HEAD)" >/dev/null 2>&1; then fail "$1"; fi; }

# Source change + well-formed fragment: OK.
printf 'fn main() { let _ = 1; }\n' > crates/x/src/lib.rs
printf -- '- **Good.** A note\n  that continues.\n' > changelog.d/good.md
git add -A && git commit -qm "feat" 
expect_ok "source change with a fragment must pass"

# Source change without a fragment: FAIL; skip label: OK.
git checkout -q "$base" && git checkout -q -b nofrag
printf 'fn main() { let _ = 2; }\n' > crates/x/src/lib.rs
git add -A && git commit -qm "feat"
expect_fail "source change without a fragment must fail"
SKIP_CHANGELOG=true "$script" "$base" "$(git rev-parse HEAD)" >/dev/null 2>&1 || fail "skip-changelog must bypass the presence check"

# Docs-only change without a fragment: OK (nothing shipped).
git checkout -q "$base" && git checkout -q -b docsonly
printf 'hello\n' > NOTES.md
git add -A && git commit -qm "docs"
expect_ok "docs-only change must not require a fragment"

# Direct [Unreleased] edit: FAIL even with a fragment present.
git checkout -q "$base" && git checkout -q -b direct
printf '# Changelog\n\n## [Unreleased]\n\n- Sneaky direct edit.\n- Existing note.\n\n## [0.1.0] - 2026-01-01\n\n- Old.\n' > CHANGELOG.md
printf -- '- **Also a fragment.** x\n' > changelog.d/also.md
git add -A && git commit -qm "docs"
expect_fail "direct [Unreleased] edit must fail"

# Editing an older release section is fine (typo fixes), no source change.
git checkout -q "$base" && git checkout -q -b oldsection
printf '# Changelog\n\n## [Unreleased]\n\n- Existing note.\n\n## [0.1.0] - 2026-01-01\n\n- Old, typo fixed.\n' > CHANGELOG.md
git add -A && git commit -qm "docs"
expect_ok "editing an already-released section must pass"

# Malformed fragment (no bullet) / conflict markers: FAIL.
git checkout -q "$base" && git checkout -q -b badfrag
printf 'Just prose, no bullet\n' > changelog.d/bad.md
git add -A && git commit -qm "bad"
expect_fail "fragment without a leading bullet must fail"
git checkout -q "$base" && git checkout -q -b markers
printf -- '- **Ok.**\n<<<<<<< HEAD\n' > changelog.d/m.md
git add -A && git commit -qm "markers"
expect_fail "fragment with conflict markers must fail"

# README.md in changelog.d is never linted as a fragment.
git checkout -q "$base" && git checkout -q -b readme
printf '# docs updated\n' > changelog.d/README.md
git add -A && git commit -qm "readme"
expect_ok "changelog.d/README.md must not be linted as a fragment"

# Base is the branch TIP on GitHub: a PR that branched before main moved must
# be judged from the merge base. Simulate main advancing with a new
# [Unreleased] entry and a source change; an untouched docs-only PR must pass.
git checkout -q main
printf 'fn main() { let _ = 99; }\n' > crates/x/src/lib.rs
printf '# Changelog\n\n## [Unreleased]\n\n- Landed on main later.\n- Existing note.\n\n## [0.1.0] - 2026-01-01\n\n- Old.\n' > CHANGELOG.md
git add -A && git commit -qm "main moved"
newtip=$(git rev-parse HEAD)
git checkout -q docsonly
"$script" "$newtip" "$(git rev-parse HEAD)" >/dev/null 2>&1 || fail "a PR behind main must be judged against the merge base, not main's tip"

# Renaming a fragment that already exists on main is not a new release note.
git checkout -q "$newtip" -b withfrag
printf -- '- **Pending on main.** x\n' > changelog.d/pending.md
git add -A && git commit -qm "pending fragment on main"
frag_base=$(git rev-parse HEAD)
git checkout -q -b rename
git mv changelog.d/pending.md changelog.d/renamed.md
printf 'fn main() { let _ = 7; }\n' > crates/x/src/lib.rs
git add -A && git commit -qm "rename + source"
if "$script" "$frag_base" "$(git rev-parse HEAD)" >/dev/null 2>&1; then fail "renaming an existing fragment must not satisfy the presence check"; fi

# Nested fragments are never assembled by the sync script, so they are refused.
git checkout -q "$base" && git checkout -q -b nested
mkdir -p changelog.d/sub && printf -- '- **Nested.** x\n' > changelog.d/sub/n.md
printf 'fn main() { let _ = 3; }\n' > crates/x/src/lib.rs
git add -A && git commit -qm "nested"
expect_fail "nested fragment must fail"

# CRLF fragments are refused (they would leak \r into CHANGELOG.md).
git checkout -q "$base" && git checkout -q -b crlf
printf -- '- **CRLF.** x\r\n' > changelog.d/crlf.md
git add -A && git commit -qm "crlf"
expect_fail "CRLF fragment must fail"

# Non-ASCII slugs are still seen (git quotePath would otherwise hide them).
git checkout -q "$base" && git checkout -q -b unicode
printf -- '- **Naïve.** x\n' > "changelog.d/fix-naïve.md"
printf 'fn main() { let _ = 4; }\n' > crates/x/src/lib.rs
git add -A && git commit -qm "unicode"
expect_ok "a non-ASCII fragment filename must satisfy the presence check"

# A malformed fragment already on main must not fail an unrelated PR.
git checkout -q main
printf 'no bullet here\n' > changelog.d/bad-on-main.md
git add -A && git commit -qm "bad fragment on main"
bad_base=$(git rev-parse HEAD)
git checkout -q -b unrelated
printf 'fn main() { let _ = 5; }\n' > crates/x/src/lib.rs
printf -- '- **Mine.** fine\n' > changelog.d/mine.md
git add -A && git commit -qm "unrelated"
"$script" "$bad_base" "$(git rev-parse HEAD)" >/dev/null 2>&1 || fail "a bad fragment on main must not fail an unrelated PR"

# CI wiring: the check runs on pull requests, is skipped for the trusted
# release PR (which legitimately rewrites [Unreleased]), and honours the label.
grep -Fq 'scripts/release/check-changelog-fragments.sh' "$ci" || fail "CI does not run the changelog fragment check"
grep -Fq "contains(github.event.pull_request.labels.*.name, 'skip-changelog')" "$ci" || fail "CI does not wire the skip-changelog label"
awk '/^  changelog:$/{s=1} s && /trusted_release_pr/{found=1} s && /^  [a-z_-]+:$/ && !/^  changelog:$/{exit} END{exit !found}' "$ci" || fail "changelog job does not exclude the trusted release PR"

echo "PASS: changelog fragments contract"
