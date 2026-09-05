#!/usr/bin/env bash
# Mold's Git-pinned backend cannot be represented by crates.io dependencies.
# Keep this guard on the existing release-contract route and local CI runner.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
fail() { echo "FAIL: $1" >&2; exit 1; }

[[ ! -e scripts/release/publish-crates.sh ]] \
  || fail "retired crates.io publisher must not remain executable"
if grep -En 'cargo[[:space:]]+publish|CARGO_REGISTRY_TOKEN|publish-crates\.sh' .github/workflows/*.yml; then
  fail "a workflow reintroduced crates.io publishing"
fi
grep -qx 'publish = false' release-plz.toml \
  || fail "release-plz must keep registry publishing disabled"
# Retired registry versions must never suppress the next release PR. Every
# lockstep crate reads the same existing vX.Y.Z baseline, including libraries
# which do not emit their own tags.
workspace_config=$(sed -n '/^\[workspace\]/,/^\[\[package\]\]/p' release-plz.toml)
grep -q -- '--registry-manifest-path' .github/workflows/release-plz.yml \
  || fail "release-plz must compare against the tagged workspace"
grep -qx 'git_tag_name = "v{{ version }}"' <<< "$workspace_config" \
  || fail "all lockstep crates must read the shared version tag"
# Track the canonical docs, including nested website and agent references.
# Git's file list avoids scanning generated site output or node_modules.
while IFS= read -r doc; do
  if grep -En 'cargo[[:space:]]+install[[:space:]]+mold-ai([^[:alnum:]_-]|$)' "$doc"; then
    fail "$doc advertises the retired registry distribution"
  fi
done < <(git ls-files -- README.md CLAUDE.md '.claude/rules/*.md' \
  'website/*.md' 'docs/*.md' 'crates/mold-cli/src/skill/*.md' \
  '.github/workflows/*.yml')
if grep -Eq '^[[:space:]]*publish[[:space:]]*=[[:space:]]*true' release-plz.toml; then
  fail "release-plz must not override registry publishing for a package"
fi
# Preserve the supported release jobs while removing the registry job.
for job in release-version release-native release-containers publish-aur; do
  grep -q "^  $job:" .github/workflows/release.yml \
    || fail "supported distribution job $job is missing"
done

bash scripts/tests/release-baseline.sh

echo "PASS: crates.io retirement contract"
