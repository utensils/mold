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
if grep -En 'cargo install mold-ai([[:space:]]|$)' README.md website/guide/installation.md .github/workflows/release.yml; then
  fail "installation instructions advertise the retired registry distribution"
fi
if grep -Eq '^[[:space:]]*publish[[:space:]]*=[[:space:]]*true' release-plz.toml; then
  fail "release-plz must not override registry publishing for a package"
fi
# Preserve the supported release jobs while removing the registry job.
for job in release-version release-native release-containers publish-aur; do
  grep -q "^  $job:" .github/workflows/release.yml \
    || fail "supported distribution job $job is missing"
done

echo "PASS: crates.io retirement contract"
