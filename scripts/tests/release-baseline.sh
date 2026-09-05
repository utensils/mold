#!/usr/bin/env bash
set -euo pipefail
script="$(cd "$(dirname "${BASH_SOURCE[0]}")/../release" && pwd)/prepare-release-baseline.sh"
parser="$(dirname "$script")/release-pr-output.sh"
[[ $(printf '{"prs":[]}' | "$parser") == '{}' ]]
[[ $(printf '{"prs":[{"number":42,"head_branch":"release-plz-next"}]}' | "$parser") == '{"number":42,"head_branch":"release-plz-next"}' ]]
for invalid in '{}' '{"prs":[{}]}' '{"prs":[{"number":42,"head_branch":""}]}' 'not json'; do
  if printf '%s' "$invalid" | "$parser" > /dev/null 2>&1; then
    echo "FAIL: malformed release output accepted" >&2; exit 1
  fi
done
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/repo"
cd "$scratch/repo"
git init -q
git config user.name Test
git config user.email test@example.invalid
printf '[workspace.package]\nversion = "0.27.1"\n' > Cargo.toml
git add Cargo.toml
git commit -qm 'chore: release v0.27.1'
if "$script" "$scratch/missing" > /dev/null 2>&1; then
  echo 'FAIL: missing release tag fell back silently' >&2; exit 1
fi
git tag v0.27.1
git tag v0.28.0-rc.1
# A higher tag on an unmerged branch must not become the baseline.
git checkout -qb future
printf '[workspace.package]\nversion = "9.0.0"\n' > Cargo.toml
git commit -qam 'chore: future release'
git tag v9.0.0
git checkout -q -
printf '[workspace.package]\nversion = "0.28.0"\n' > Cargo.toml
git commit -qam 'feat: next release candidate'
manifest=$("$script" "$scratch/baseline")
[[ "$manifest" == "$scratch/baseline/Cargo.toml" ]]
grep -qx 'version = "0.27.1"' "$manifest"
grep -qx 'version = "0.28.0"' Cargo.toml
[[ $(git -C "$scratch/baseline" rev-parse HEAD) == "$(git rev-parse v0.27.1)" ]]
if "$script" "$scratch/baseline" > /dev/null 2>&1; then
  echo 'FAIL: existing baseline was replaced' >&2; exit 1
fi
git worktree remove "$scratch/baseline"
git tag v0.29.0
if "$script" "$scratch/mismatch" > /dev/null 2>&1; then
  echo 'FAIL: mismatched release tag accepted' >&2; exit 1
fi
echo 'PASS: tagged release baseline contract'
