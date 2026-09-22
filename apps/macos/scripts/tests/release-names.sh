#!/usr/bin/env bash
# The names the Makefile PRODUCES are the names the workflows UPLOAD.
#
# **Fails today**: `make dmg` named the file after `MARKETING_VERSION`
# ("Mold-0.1.0.dmg") while the distribution workflow computed a nightly
# version and then copied "Mold-0.1.0-nightly.<count>.dmg" -- a file that has
# never existed. Under `set -euo pipefail` the step died at the `cp`, and
# nightly is the DEFAULT of the dispatch input, so the first thing anybody ran
# was the thing that could not work. Nothing noticed, because no test compared
# the two sides.
#
# This asks MAKE itself (`make -n`, no build) rather than re-deriving the name
# here, and then greps the workflow for the same expression. Both sides, one
# check.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
MACOS="$(cd "$HERE/.." && pwd)"
WORKFLOWS="$(cd "$MACOS/../.." && pwd)/.github/workflows"

recipe() {
  # `make -n` prints the commands a target WOULD run.
  ( cd "$MACOS" && make -n "$@" 2>/dev/null )
}

# 1. Native and other interfaces share the release-plz workspace version.
marketing=$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version = "\([^"]*\)"/\1/p' "$MACOS/../../Cargo.toml" | head -1)
[ "$("$MACOS/scripts/workspace-version.sh")" = "$marketing" ] || {
  echo "FAIL: native version differs from workspace" >&2; exit 1;
}

expected="Mold-native-$marketing.dmg"
for target in dmg notarize; do
  if ! recipe "$target" | grep -q "$expected"; then
    echo "FAIL: \`make $target\` does not name $expected" >&2
    recipe "$target" >&2
    exit 1
  fi
done

# 2. The nightly case: ONE variable moves the bundle's version AND the
#    artifact name together, because the workflow sets exactly one.
nightly=$("$MACOS/../../scripts/create-desktop-nightly-version.sh" "$marketing" "$(git -C "$MACOS" rev-list --count HEAD)")
expected_nightly="Mold-native-$nightly.dmg"
for target in dmg notarize; do
  if ! recipe MARKETING_VERSION="$nightly" "$target" | grep -q "$expected_nightly"; then
    echo "FAIL: \`make $target MARKETING_VERSION=$nightly\` does not name $expected_nightly" >&2
    recipe MARKETING_VERSION="$nightly" "$target" >&2
    exit 1
  fi
done
# And it must reach the BUILD too, or the bundle keeps saying 0.1.0 while the
# DMG around it claims otherwise -- Sparkle shows the short version in its
# update alert, so that mismatch is what a user reads.
if ! recipe MARKETING_VERSION="$nightly" build | grep -q "MARKETING_VERSION=$nightly"; then
  echo "FAIL: MARKETING_VERSION does not reach xcodebuild" >&2
  exit 1
fi

# 3. The workflow expects exactly that name, built from the same one variable.
distribution="$WORKFLOWS/macos-native-distribution.yml"
[ -f "$distribution" ] || { echo "FAIL: no $distribution" >&2; exit 1; }
if ! grep -q 'dmg_name="Mold-native-\$version.dmg"' "$distribution"; then
  echo "FAIL: the workflow does not expect Mold-native-\$version.dmg" >&2
  exit 1
fi
if ! grep -q 'MARKETING_VERSION: \${{ steps.release.outputs.version }}' "$distribution"; then
  echo "FAIL: the workflow does not hand its resolved version to make" >&2
  exit 1
fi

# 4. No native workflow may create a GitHub release. The native artifacts ride
#    releases somebody else owns -- release.yml's `v*` and desktop.yml's
#    rolling `latest` -- and a `gh release create` here is how the repository's
#    Latest pointer gets taken away from the CLI download links and from the
#    Tauri app's own stable updater.
for workflow in "$WORKFLOWS"/macos-native*.yml; do
  if grep -q 'gh release create' "$workflow"; then
    echo "FAIL: $(basename "$workflow") creates a release; native assets ride existing ones" >&2
    exit 1
  fi
done

echo "release names ok ($expected, $expected_nightly)"

# Execute the actual workflow resolver, not a second implementation of it.
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for channel in stable nightly; do
  ruby -ryaml -e '
    doc = YAML.safe_load(File.read(ARGV[0]), aliases: true)
    step = doc["jobs"]["distribution"]["steps"].find { |s| s["id"] == "release" }
    puts step.fetch("run").gsub("${{ inputs.channel }}", ARGV[1])
  ' "$distribution" "$channel" > "$work/resolve.sh"
  (cd "$MACOS" && GITHUB_REF_TYPE=branch GITHUB_REF_NAME=main \
    GITHUB_OUTPUT="$work/$channel" GITHUB_ENV="$work/$channel-env" bash "$work/resolve.sh")
  expected_version="$marketing"
  [ "$channel" = stable ] || expected_version="$nightly"
  grep -Fx "version=$expected_version" "$work/$channel"
  grep -Fx "dmg_name=Mold-native-$expected_version.dmg" "$work/$channel"
done

# Future release bumps flow through automatically; malformed or missing
# workspace versions fail closed instead of creating an unnamed artifact.
fixture="$work/repository"
mkdir -p "$fixture/apps/macos/scripts"
cp "$MACOS/scripts/workspace-version.sh" "$fixture/apps/macos/scripts/"
printf '[workspace.package]\nversion = "9.8.7"\n' > "$fixture/Cargo.toml"
[ "$(bash "$fixture/apps/macos/scripts/workspace-version.sh")" = "9.8.7" ]
for bad in '' '0.31' '0.31.0-nightly.1' '01.2.3'; do
  printf '[workspace.package]\nversion = "%s"\n' "$bad" > "$fixture/Cargo.toml"
  if bash "$fixture/apps/macos/scripts/workspace-version.sh" > "$work/invalid-out" 2> "$work/invalid-error"; then
    echo "FAIL: native version accepted invalid workspace version '$bad'" >&2
    exit 1
  fi
  [ ! -s "$work/invalid-out" ]
  grep -q 'must be a plain three-part SemVer' "$work/invalid-error"
done
printf '[package]\nversion = "1.2.3"\n' > "$fixture/Cargo.toml"
if bash "$fixture/apps/macos/scripts/workspace-version.sh" > "$work/invalid-out" 2> "$work/invalid-error"; then
  echo 'FAIL: native version accepted a manifest without workspace.package.version' >&2
  exit 1
fi
echo 'native workspace version validation ok'
