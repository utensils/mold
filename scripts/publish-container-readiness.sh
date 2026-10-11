#!/usr/bin/env bash
set -euo pipefail

# First validated publication wins. Rebuilding one source with stable/nightly
# build arguments can legitimately yield another digest; never rewrite a pin.
[[ $# == 3 ]] || { echo "usage: $0 <github-repository> <source-sha> <manifest.json>" >&2; exit 64; }
repository="$1"
source_sha="$2"
manifest="$3"
[[ "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ && "$source_sha" =~ ^[0-9a-f]{40}$ ]] || exit 64
asset="mold-container-digests-$source_sha.json"
script_dir="$(cd "$(dirname "$0")" && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

validate() {
  local candidate="$1"
  jq -e --arg source "$source_sha" --arg repo "ghcr.io/$repository" '
    .schema_version == "mold.container-digests.v1" and .source_sha == $source
    and .release_tag == ("sha-" + $source) and .repository == $repo
    and (.targets | type == "object" and length == 6)
  ' "$candidate" >/dev/null
  local target
  for target in sm80 sm86 sm89 sm90 sm100 sm120; do
    jq --arg target "$target" '{target: $target, tag: .targets[$target].tag, digest: .targets[$target].digest}' \
      "$candidate" > "$scratch/$target.json"
  done
  "$script_dir/create-container-digest-manifest.sh" "$scratch/validated.json" \
    "ghcr.io/$repository" "sha-$source_sha" "$source_sha" "$scratch"/sm*.json
}

validate "$manifest"
cp "$manifest" "$scratch/$asset"
mkdir "$scratch/existing"
if gh release download latest --repo "$repository" --pattern "$asset" --dir "$scratch/existing"; then
  validate "$scratch/existing/$asset"
  echo "Preserved existing verified container readiness for $source_sha"
elif gh release upload latest "$scratch/$asset" --repo "$repository"; then
  echo "Published container readiness for $source_sha"
else
  # An upload conflict can be another publisher winning. An API/network error
  # is not success: only a fresh, completely validated existing asset is.
  gh release download latest --repo "$repository" --pattern "$asset" --dir "$scratch/existing"
  validate "$scratch/existing/$asset"
  echo "Preserved concurrently published container readiness for $source_sha"
fi
