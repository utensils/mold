#!/usr/bin/env bash
set -euo pipefail

[[ $# == 3 ]] || { echo "usage: $0 <github-repository> <source-sha> <manifest.json>" >&2; exit 64; }
repository="$1"
source_sha="$2"
manifest="$3"
[[ "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ && "$source_sha" =~ ^[0-9a-f]{40}$ ]] || exit 64
# The workflow serializes all promotions. Old scheduled runs and stable-tag
# reruns still publish their own readiness, but cannot downgrade latest.
main_sha="$(gh api "repos/$repository/git/ref/heads/main" --jq '.object.sha')"
if [[ "$main_sha" != "$source_sha" ]]; then
  echo "Source is no longer current main; leaving latest container aliases unchanged"
  exit 0
fi
for target in sm80 sm86 sm89 sm90 sm100 sm120; do
  suffix="-$target"
  [[ "$target" != sm89 ]] || suffix=""
  digest="$(jq -er --arg target "$target" '.targets[$target].digest' "$manifest")"
  [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 65
  image="ghcr.io/$repository"
  docker buildx imagetools create --prefer-index=false --tag "$image:latest$suffix" "$image@$digest"
  verified="$(docker buildx imagetools inspect "$image:latest$suffix" --format '{{json .Manifest.Digest}}' | tr -d '"')"
  [[ "$verified" == "$digest" ]] || { echo "latest container alias verification failed: $target" >&2; exit 1; }
done
