#!/usr/bin/env bash
# Supply release-plz's supported local registry-manifest baseline without
# packaging Git-only dependencies or comparing against retired crates.io.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 DESTINATION" >&2; exit 1; }
baseline=$1
[[ ! -e "$baseline" ]] || { echo "baseline destination already exists" >&2; exit 1; }
# Only stable release tags reachable from this revision can be a baseline.
tag=$(git tag --merged HEAD --sort=-version:refname | sed -n '/^v[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*$/p' | sed -n '1p')
[[ -n "$tag" ]] || { echo "no reachable stable release tag; refusing registry fallback" >&2; exit 1; }
manifest=$(git show "refs/tags/$tag:Cargo.toml")
version=$(sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\([^"]*\)"/\1/p' <<< "$manifest" | sed -n '1p')
[[ "v$version" == "$tag" ]] || { echo "release tag $tag does not match its workspace version" >&2; exit 1; }
git worktree add --detach "$baseline" "refs/tags/$tag" >&2
printf '%s/Cargo.toml\n' "$baseline"
