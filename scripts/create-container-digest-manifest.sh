#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <output.json> <repository> <release-tag> <source-sha> <fragment.json>..." >&2
  exit 64
fi

output="$1"
repository="$2"
release_tag="$3"
source_sha="$4"
shift 4

[[ "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] \
  || { echo "invalid release tag: $release_tag" >&2; exit 64; }
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "invalid source SHA: $source_sha" >&2; exit 64; }
for command in jq realpath; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 69; }
done

bare_version="${release_tag#v}"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
jq -s \
  --arg repository "$repository" \
  --arg release_tag "$release_tag" \
  --arg source_sha "$source_sha" \
  --arg bare_version "$bare_version" '
    def valid_digest:
      type == "string" and test("^sha256:[0-9a-f]{64}$");
    if length != 6 then
      error("expected exactly six CUDA target fragments")
    elif ([.[].target] | unique | sort) !=
      ["sm100", "sm120", "sm80", "sm86", "sm89", "sm90"] then
      error("CUDA target fragments are incomplete or duplicated")
    elif any(.[];
      (.digest | valid_digest | not)
      or .tag != ($bare_version + (if .target == "sm89" then "" else "-" + .target end))) then
      error("CUDA target fragment has an invalid digest or tag")
    else
      {
        schema_version: "mold.container-digests.v1",
        release_tag: $release_tag,
        source_sha: $source_sha,
        repository: $repository,
        targets: (map({key: .target, value: {tag: .tag, digest: .digest}}) | from_entries)
      }
    end
  ' "$@" >"$tmp"
mkdir -p "$(dirname "$output")"
mv "$tmp" "$output"
trap - EXIT
