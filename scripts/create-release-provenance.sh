#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <output.json> <release-tag> <source-sha> <SHA256SUMS> <artifacts-dir>" >&2
  exit 64
fi

output="$1"
release_tag="$2"
source_sha="$3"
sums="$4"
artifacts_dir="$5"
[[ "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] \
  || { echo "invalid release tag: $release_tag" >&2; exit 64; }
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "invalid source SHA: $source_sha" >&2; exit 64; }
[[ -f "$sums" ]] || { echo "missing SHA256SUMS: $sums" >&2; exit 66; }
[[ -d "$artifacts_dir" ]] || { echo "missing artifacts directory: $artifacts_dir" >&2; exit 66; }

checksum_for() {
  local filename="$1"
  awk -v filename="$filename" '
    {
      candidate = $2
      sub(/^\*/, "", candidate)
      sub(/^\.\//, "", candidate)
    }
    candidate == filename {
      if (found++) exit 2
      value = $1
    }
    END {
      if (found != 1 || value !~ /^[0-9a-f]{64}$/) exit 1
      print value
    }
  ' "$sums"
}

if awk '
  {
    candidate = $2
    sub(/^\*/, "", candidate)
    sub(/^\.\//, "", candidate)
  }
  candidate == "mold-release-provenance.json" { found = 1 }
  END { exit !found }
' "$sums"; then
  echo "SHA256SUMS already contains release provenance; generate provenance from artifact-only checksums, then append its checksum" >&2
  exit 65
fi

sm86_name="mold-x86_64-unknown-linux-gnu-cuda-sm86.tar.gz"
sm89_name="mold-x86_64-unknown-linux-gnu-cuda-sm89.tar.gz"
sm86_archive_sha="$(checksum_for "$sm86_name")" \
  || { echo "SHA256SUMS lacks one valid $sm86_name entry" >&2; exit 65; }
sm89_archive_sha="$(checksum_for "$sm89_name")" \
  || { echo "SHA256SUMS lacks one valid $sm89_name entry" >&2; exit 65; }
[[ "$sm86_archive_sha" != "$sm89_archive_sha" ]] \
  || { echo "sm86 and sm89 release archives unexpectedly have the same checksum" >&2; exit 65; }
for archive in "$artifacts_dir/$sm86_name" "$artifacts_dir/$sm89_name"; do
  [[ -f "$archive" ]] || { echo "missing release archive: $archive" >&2; exit 66; }
done
[[ "$(sha256sum "$artifacts_dir/$sm86_name" | awk '{print $1}')" == "$sm86_archive_sha" ]] \
  || { echo "$sm86_name does not match SHA256SUMS" >&2; exit 65; }
[[ "$(sha256sum "$artifacts_dir/$sm89_name" | awk '{print $1}')" == "$sm89_archive_sha" ]] \
  || { echo "$sm89_name does not match SHA256SUMS" >&2; exit 65; }
sm86_binary_sha="$(tar -xOzf "$artifacts_dir/$sm86_name" mold | sha256sum | awk '{print $1}')"
sm89_binary_sha="$(tar -xOzf "$artifacts_dir/$sm89_name" mold | sha256sum | awk '{print $1}')"
[[ "$sm86_binary_sha" != "$sm89_binary_sha" ]] \
  || { echo "sm86 and sm89 release binaries unexpectedly have the same checksum" >&2; exit 65; }

jq -n \
  --arg release_tag "$release_tag" \
  --arg source_sha "$source_sha" \
  --arg sm86_name "$sm86_name" \
  --arg sm86_archive_sha "$sm86_archive_sha" \
  --arg sm86_binary_sha "$sm86_binary_sha" \
  --arg sm89_name "$sm89_name" \
  --arg sm89_archive_sha "$sm89_archive_sha" \
  --arg sm89_binary_sha "$sm89_binary_sha" '
    {
      schema_version: "mold.release.provenance.v1",
      release_tag: $release_tag,
      source_sha: $source_sha,
      artifacts: {
        sm86: {
          filename: $sm86_name,
          archive_sha256: $sm86_archive_sha,
          binary_sha256: $sm86_binary_sha,
          cuda_target: "sm_86"
        },
        sm89: {
          filename: $sm89_name,
          archive_sha256: $sm89_archive_sha,
          binary_sha256: $sm89_binary_sha,
          cuda_target: "sm_89"
        }
      }
    }
  ' >"$output"
