#!/usr/bin/env bash
# Install the pinned Linux amd64 release without Docker Hub authentication.
set -euo pipefail

destination=${1:?usage: install-actionlint.sh DESTINATION_DIRECTORY}
archive=actionlint_1.7.12_linux_amd64.tar.gz
checksum=8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

curl --fail --location --retry 3 \
  "https://github.com/rhysd/actionlint/releases/download/v1.7.12/$archive" \
  --output "$scratch/$archive"
if ! printf '%s  %s\n' "$checksum" "$scratch/$archive" | sha256sum --check --status; then
  echo 'actionlint checksum mismatch; refusing extraction' >&2
  exit 1
fi
tar -xzf "$scratch/$archive" -C "$scratch" actionlint
mkdir -p "$destination"
install -m 755 "$scratch/actionlint" "$destination/actionlint"
