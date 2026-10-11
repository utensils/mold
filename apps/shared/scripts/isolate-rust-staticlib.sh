#!/usr/bin/env bash
# A Swift app can also link the Rust engine. Keep this independent runtime's
# implementation symbols local while preserving every documented C ABI entry.
set -euo pipefail
input=$1
arch=$2
platform=$3
sdk=$4
header=$5
output=$6
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
grep -Eo 'mold_secure_[a-z_]+' "$header" | sort -u | sed 's/^/_/' > "$scratch/exports"
xcrun ld -r -arch "$arch" -platform_version "$platform" 26.0 "$(xcrun --sdk "$sdk" --show-sdk-version)" \
  -all_load "$input" -o "$scratch/secure.o"
xcrun nmedit -s "$scratch/exports" "$scratch/secure.o"
xcrun nm --no-llvm-bc -gU "$scratch/secure.o" | awk '{ print $NF }' | sort -u > "$scratch/actual"
if ! diff -u "$scratch/exports" "$scratch/actual"; then
  echo 'Secure archive must expose exactly its documented C ABI.' >&2
  exit 1
fi
xcrun libtool -static -D -o "$output" "$scratch/secure.o"
