#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$repo_root/scripts/verify-desktop-updater-signature.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

for command in jq minisign; do
  if ! command -v "$command" > /dev/null 2>&1; then
    echo "error: $command is required for this test" >&2
    exit 1
  fi
done

encode_base64() {
  base64 < "$1" | tr -d '\r\n'
}

payload="$tmp/Mold_0.16.1_aarch64.app.tar.gz"
raw_signature="$tmp/payload.minisig"
tauri_signature="$payload.sig"
public_key="$tmp/updater.pub"
secret_key="$tmp/updater.key"
config="$tmp/tauri.conf.json"

printf 'signed updater payload\n' > "$payload"
minisign -G -W -p "$public_key" -s "$secret_key" > /dev/null
minisign -S -s "$secret_key" -m "$payload" -x "$raw_signature" > /dev/null
encode_base64 "$raw_signature" > "$tauri_signature"

jq -n \
  --arg pubkey "$(encode_base64 "$public_key")" \
  '{plugins: {updater: {pubkey: $pubkey}}}' > "$config"

"$script" \
  --artifact "$payload" \
  --signature "$tauri_signature" \
  --config "$config" > /dev/null \
  || fail "valid Tauri updater signature was rejected"

printf 'tampered updater payload\n' > "$payload"
if "$script" \
  --artifact "$payload" \
  --signature "$tauri_signature" \
  --config "$config" > /dev/null 2>&1; then
  fail "tampered updater payload was accepted"
fi

printf 'signed updater payload\n' > "$payload"
other_public_key="$tmp/other.pub"
other_secret_key="$tmp/other.key"
minisign -G -W -p "$other_public_key" -s "$other_secret_key" > /dev/null
jq -n \
  --arg pubkey "$(encode_base64 "$other_public_key")" \
  '{plugins: {updater: {pubkey: $pubkey}}}' > "$config"
if "$script" \
  --artifact "$payload" \
  --signature "$tauri_signature" \
  --config "$config" > /dev/null 2>&1; then
  fail "signature from a different updater key was accepted"
fi

printf 'not base64\n' > "$tauri_signature"
if "$script" \
  --artifact "$payload" \
  --signature "$tauri_signature" \
  --config "$config" > /dev/null 2>&1; then
  fail "malformed Tauri signature was accepted"
fi

echo "PASS: verify-desktop-updater-signature"
