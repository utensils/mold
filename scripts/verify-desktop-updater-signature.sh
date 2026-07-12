#!/usr/bin/env bash
# Verify a Tauri updater artifact with the exact public key shipped in the app.
# Tauri stores both the public-key file and Minisign signature file as base64.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: verify-desktop-updater-signature.sh \
  --artifact <path-to-app.tar.gz> \
  --signature <path-to-app.tar.gz.sig> \
  --config <path-to-tauri.conf.json>
EOF
  exit 2
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "error: $option requires a value" >&2
    usage
  fi
}

artifact=""
signature_path=""
config=""

while (($# > 0)); do
  case "$1" in
    --artifact)
      require_value "$1" "${2:-}"
      artifact="$2"
      shift 2
      ;;
    --signature)
      require_value "$1" "${2:-}"
      signature_path="$2"
      shift 2
      ;;
    --config)
      require_value "$1" "${2:-}"
      config="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      ;;
  esac
done

for required in artifact signature_path config; do
  if [[ -z "${!required}" ]]; then
    echo "error: missing required argument: $required" >&2
    usage
  fi
done

for path in "$artifact" "$signature_path" "$config"; do
  if [[ ! -f "$path" ]]; then
    echo "error: required file not found: $path" >&2
    exit 1
  fi
done

for command in jq minisign; do
  if ! command -v "$command" > /dev/null 2>&1; then
    echo "error: $command is required" >&2
    exit 1
  fi
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/mold-updater-verify.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

decode_base64() {
  local input="$1"
  local output="$2"
  if base64 --decode < "$input" > "$output" 2> /dev/null; then
    return 0
  fi
  if base64 -D < "$input" > "$output" 2> /dev/null; then
    return 0
  fi
  echo "error: invalid base64 in $input" >&2
  return 1
}

encoded_public_key="$tmp/updater.pub.b64"
public_key="$tmp/updater.pub"
minisign_signature="$tmp/updater.minisig"

if ! jq -er \
  '.plugins.updater.pubkey | select(type == "string" and length > 0)' \
  "$config" > "$encoded_public_key"; then
  echo "error: updater public key is missing from $config" >&2
  exit 1
fi

decode_base64 "$encoded_public_key" "$public_key"
decode_base64 "$signature_path" "$minisign_signature"

if ! minisign -V \
  -p "$public_key" \
  -m "$artifact" \
  -x "$minisign_signature"; then
  echo "error: updater signature does not match the public key in $config" >&2
  exit 1
fi

echo "Verified updater signature with the public key embedded in $config."
