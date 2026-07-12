#!/usr/bin/env bash
# Create the static JSON consumed by Tauri's macOS updater. The updater
# signature must be embedded in the manifest; a URL to the .sig is invalid.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: create-desktop-update-manifest.sh \
  --channel <stable|nightly> \
  --version <semver> \
  --artifact <path-to-app.tar.gz> \
  --signature <path-to-app.tar.gz.sig> \
  --repository <owner/repo> \
  --release-tag <vVERSION|latest> \
  --output <mold-desktop-CHANNEL.json> \
  [--pub-date <RFC3339 UTC>] \
  [--notes <release notes>]
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

channel=""
version=""
artifact=""
signature_path=""
repository=""
release_tag=""
output=""
pub_date=""
notes=""

while (($# > 0)); do
  case "$1" in
    --channel)
      require_value "$1" "${2:-}"
      channel="$2"
      shift 2
      ;;
    --version)
      require_value "$1" "${2:-}"
      version="$2"
      shift 2
      ;;
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
    --repository)
      require_value "$1" "${2:-}"
      repository="$2"
      shift 2
      ;;
    --release-tag)
      require_value "$1" "${2:-}"
      release_tag="$2"
      shift 2
      ;;
    --output)
      require_value "$1" "${2:-}"
      output="$2"
      shift 2
      ;;
    --pub-date)
      require_value "$1" "${2:-}"
      pub_date="$2"
      shift 2
      ;;
    --notes)
      require_value "$1" "${2:-}"
      notes="$2"
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

for required in channel version artifact signature_path repository release_tag output; do
  if [[ -z "${!required}" ]]; then
    echo "error: missing required argument: $required" >&2
    usage
  fi
done

if [[ "$channel" != "stable" && "$channel" != "nightly" ]]; then
  echo "error: channel must be stable or nightly" >&2
  exit 1
fi

# Tauri parses the update version as SemVer. Keep this validation deliberately
# strict enough to catch malformed CI output while accepting normal prerelease
# and build identifiers.
semver_re='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$'
if [[ ! "$version" =~ $semver_re ]]; then
  echo "error: invalid SemVer: $version" >&2
  exit 1
fi

if [[ ! "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "error: invalid GitHub repository: $repository" >&2
  exit 1
fi

case "$channel" in
  stable)
    if [[ "$release_tag" != "v$version" ]]; then
      echo "error: stable release tag must be v$version" >&2
      exit 1
    fi
    ;;
  nightly)
    if [[ "$release_tag" != "latest" ]]; then
      echo "error: nightly release tag must be latest" >&2
      exit 1
    fi
    ;;
esac

expected_output="mold-desktop-$channel.json"
if [[ "$(basename "$output")" != "$expected_output" ]]; then
  echo "error: $channel manifest must be named $expected_output" >&2
  exit 1
fi

if [[ ! -f "$artifact" ]]; then
  echo "error: updater artifact not found: $artifact" >&2
  exit 1
fi
if [[ ! -f "$signature_path" ]]; then
  echo "error: updater signature not found: $signature_path" >&2
  exit 1
fi

artifact_name="$(basename "$artifact")"
expected_artifact_name="Mold_${version}_aarch64.app.tar.gz"
if [[ "$artifact_name" != "$expected_artifact_name" ]]; then
  echo "error: updater artifact must be named $expected_artifact_name" >&2
  exit 1
fi

signature="$(tr -d '\r\n' < "$signature_path")"
if [[ -z "$signature" ]]; then
  echo "error: updater signature is empty: $signature_path" >&2
  exit 1
fi

if [[ -z "$pub_date" ]]; then
  pub_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
if [[ ! "$pub_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
  echo "error: pub-date must be an RFC3339 UTC timestamp" >&2
  exit 1
fi

if [[ -z "$notes" ]]; then
  if [[ "$channel" == "stable" ]]; then
    notes="Mold desktop $version."
  else
    notes="Mold desktop nightly $version from main."
  fi
fi

if ! command -v jq > /dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

output_dir="$(dirname "$output")"
if [[ ! -d "$output_dir" ]]; then
  echo "error: output directory does not exist: $output_dir" >&2
  exit 1
fi

download_url="https://github.com/$repository/releases/download/$release_tag/$artifact_name"
tmp_output="$(mktemp "$output_dir/.${expected_output}.tmp.XXXXXX")"
trap 'rm -f "${tmp_output:-}"' EXIT

jq -n \
  --arg version "$version" \
  --arg notes "$notes" \
  --arg pub_date "$pub_date" \
  --arg url "$download_url" \
  --arg signature "$signature" \
  '{
    version: $version,
    notes: $notes,
    pub_date: $pub_date,
    platforms: {
      "darwin-aarch64": {
        url: $url,
        signature: $signature
      }
    }
  }' > "$tmp_output"

mv "$tmp_output" "$output"
trap - EXIT
