#!/usr/bin/env bash
# Keep the rolling `latest` release bounded without touching the manifest's
# current target or unrelated CLI/release assets.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: prune-desktop-nightly-assets.sh \
  --repository <owner/repo> \
  --release-tag <tag> \
  --current-version <nightly-semver> \
  [--keep <build-count>]
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

repository=""
release_tag=""
current_version=""
keep="10"

while (($# > 0)); do
  case "$1" in
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
    --current-version)
      require_value "$1" "${2:-}"
      current_version="$2"
      shift 2
      ;;
    --keep)
      require_value "$1" "${2:-}"
      keep="$2"
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

for required in repository release_tag current_version; do
  if [[ -z "${!required}" ]]; then
    echo "error: missing required argument: $required" >&2
    usage
  fi
done

if [[ ! "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "error: invalid GitHub repository: $repository" >&2
  exit 1
fi
if [[ ! "$release_tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "error: invalid release tag: $release_tag" >&2
  exit 1
fi
if [[ ! "$current_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+-nightly\.[1-9][0-9]*$ ]]; then
  echo "error: invalid nightly version: $current_version" >&2
  exit 1
fi
if [[ ! "$keep" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: keep must be a positive integer" >&2
  exit 1
fi

for command in gh jq; do
  if ! command -v "$command" > /dev/null 2>&1; then
    echo "error: $command is required" >&2
    exit 1
  fi
done

assets_json="$(gh release view "$release_tag" --repo "$repository" --json assets)"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/mold-nightly-prune.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
deletions="$tmp/deletions.txt"

# Only version-unique updater payloads, signatures, and DMGs match this
# expression. The channel manifest and generic CLI assets can never be selected.
asset_pattern='^Mold_(?<version>[0-9]+\.[0-9]+\.[0-9]+-nightly\.[1-9][0-9]*)_aarch64\.(?:app\.tar\.gz(?:\.sig)?|dmg)$'

jq -r \
  --arg pattern "$asset_pattern" \
  --arg current "$current_version" \
  --argjson keep "$keep" \
  '
    [.assets[]
      | select(.name | test($pattern))
      | . + {desktopVersion: (.name | capture($pattern).version)}
    ]
    | group_by(.desktopVersion)
    | map({
        version: .[0].desktopVersion,
        createdAt: (map(.createdAt) | max),
        assets: .
      })
    | sort_by(.createdAt)
    | reverse
    | . as $groups
    | (($groups | map(.version | select(. != $current)) | .[0:($keep - 1)]) + [$current]) as $retained
    | $groups[]
    | .version as $version
    | select(($retained | index($version)) == null)
    | .assets[].name
  ' <<< "$assets_json" > "$deletions"

deleted=0
while IFS= read -r asset; do
  [[ -n "$asset" ]] || continue
  gh release delete-asset "$release_tag" "$asset" \
    --repo "$repository" \
    --yes
  deleted=$((deleted + 1))
done < "$deletions"

echo "Pruned $deleted old desktop nightly assets; retained $keep build generations."
