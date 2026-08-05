#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

# Topological order for every publishable workspace crate. Keep this list
# complete: crates.io resolves path dependencies as registry dependencies, so a
# dependent cannot be packaged until the exact dependency version is indexed.
packages=(
  mold-ai-core
  mold-ai-candle
  mold-ai-db
  mold-ai-catalog
  mold-ai-scheduler
  mold-ai-inference
  mold-ai-discord
  mold-ai-server
  mold-ai-tui
  mold-ai
)

if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${packages[@]}"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--list]" >&2
  exit 2
fi

: "${CARGO_REGISTRY_TOKEN:?CARGO_REGISTRY_TOKEN must be set}"

version=$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1)
if [[ -z "$version" ]]; then
  echo "error: could not read [workspace.package] version" >&2
  exit 1
fi

registry_api="${CARGO_REGISTRY_API:-https://crates.io/api/v1/crates}"
index_attempts="${CARGO_INDEX_ATTEMPTS:-40}"
index_interval="${CARGO_INDEX_INTERVAL_SECONDS:-15}"
user_agent="mold-release/${version} (https://github.com/utensils/mold)"

version_exists() {
  local package="$1"
  local status
  status=$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
    --retry 3 --user-agent "$user_agent" "$registry_api/$package/$version")
  case "$status" in
    200) return 0 ;;
    404) return 1 ;;
    *)
      echo "error: crates.io returned HTTP $status for $package@$version" >&2
      exit 1
      ;;
  esac
}

wait_for_index() {
  local package="$1"
  local attempt
  for ((attempt = 1; attempt <= index_attempts; attempt++)); do
    if cargo info "$package@$version" --registry crates-io >/dev/null 2>&1; then
      echo "Indexed $package@$version."
      return 0
    fi
    if ((attempt < index_attempts)); then
      echo "Waiting for crates.io index: $package@$version ($attempt/$index_attempts)"
      sleep "$index_interval"
    fi
  done
  echo "error: $package@$version did not become resolvable from crates.io" >&2
  return 1
}

for package in "${packages[@]}"; do
  if version_exists "$package"; then
    echo "Already published: $package@$version"
  else
    echo "Publishing $package@$version..."
    cargo publish --locked -p "$package"
  fi
  # This also covers partial-release recovery: an API-visible dependency may
  # still be absent from the sparse index used to package the next crate.
  wait_for_index "$package"
done

echo "Published all Mold workspace crates for $version."
