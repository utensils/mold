#!/usr/bin/env bash
# Shared release-plz version; never keep a second native version literal.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
version=$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version = "\([^"]*\)"/\1/p' "$root/Cargo.toml" | head -1)
if [[ ! "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
  echo 'error: workspace.package.version must be a plain three-part SemVer' >&2
  exit 1
fi
printf '%s\n' "$version"
