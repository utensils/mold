#!/usr/bin/env bash
# Keep the rolling `latest` release bounded, touching only THIS app's nightly
# DMGs.
#
# Modelled on `scripts/prune-desktop-nightly-assets.sh`, and it does the
# selection in `jq` for the same reason that one does: a `grep` pipeline that
# matches nothing exits 1, and under `set -euo pipefail` that fails the job
# AFTER a successful publish -- a red run for a release that went out fine.
# `jq` on an empty set prints nothing and exits 0.
#
# The pattern is anchored to the native nightly's own shape. `latest` is
# shared with the Tauri app (`Mold_<ver>_aarch64.dmg`) and with CLI archives;
# a loose `Mold-.*\.dmg` would happily delete a future sibling's asset.
set -euo pipefail

repository=""
release_tag="latest"
current=""
keep="10"

while (($# > 0)); do
  case "$1" in
    --repository) repository="${2:?--repository needs a value}"; shift 2 ;;
    --release-tag) release_tag="${2:?--release-tag needs a value}"; shift 2 ;;
    --current-asset) current="${2:?--current-asset needs a value}"; shift 2 ;;
    --keep) keep="${2:?--keep needs a value}"; shift 2 ;;
    --list-only) list_only=1; shift ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done
list_only="${list_only:-0}"

if [[ -z "$repository" || -z "$current" ]]; then
  echo "usage: prune-native-nightly-assets.sh --repository <owner/repo> --current-asset <name> [--release-tag <tag>] [--keep <n>]" >&2
  exit 2
fi

# `--list-only` exists so the selection can be tested without GitHub: it reads
# a release JSON on stdin instead of asking for one.
if [[ "$list_only" == "1" ]]; then
  assets_json=$(cat)
else
  assets_json=$(gh release view "$release_tag" --repo "$repository" --json assets)
fi

stale=$(printf '%s' "$assets_json" | jq -r --arg current "$current" --argjson keep "$keep" '
  [ .assets[].name
    | select(test("^Mold-native-.*-nightly\\.[0-9]+\\.dmg$"))
    | select(. != $current) ]
  | sort
  | reverse
  # `keep` counts the generations that survive INCLUDING the one just
  # published, so this list keeps one fewer of the older ones.
  | .[([$keep - 1, 0] | max):]
  | .[]
')

if [[ -z "$stale" ]]; then
  # To stderr: `--list-only` is read by a test, and a friendly sentence on
  # stdout would be indistinguishable from an asset name.
  echo "  nothing to prune on $release_tag" >&2
  exit 0
fi

while IFS= read -r asset; do
  [[ -n "$asset" ]] || continue
  if [[ "$list_only" == "1" ]]; then
    printf '%s\n' "$asset"
  else
    echo "  pruning $asset"
    gh release delete-asset "$release_tag" "$asset" --repo "$repository" --yes || true
  fi
done <<< "$stale"
