#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <published-binary>" >&2
  exit 64
fi

binary=$1
claim_marker='mold.minimax-h3.attention-rc.kernel-compiled.v1'
omitted_marker='mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=omitted'
compiled_markers=(
  'mold.minimax-h3.attention-release-provenance.v2:h3-rc=compiled:global-flash=omitted'
  'mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=compiled'
  'mold.minimax-h3.attention-release-provenance.v2:h3-rc=compiled:global-flash=compiled'
)

[[ -f "$binary" ]] \
  || { echo "published binary is missing: $binary" >&2; exit 1; }

# This positive marker is selected by the same compile-time feature gates as
# the candidate code and deliberately retained by every shipping entry point.
# Requiring it closes the dead-strip loophole in a forbidden-marker-only scan:
# missing or unknown provenance fails rather than being mistaken for omission.
if ! grep -aFq "$omitted_marker" "$binary"; then
  echo "published binary lacks omitted/omitted MiniMax H3 attention provenance" >&2
  exit 1
fi

for compiled_marker in "${compiled_markers[@]}"; do
  if grep -aFq "$compiled_marker" "$binary"; then
    echo "published binary reports compiled MiniMax H3/global FlashAttention code" >&2
    exit 1
  fi
done

# The isolated qualification executable also retains this claim in its JSON
# report. Keep it as a second, independent rejection signal.
if grep -aFq "$claim_marker" "$binary"; then
  echo "published binary contains the forbidden MiniMax H3 attention release-candidate claim" >&2
  exit 1
fi

echo "verified MiniMax H3 attention release-candidate exclusion"
