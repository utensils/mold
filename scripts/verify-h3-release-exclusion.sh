#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <published-binary>" >&2
  exit 64
fi

binary=$1
claim_marker='mold.minimax-h3.attention-rc.kernel-compiled.v1'

[[ -f "$binary" ]] \
  || { echo "published binary is missing: $binary" >&2; exit 1; }

# The isolated qualification executable deliberately retains this marker in
# its JSON report. Published Mold binaries must never contain it: H3 remains
# authorization-gated and no release feature set compiles the candidate.
if grep -aFq "$claim_marker" "$binary"; then
  echo "published binary contains the forbidden MiniMax H3 attention release-candidate claim" >&2
  exit 1
fi

echo "verified MiniMax H3 attention release-candidate exclusion"
