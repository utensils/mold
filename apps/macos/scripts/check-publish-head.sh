#!/usr/bin/env bash
# Is the commit this build came from still the tip of the ref that asked for it?
#
# The desktop nightly learned this the hard way: `main` advances while a
# 40-minute signed build runs, and the older run then flips the channel
# pointer at an app nobody asked for any more
# (`.github/workflows/desktop.yml:634-685`, `check-desktop-nightly-main-head.sh`).
# The answer there is to ask TWICE -- once before uploading the immutable
# assets and again immediately before the pointer -- and to SKIP rather than
# fail when the answer is no.
#
# Exit codes are that contract: 0 still ours, 10 stale (skip, not a failure),
# anything else a real error.
#
# A TAG cannot move, so a tag build is always current. Saying that here rather
# than in the workflow keeps the two triggers on one rule.
set -euo pipefail

EXPECTED="${1:?usage: check-publish-head.sh <sha>}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
REF_TYPE="${GITHUB_REF_TYPE:-branch}"
REF_NAME="${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"

if [ "$REF_TYPE" = "tag" ]; then
  echo "  $REF_NAME is a tag; $EXPECTED cannot go stale"
  exit 0
fi

head="$(gh api "repos/$GITHUB_REPOSITORY/commits/$REF_NAME" --jq .sha)"
if [ -z "$head" ]; then
  echo "could not read the head of $REF_NAME" >&2
  exit 1
fi
if [ "$head" != "$EXPECTED" ]; then
  echo "  $REF_NAME has moved to ${head:0:12}; this build is $EXPECTED"
  exit 10
fi
echo "  $REF_NAME is still ${EXPECTED:0:12}"
