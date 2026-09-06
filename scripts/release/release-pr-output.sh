#!/usr/bin/env bash
# Normalize the pinned release-plz CLI response for the existing sync step.
set -euo pipefail
jq -ce '.prs | if type != "array" then error("missing PR array")
  elif length == 0 then {}
  elif length == 1 and (.[0].number | type) == "number"
    and (.[0].head_branch | type) == "string" and (.[0].head_branch | length) > 0
    then .[0]
  else error("unexpected release PR response") end'
