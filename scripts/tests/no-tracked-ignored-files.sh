#!/usr/bin/env bash
# release-plz refuses to run while any tracked file is also matched by
# .gitignore ("the working directory of this project has uncommitted
# changes"). #1767 committed a hunyuan3d test fixture under the repo-wide
# `*.png` rule, and every release PR run on main failed after it -- silently,
# from the PR's point of view. This fails the PR instead: add a `!path`
# negation beside the others in .gitignore for any file it names.
set -euo pipefail

cd "$(dirname "$0")/../.."
offenders="$(git ls-files -ci --exclude-standard)"
if [ -n "$offenders" ]; then
  echo "Tracked files that .gitignore also matches (release-plz will refuse to run):"
  echo "$offenders" | sed 's/^/  /'
  echo "Add a '!<path>' negation for each in .gitignore."
  exit 1
fi
echo "  no tracked file is gitignored"
