#!/usr/bin/env bash
# Prove both external analyzers remain active with the native binary.
set -euo pipefail
binary=${1:?usage: actionlint-native.sh ACTIONLINT_BINARY}
shellcheck=$(command -v shellcheck)
pyflakes=$(command -v pyflakes3 || command -v pyflakes)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
"$binary" -version | grep -Fx '1.7.12'
cat > "$scratch/bad.yml" <<'YAML'
name: Analyzer fixture
on: push
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: echo $INPUT_VALUE
      - shell: python
        run: print(undefined_name)
YAML
if "$binary" -shellcheck="$shellcheck" -pyflakes="$pyflakes" "$scratch/bad.yml" > "$scratch/diagnostics" 2>&1; then
  echo 'actionlint accepted intentionally invalid shell and Python' >&2
  exit 1
fi
grep -F '[shellcheck]' "$scratch/diagnostics"
grep -F '[pyflakes]' "$scratch/diagnostics"
echo 'PASS: native actionlint retains ShellCheck and Pyflakes'
