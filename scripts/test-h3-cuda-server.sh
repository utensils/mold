#!/usr/bin/env bash
# Run inside the CUDA devshell (or CI's CUDA toolkit environment).
set -euo pipefail
test_config="$(mktemp -d)"
trap 'rm -rf "$test_config"' EXIT
unset_args=()
# A developer's model store, profiles, backend overrides and credentials must
# never shape a library test. Preserve CUDA/build tooling variables.
while IFS= read -r name; do
  [[ "$name" == MOLD_* ]] && unset_args+=(-u "$name")
done < <(compgen -e)
timeout --signal=TERM --kill-after=30s 30m \
  env "${unset_args[@]}" XDG_CONFIG_HOME="$test_config" \
  cargo test --locked -p mold-ai-server --lib --features h3-cuda -- "$@"
