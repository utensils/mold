#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

cargo test -p mold-ai-inference --test family_capability_contract
cargo test -p mold-ai-scheduler --test multi_gpu_acceptance
cargo test -p mold-ai-server execution_plan::tests::
bash scripts/tests/local-multi-gpu-qualification-contract.sh

matrix="docs/qualification/multi-gpu-family-matrix.md"
for family in \
  flux flux2 sd15 sdxl sd3 z-image qwen-image qwen-image-edit \
  ltx-video ltx2 wuerstchen; do
  grep -Fq "\`$family\`" "$matrix" || {
    echo "qualification matrix is missing family $family" >&2
    exit 1
  }
done

grep -Fq 'Deferred; not hardware-qualified' "$matrix" || {
  echo "qualification matrix must retain the deferred hardware disclaimer" >&2
  exit 1
}

echo "multi-GPU deterministic family and arbitrary-N acceptance passed"
