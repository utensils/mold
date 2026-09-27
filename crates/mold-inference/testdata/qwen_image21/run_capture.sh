#!/usr/bin/env bash
# Re-run every capture stage on ONE 46 GB GPU, one process per memory phase
# (the fp32 text encoder and the fp32 transformer do not fit together).
# Provenance only; see README.md.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
py="${QWEN21_VENV:-/home/jamesbrink/Projects/mold/tmp/qwen21-venv}/bin/python"
logs="${QWEN21_LOGS:-/storage/mold/fixtures/qwen_image21/logs}"
mkdir -p "$logs"
# Torch must load ONLY its bundled cuDNN: a devshell LD_LIBRARY_PATH mixes a
# different cuDNN release's engine libraries into the process (capture.py
# asserts this). Drop every entry that carries a libcudnn; keep the rest
# (driver, libstdc++, zlib, ...).
clean=""
IFS=: read -r -a ld_dirs <<<"${LD_LIBRARY_PATH:-}"
for d in "${ld_dirs[@]}"; do
  [ -n "$d" ] || continue
  compgen -G "$d/libcudnn*" >/dev/null && continue
  clean="${clean:+$clean:}$d"
done
export LD_LIBRARY_PATH="$clean"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
for stages in ${QWEN21_STAGES:-images,viggle,pillow,cpu encode transformer,lora e2e bf16 alpha manifest}; do
  echo "== $stages"
  "$py" "$here/capture.py" --stages "$stages" >"$logs/log_${stages//,/_}.txt" 2>&1 || {
    echo "stage $stages FAILED, see $logs/log_${stages//,/_}.txt"
    exit 1
  }
done
echo "== done"
