#!/usr/bin/env bash
set -euo pipefail

# Provision the ComfyUI CUDA oracle for the LTX-2.5 qualification (#1398,
# #1414). UAT tooling only; nothing here ships.
#
# Idempotent by adoption: an existing pinned clone or venv is verified and
# kept, never rebuilt — the working venv the campaign already built must
# survive re-runs. Everything lands under LTX25_REFERENCES_ROOT (default
# `<repo>/tmp`, gitignored) using the `*-upstream` directory names
# capture-ltx25-comfy-cuda-reference.sh and the verification harness expect.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
references_root="${LTX25_REFERENCES_ROOT:-$repo_root/tmp}"
venv="${LTX25_COMFY_VENV:-$references_root/comfyui-venv}"
extra_paths="${LTX25_COMFY_EXTRA_PATHS:-$references_root/comfyui-extra-model-paths.yaml}"
models_dir="${MOLD_MODELS_DIR:-}"
python_version="${LTX25_COMFY_PYTHON_VERSION:-3.13}"
mode=provision

fail() {
  echo "LTX-2.5 ComfyUI oracle provisioning failed: $*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
usage: provision-ltx25-comfy-oracle.sh [--verify]

Provisions (or, with --verify, only checks) the pinned ComfyUI CUDA oracle:
the five reference clones under LTX25_REFERENCES_ROOT, the uv-built torch
cu130 venv, and the extra_model_paths.yaml derived from MOLD_MODELS_DIR.
Existing clones and venvs at the right pins are adopted, never rebuilt.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify) mode=verify ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail "unknown argument: $1"
      ;;
  esac
  shift
done

# One row per clone: directory (relative to references_root), URL, pinned
# commit. ComfyUI-GGUF lives inside the ComfyUI checkout's custom_nodes so
# the server loads it without configuration.
clone_rows() {
  cat <<'ROWS'
comfyui-upstream https://github.com/comfyanonymous/ComfyUI.git a1079ba16f2674734b065eb036fbfdddaa321a4d
comfyui-ltxvideo-upstream https://github.com/Lightricks/ComfyUI-LTXVideo.git 15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d
ltx-2-upstream https://github.com/Lightricks/LTX-2.git 400fd31054597515f47125691032c04b1c3ee24e
diffusers-upstream https://github.com/huggingface/diffusers.git 95c0d467cc2a4770b71fa25a117320377e6eb08f
comfyui-upstream/custom_nodes/ComfyUI-GGUF https://github.com/city96/ComfyUI-GGUF.git 6ea2651e7df66d7585f6ffee804b20e92fb38b8a
ROWS
}

command -v git >/dev/null 2>&1 || fail "missing command: git"
[[ -n "$models_dir" ]] || fail "MOLD_MODELS_DIR is required (extra_model_paths.yaml points ComfyUI at the mold store)"

verify_clone() {
  local dir="$1" pin="$2" path actual
  path="$references_root/$dir"
  [[ -d "$path/.git" ]] || {
    echo "missing pinned reference clone: $path"
    return 1
  }
  actual="$(git -C "$path" rev-parse HEAD)"
  [[ "$actual" == "$pin" ]] || {
    echo "$dir is at $actual, expected $pin"
    return 1
  }
  [[ -z "$(git -C "$path" status --porcelain)" ]] || {
    echo "$dir has uncommitted or untracked changes"
    return 1
  }
}

ensure_clone() {
  local dir="$1" url="$2" pin="$3" path
  path="$references_root/$dir"
  if [[ -d "$path/.git" ]]; then
    if [[ "$(git -C "$path" rev-parse HEAD)" == "$pin" ]]; then
      echo "adopted $dir @ $pin"
      return 0
    fi
    echo "repinning $dir to $pin"
    git -C "$path" fetch origin "$pin"
    git -C "$path" checkout --detach "$pin"
    return 0
  fi
  echo "cloning $dir @ $pin"
  mkdir -p "$(dirname "$path")"
  git clone "$url" "$path"
  git -C "$path" checkout --detach "$pin"
}

# The torch probe needs the NVIDIA driver stub and libstdc++ on NixOS; both
# paths are added only when they exist so the probe still runs elsewhere.
torch_probe_library_path() {
  local parts=()
  [[ -d /run/opengl-driver/lib ]] && parts+=(/run/opengl-driver/lib)
  local gcc_lib
  gcc_lib="$(ls -d /nix/store/*-gcc-*-lib/lib 2>/dev/null | head -1 || true)"
  [[ -n "$gcc_lib" && -d "$gcc_lib" ]] && parts+=("$gcc_lib")
  local joined
  joined="$(IFS=:; echo "${parts[*]}")"
  echo "${joined}${joined:+:}${LD_LIBRARY_PATH:-}"
}

verify_torch_cuda() {
  local python="$venv/bin/python"
  [[ -x "$python" ]] || {
    echo "missing venv python: $python"
    return 1
  }
  if ! env LD_LIBRARY_PATH="$(torch_probe_library_path)" "$python" - <<'PY'
import json
import sys
import torch

ok = torch.cuda.is_available()
print(json.dumps({"torch": torch.__version__,
                  "cuda_available": ok,
                  "device": torch.cuda.get_device_name(0) if ok else None}))
sys.exit(0 if ok else 3)
PY
  then
    echo "the venv torch cannot initialise CUDA (probe under LD_LIBRARY_PATH=$(torch_probe_library_path))"
    return 1
  fi
}

verify_all() {
  local status=0 dir url pin
  while read -r dir url pin; do
    [[ -n "$dir" ]] || continue
    verify_clone "$dir" "$pin" || status=1
  done < <(clone_rows)
  [[ -f "$references_root/comfyui-upstream/main.py" ]] || {
    echo "ComfyUI checkout has no main.py"
    status=1
  }
  [[ -f "$references_root/comfyui-upstream/custom_nodes/ComfyUI-GGUF/nodes.py" ]] || {
    echo "ComfyUI-GGUF custom node is not installed inside the ComfyUI checkout"
    status=1
  }
  if [[ -s "$extra_paths" ]]; then
    grep -Fq "base_path: $models_dir" "$extra_paths" || {
      echo "$extra_paths does not point at MOLD_MODELS_DIR=$models_dir"
      status=1
    }
  else
    echo "missing model-path configuration: $extra_paths"
    status=1
  fi
  verify_torch_cuda || status=1
  return "$status"
}

if [[ "$mode" == verify ]]; then
  verify_all || fail "the oracle root is not fully provisioned"
  echo "ComfyUI CUDA oracle verified: $references_root"
  exit 0
fi

mkdir -p "$references_root"
while read -r dir url pin; do
  [[ -n "$dir" ]] || continue
  ensure_clone "$dir" "$url" "$pin"
done < <(clone_rows)

if [[ -x "$venv/bin/python" ]]; then
  echo "adopted existing venv: $venv"
else
  command -v uv >/dev/null 2>&1 || fail "missing command: uv (needed to build $venv)"
  echo "building venv: $venv (python $python_version, torch cu130)"
  uv venv "$venv" --python "$python_version"
  # cu130 is required: comfy-kitchen disables its CUDA backend below cu13,
  # and driver 580.142 is CUDA-13-capable.
  uv pip install --python "$venv/bin/python" torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130
  uv pip install --python "$venv/bin/python" -r "$references_root/comfyui-upstream/requirements.txt"
  if [[ -f "$references_root/comfyui-upstream/custom_nodes/ComfyUI-GGUF/requirements.txt" ]]; then
    uv pip install --python "$venv/bin/python" \
      -r "$references_root/comfyui-upstream/custom_nodes/ComfyUI-GGUF/requirements.txt"
  fi
  uv pip install --python "$venv/bin/python" 'gguf>=0.13.0'
fi

# Derived from MOLD_MODELS_DIR, so rewriting it on every run is the
# idempotent behaviour: the file has no hand-edited state.
tmp_yaml="$extra_paths.tmp.$$"
cat >"$tmp_yaml" <<YAML
# Generated by scripts/provision-ltx25-comfy-oracle.sh — do not hand-edit.
# Points the pinned ComfyUI oracle at the mold model store (read-only use).
mold:
  base_path: $models_dir
  diffusion_models: |
    ltx-2.5-22b-distilled-int8-conv/diffusion_models
    ltx-2.5-22b-dev-int8-conv/diffusion_models
    ltx-2.5-22b-distilled-q4
  text_encoders: shared/ltx2/text_encoders
  vae: shared/ltx2/vae
  latent_upscale_models: shared/ltx2/latent_upscale_models
YAML
mv "$tmp_yaml" "$extra_paths"
echo "wrote $extra_paths"

verify_all || fail "provisioning completed but verification failed"
echo "ComfyUI CUDA oracle ready: $references_root"
