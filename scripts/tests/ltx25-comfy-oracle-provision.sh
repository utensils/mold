#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$repo_root/scripts/provision-ltx25-comfy-oracle.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() {
  echo "LTX-2.5 ComfyUI oracle provisioning contract failed: $*" >&2
  exit 1
}

bash -n "$runner"
"$runner" --help | grep -Fq -- '--verify' || fail "help does not describe --verify"

# The five pins the capture scripts expect, spelled once in the runner.
for pin in \
  a1079ba16f2674734b065eb036fbfdddaa321a4d \
  15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d \
  400fd31054597515f47125691032c04b1c3ee24e \
  95c0d467cc2a4770b71fa25a117320377e6eb08f \
  6ea2651e7df66d7585f6ffee804b20e92fb38b8a; do
  grep -Fq "$pin" "$runner" || fail "runner lost the $pin pin"
done
grep -Fq '/run/opengl-driver/lib' "$runner" \
  || fail "runner does not expose the NixOS driver libraries to the torch probe"
grep -Fq -- '-gcc-*-lib/lib' "$runner" \
  || fail "runner does not probe the gcc runtime library path torch needs on NixOS"
grep -Fq 'download.pytorch.org/whl/cu130' "$runner" \
  || fail "runner lost the cu130 torch index (comfy-kitchen disables CUDA below cu13)"

# --- fake, fully provisioned oracle root -------------------------------------
refs="$tmp/refs"
models="$tmp/models"
mkdir -p "$models" "$tmp/bin"
for name in comfyui-upstream comfyui-ltxvideo-upstream ltx-2-upstream diffusers-upstream; do
  mkdir -p "$refs/$name/.git"
done
mkdir -p "$refs/comfyui-upstream/custom_nodes/ComfyUI-GGUF/.git"
touch "$refs/comfyui-upstream/main.py" "$refs/comfyui-upstream/requirements.txt" \
  "$refs/comfyui-upstream/custom_nodes/ComfyUI-GGUF/nodes.py" \
  "$refs/comfyui-upstream/custom_nodes/ComfyUI-GGUF/requirements.txt"
mkdir -p "$refs/comfyui-venv/bin"
cat >"$refs/comfyui-venv/bin/python" <<'FAKE'
#!/usr/bin/env bash
# The provision --verify torch probe feeds a script on stdin; a real venv
# imports torch there. The fixture swallows it and reports a healthy device.
cat >/dev/null
echo '{"torch": "fixture", "cuda_available": true, "device": "fixture RTX 4090"}'
FAKE
chmod +x "$refs/comfyui-venv/bin/python"

real_git="$(command -v git)"
cat >"$tmp/bin/git" <<FAKE
#!/usr/bin/env bash
set -euo pipefail
if [[ "\$1" == -C && "\$2" == "$refs/"* ]]; then
  if [[ "\$3" == status && "\$4" == --porcelain ]]; then
    exit 0
  fi
  [[ "\$3" == rev-parse && "\$4" == HEAD ]] || exec "$real_git" "\$@"
  case "\$2" in
    */custom_nodes/ComfyUI-GGUF) echo 6ea2651e7df66d7585f6ffee804b20e92fb38b8a ;;
    */comfyui-upstream) echo a1079ba16f2674734b065eb036fbfdddaa321a4d ;;
    */comfyui-ltxvideo-upstream) echo 15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d ;;
    */ltx-2-upstream) echo 400fd31054597515f47125691032c04b1c3ee24e ;;
    */diffusers-upstream) echo 95c0d467cc2a4770b71fa25a117320377e6eb08f ;;
  esac
else
  exec "$real_git" "\$@"
fi
FAKE
# An adopted root must never rebuild: any uv invocation is a contract failure.
cat >"$tmp/bin/uv" <<'FAKE'
#!/usr/bin/env bash
echo "uv must not run against an adopted venv" >&2
exit 97
FAKE
chmod +x "$tmp/bin/git" "$tmp/bin/uv"

provision() {
  PATH="$tmp/bin:$PATH" LTX25_REFERENCES_ROOT="$refs" MOLD_MODELS_DIR="$models" "$runner" "$@"
}

# Provisioning an already complete root adopts it (uv would exit 97) and
# writes the model-paths YAML from MOLD_MODELS_DIR.
provision >/dev/null || fail "provisioning did not adopt a complete oracle root"
yaml="$refs/comfyui-extra-model-paths.yaml"
[[ -s "$yaml" ]] || fail "provisioning did not write $yaml"
grep -Fq "base_path: $models" "$yaml" || fail "YAML does not point at MOLD_MODELS_DIR"
grep -Fq 'ltx-2.5-22b-distilled-int8-conv/diffusion_models' "$yaml" \
  || fail "YAML does not expose the INT8 transformer directory"
grep -Fq 'ltx-2.5-22b-distilled-q4' "$yaml" || fail "YAML does not expose the GGUF directory"
grep -Fq 'shared/ltx2/text_encoders' "$yaml" || fail "YAML does not expose the shared encoders"
provision >/dev/null || fail "provisioning is not idempotent"

provision --verify >/dev/null || fail "--verify refused a healthy oracle root"

# --- negatives ---------------------------------------------------------------
mv "$refs/comfyui-venv/bin/python" "$tmp/python.bak"
if provision --verify >/dev/null 2>&1; then
  fail "--verify accepted a missing venv python"
fi
mv "$tmp/python.bak" "$refs/comfyui-venv/bin/python"

cat >"$tmp/bin/python-cuda-less" <<'FAKE'
#!/usr/bin/env bash
cat >/dev/null
echo "torch cannot initialise CUDA" >&2
exit 3
FAKE
chmod +x "$tmp/bin/python-cuda-less"
cp "$tmp/bin/python-cuda-less" "$refs/comfyui-venv/bin/python"
if provision --verify >/dev/null 2>&1; then
  fail "--verify accepted a torch build that cannot initialise CUDA"
fi
cat >"$refs/comfyui-venv/bin/python" <<'FAKE'
#!/usr/bin/env bash
cat >/dev/null
echo '{"torch": "fixture", "cuda_available": true, "device": "fixture RTX 4090"}'
FAKE
chmod +x "$refs/comfyui-venv/bin/python"

sed -i 's/a1079ba16f2674734b065eb036fbfdddaa321a4d/1111111111111111111111111111111111111111/' "$tmp/bin/git"
if provision --verify >/dev/null 2>&1; then
  fail "--verify accepted a ComfyUI clone at the wrong commit"
fi
sed -i 's/1111111111111111111111111111111111111111/a1079ba16f2674734b065eb036fbfdddaa321a4d/' "$tmp/bin/git"

mv "$refs/comfyui-upstream/custom_nodes/ComfyUI-GGUF" "$tmp/gguf.bak"
if provision --verify >/dev/null 2>&1; then
  fail "--verify accepted a missing ComfyUI-GGUF custom node"
fi
mv "$tmp/gguf.bak" "$refs/comfyui-upstream/custom_nodes/ComfyUI-GGUF"

rm "$yaml"
if provision --verify >/dev/null 2>&1; then
  fail "--verify accepted a missing extra_model_paths.yaml"
fi
provision >/dev/null
provision --verify >/dev/null || fail "re-provisioning did not restore a verifiable root"

if PATH="$tmp/bin:$PATH" LTX25_REFERENCES_ROOT="$refs" MOLD_MODELS_DIR="" "$runner" >/dev/null 2>&1; then
  fail "provisioning ran without MOLD_MODELS_DIR"
fi

echo "LTX-2.5 ComfyUI oracle provisioning contract OK"
