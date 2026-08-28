#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$repo_root/scripts/capture-ltx25-comfy-cuda-reference.sh"
int8_graph="$repo_root/scripts/fixtures/ltx25-comfy-cuda-int8-api-prompt.json"
gguf_graph="$repo_root/scripts/fixtures/ltx25-comfy-cuda-gguf-q4-api-prompt.json"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() {
  echo "LTX-2.5 ComfyUI CUDA reference contract failed: $*" >&2
  exit 1
}

bash -n "$runner"

# Both graphs are the Metal 29-node LTX-2.5 workflow with a CUDA filename
# prefix; the GGUF graph swaps node 1 for city96's loader over the Q4_K_M file.
jq -e '
  length == 29
  and .["1"].class_type == "UNETLoader"
  and .["1"].inputs.unet_name == "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
  and .["29"].inputs.filename_prefix == "ltx25-comfy-int8-cuda-seed-25026"
' "$int8_graph" >/dev/null || fail "INT8 graph fixture is not the pinned CUDA workflow"
jq -e '
  length == 29
  and .["1"].class_type == "UnetLoaderGGUF"
  and .["1"].inputs == {unet_name: "LTX-2.5-Distilled-Q4_K_M.gguf"}
  and .["29"].inputs.filename_prefix == "ltx25-comfy-gguf-q4-cuda-seed-25026"
' "$gguf_graph" >/dev/null || fail "GGUF graph fixture is not the pinned CUDA workflow"
# Everything except the loader and the output name is byte-identical between
# the two graphs, so the GGUF oracle differs from the INT8 one by exactly the
# transformer.
diff <(jq -S 'del(.["1"]) | del(.["29"].inputs.filename_prefix)' "$int8_graph") \
  <(jq -S 'del(.["1"]) | del(.["29"].inputs.filename_prefix)' "$gguf_graph") >/dev/null \
  || fail "INT8 and GGUF graphs diverge beyond the loader and filename prefix"
# The CUDA graph is the Metal graph with only its filename prefix changed.
metal_graph="$repo_root/scripts/fixtures/ltx25-comfy-metal-api-prompt.json"
diff <(jq -S 'del(.["29"].inputs.filename_prefix)' "$metal_graph") \
  <(jq -S 'del(.["29"].inputs.filename_prefix)' "$int8_graph") >/dev/null \
  || fail "INT8 CUDA graph diverges from the Metal graph beyond the filename prefix"

[[ "$(LTX25_COMFY_VALIDATE_ONLY=1 "$runner" --graph int8)" == "$int8_graph" ]] \
  || fail "validate-only did not select the INT8 graph"
[[ "$(LTX25_COMFY_VALIDATE_ONLY=1 "$runner" --graph gguf-q4)" == "$gguf_graph" ]] \
  || fail "validate-only did not select the GGUF graph"
if LTX25_COMFY_VALIDATE_ONLY=1 "$runner" --graph nvfp4 >/dev/null 2>&1; then
  fail "runner accepted an unknown graph selector"
fi

tampered="$tmp/tampered.json"
jq '.["11"].inputs.noise_seed = 1' "$int8_graph" >"$tampered"
if LTX25_COMFY_VALIDATE_ONLY=1 LTX25_COMFY_GRAPH="$tampered" "$runner" --graph int8 >/dev/null 2>&1; then
  fail "runner accepted a graph whose stage-1 seed was changed"
fi
jq '.["1"].inputs.unet_name = "LTX-2.5-Distilled-Q8_0.gguf"' "$gguf_graph" >"$tampered"
if LTX25_COMFY_VALIDATE_ONLY=1 LTX25_COMFY_GRAPH="$tampered" "$runner" --graph gguf-q4 >/dev/null 2>&1; then
  fail "runner accepted a GGUF graph over a different quantization tier"
fi

guard() {
  LTX25_COMFY_TEST_GUARD=1 \
    LTX25_TEST_AVAIL_PERCENT="$1" LTX25_TEST_RSS_KIB="$2" \
    LTX25_TEST_ELAPSED="$3" LTX25_TEST_GPU_USED_MIB="$4" "$runner" --graph int8
}
[[ "$(guard "" 0 0 100)" == pressure_unreadable ]] || fail "unreadable host memory must abort"
[[ "$(guard 50 0 0 "")" == gpu_unreadable ]] || fail "unreadable GPU memory must abort"
[[ "$(guard 19 0 0 100)" == host_memory ]] || fail "host memory below 20% must abort"
[[ "$(guard 50 50331649 0 100)" == server_rss ]] || fail "server RSS above 48 GiB must abort"
[[ "$(guard 50 0 3601 100)" == timeout ]] || fail "one hour must abort"
[[ -z "$(guard 50 50331648 3600 100)" ]] || fail "healthy readings must not abort"

grep -Fq 'mold.ltx25.comfy-cuda-reference.v1' "$runner" || fail "runner lost the CUDA manifest schema"
grep -Fq 'backend:"CUDA"' "$runner" || fail "runner lost the CUDA backend attestation"
grep -Fq 'torch_cuda_unavailable' "$runner" || fail "runner cannot defer on a torch build without CUDA"
grep -Fq 'LD_LIBRARY_PATH=/run/opengl-driver/lib' "$runner" \
  || fail "runner does not expose the NixOS driver libraries to torch"
if grep -Fq 'memory_pressure' "$runner"; then
  fail "runner still calls the macOS memory_pressure tool"
fi

LTX25_COMFY_TEST_STABLE_SEAL=1 "$runner" --graph int8

echo "LTX-2.5 ComfyUI CUDA reference contract OK"
