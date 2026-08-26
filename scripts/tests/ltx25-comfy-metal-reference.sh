#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$repo_root/scripts/capture-ltx25-comfy-metal-reference.sh"
fixture="$repo_root/scripts/fixtures/ltx25-comfy-metal-api-prompt.json"

bash -n "$runner"
jq -e '
  length == 29
  and .["1"].inputs.unet_name == "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
  and .["2"].inputs.vae_name == "ltx-2.5-video-vae-conv-bf16.safetensors"
  and .["4"].inputs.clip_name == "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
  and .["8"].inputs == {width:128,height:128,length:9,batch_size:1}
  and .["11"].inputs.noise_seed == 25026
  and .["20"].inputs.noise_seed == 42
  and .["22"].inputs.sigmas == "0.85, 0.7250, 0.4219, 0.0"
  and .["28"].inputs.fps == 24
  and .["29"].inputs.filename_prefix == "ltx25-comfy-int8-mps-seed-25026"
' "$fixture" >/dev/null

LTX25_COMFY_VALIDATE_ONLY=1 "$runner" >/dev/null

guard_cause() {
  LTX25_COMFY_TEST_GUARD=1 LTX25_TEST_PRESSURE="$1" LTX25_TEST_RSS_KIB="$2" \
    LTX25_TEST_ELAPSED="$3" "$runner"
}

[[ "$(guard_cause unreadable 0 0)" == pressure_unreadable ]]
[[ "$(guard_cause 19 0 0)" == memory_pressure ]]
[[ "$(guard_cause 50 37748737 0)" == server_rss ]]
[[ "$(guard_cause 50 1024 3601)" == timeout ]]
[[ -z "$(guard_cause 50 1024 3600)" ]]
LTX25_COMFY_TEST_STABLE_SEAL=1 "$runner"
echo "LTX-2.5 ComfyUI Metal reference contract OK"
