#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <published-binary>" >&2
  exit 64
fi

binary=$1
claim_marker='mold.minimax-h3.attention-rc.kernel-compiled.v1'
private_uat_marker='mold.minimax-h3.private-uat-artifact-reader.v1'
private_qwen_support_marker='mold.minimax-h3.private-uat-qwen-support-loader.v1'
private_runtime_record_marker='mold.minimax-h3.private-runtime-record-producer.v1'
private_runtime_observation_markers=(
  'mold.minimax-h3.private-uat-runtime-bound-observation.v1'
  'mold.minimax-h3.private-uat-runtime-bound-observation.v2'
  'mold.minimax-h3.private-uat-runtime-bound-observation.v3'
  'mold.minimax-h3.private-uat-runtime-bound-observation.v4'
  'mold.minimax-h3.private-uat-runtime-bound-observation.v5'
)
private_qwen_capture_marker='mold.minimax-h3.private-uat-exact-bf16-qwen-layer50-capture.v1'
private_visual_vae_capture_marker='mold.minimax-h3.private-uat-visual-vae-f32-fp16-capture.v1'
private_audio_capture_marker='mold.minimax-h3.private-uat-exact-fp32-audio-vae-capture.v1'
private_transformer_capture_marker='mold.minimax-h3.private-uat-transformer-capture.v1'
omitted_marker='mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=omitted'
h3_compiled_marker='mold.minimax-h3.attention-release-provenance.v2:h3-rc=compiled:global-flash=omitted'
global_compiled_markers=(
  'mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=compiled'
  'mold.minimax-h3.attention-release-provenance.v2:h3-rc=compiled:global-flash=compiled'
)

[[ -f "$binary" ]] \
  || { echo "published binary is missing: $binary" >&2; exit 1; }

# These mutually exclusive markers are selected by the same compile-time
# feature gates as the H3-scoped and global kernels. Requiring exactly one
# closes the dead-strip loophole: missing or ambiguous provenance fails rather
# than being mistaken for an ordinary or H3-enabled release.
provenance_count=0
grep -aFq "$omitted_marker" "$binary" && provenance_count=$((provenance_count + 1))
grep -aFq "$h3_compiled_marker" "$binary" && provenance_count=$((provenance_count + 1))
if [[ $provenance_count -ne 1 ]]; then
  echo "published binary lacks unambiguous MiniMax H3 attention provenance" >&2
  exit 1
fi

for compiled_marker in "${global_compiled_markers[@]}"; do
  if grep -aFq "$compiled_marker" "$binary"; then
    echo "published binary reports forbidden global FlashAttention code" >&2
    exit 1
  fi
done

# The H3-scoped kernel claim must agree with the positive provenance marker.
# A normal build may omit both; a public H3 build must retain both.
if grep -aFq "$h3_compiled_marker" "$binary"; then
  grep -aFq "$claim_marker" "$binary" || {
    echo "published H3 binary lacks its H3-scoped kernel claim" >&2
    exit 1
  }
elif grep -aFq "$claim_marker" "$binary"; then
  echo "ordinary published binary contains an unexpected H3 kernel claim" >&2
  exit 1
fi

if grep -aFq "$private_uat_marker" "$binary"; then
  echo "published binary contains the forbidden MiniMax H3 private-UAT artifact reader" >&2
  exit 1
fi

if grep -aFq "$private_qwen_support_marker" "$binary"; then
  echo "published binary contains the forbidden MiniMax H3 private Qwen support loader" >&2
  exit 1
fi

if grep -aFq "$private_runtime_record_marker" "$binary"; then
  echo "published binary contains the forbidden MiniMax H3 runtime-record producer" >&2
  exit 1
fi

for private_runtime_observation_marker in "${private_runtime_observation_markers[@]}"; do
  if grep -aFq "$private_runtime_observation_marker" "$binary"; then
    echo "published binary contains the forbidden MiniMax H3 runtime-bound observer" >&2
    exit 1
  fi
done

if grep -aFq "$private_qwen_capture_marker" "$binary"; then
  echo "published binary contains the forbidden MiniMax H3 exact-BF16 Qwen capture adapter" >&2
  exit 1
fi

if grep -aFq "$private_visual_vae_capture_marker" "$binary"; then
  echo "published binary contains the forbidden MiniMax H3 visual-VAE capture adapter" >&2
  exit 1
fi

if grep -aFq "$private_audio_capture_marker" "$binary"; then
  echo "published binary contains the forbidden MiniMax H3 exact-FP32 AudioVAE capture adapter" >&2
  exit 1
fi

if grep -aFq "$private_transformer_capture_marker" "$binary"; then
  echo "published binary contains the forbidden MiniMax H3 transformer capture adapter" >&2
  exit 1
fi

echo "verified MiniMax H3 release provenance and development-tool exclusion"
