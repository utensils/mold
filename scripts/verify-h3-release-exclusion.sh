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
private_qwen_capture_marker='mold.minimax-h3.private-uat-exact-bf16-qwen-layer50-capture.v1'
private_visual_vae_capture_marker='mold.minimax-h3.private-uat-visual-vae-f32-fp16-capture.v1'
private_audio_capture_marker='mold.minimax-h3.private-uat-exact-fp32-audio-vae-capture.v1'
private_transformer_capture_marker='mold.minimax-h3.private-uat-transformer-capture.v1'
omitted_marker='mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=omitted'
compiled_markers=(
  'mold.minimax-h3.attention-release-provenance.v2:h3-rc=compiled:global-flash=omitted'
  'mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=compiled'
  'mold.minimax-h3.attention-release-provenance.v2:h3-rc=compiled:global-flash=compiled'
)

[[ -f "$binary" ]] \
  || { echo "published binary is missing: $binary" >&2; exit 1; }

# This positive marker is selected by the same compile-time feature gates as
# the candidate code and deliberately retained by every shipping entry point.
# Requiring it closes the dead-strip loophole in a forbidden-marker-only scan:
# missing or unknown provenance fails rather than being mistaken for omission.
if ! grep -aFq "$omitted_marker" "$binary"; then
  echo "published binary lacks omitted/omitted MiniMax H3 attention provenance" >&2
  exit 1
fi

for compiled_marker in "${compiled_markers[@]}"; do
  if grep -aFq "$compiled_marker" "$binary"; then
    echo "published binary reports compiled MiniMax H3/global FlashAttention code" >&2
    exit 1
  fi
done

# The isolated qualification executable also retains this claim in its JSON
# report. Keep it as a second, independent rejection signal.
if grep -aFq "$claim_marker" "$binary"; then
  echo "published binary contains the forbidden MiniMax H3 attention release-candidate claim" >&2
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

echo "verified MiniMax H3 development-only runtime exclusion"
