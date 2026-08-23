#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

h3_feature_test_bin="$(mktemp -t mold-h3-server-features.XXXXXX)"
trap 'rm -f "$h3_feature_test_bin"' EXIT
rustc --edition=2021 --test \
  crates/mold-server/build_support/h3_server_features.rs \
  -o "$h3_feature_test_bin"
"$h3_feature_test_bin"

require_text() {
  local file=$1
  local text=$2
  local message=$3
  grep -Fq -- "$text" "$file" || fail "$message"
}

# Ordering gates below anchor on source markers and compare their line numbers.
# A marker that resolves to zero or several lines must FAIL rather than be
# swallowed: `(( ))` treats a multi-line value as a syntax error, which returns
# non-zero and therefore reads as "the ordering is fine" (#1207). Every marker
# resolution goes through one of these helpers so an ambiguous anchor is
# reported by name instead of silently disarming its check.
#
# `fail` exits the command substitution; the caller's plain assignment inherits
# that status and `set -e` ends the script. Assign these at top level only —
# `local x=$(...)` would mask the status.
marker_lines() {
  # Reads text on stdin; prints one line number per match of the fixed marker.
  grep -nF -- "$1" | cut -d: -f1
}

sole_line_of() {
  # Exactly one match of $2 in the text on stdin, reported as $1.
  local label=$1
  local marker=$2
  local -a hits=()
  local line
  while IFS= read -r line; do
    hits+=("$line")
  done < <(marker_lines "$marker")
  if ((${#hits[@]} != 1)); then
    fail "release-contract marker '${label}' resolved ${#hits[@]} lines (expected exactly 1): ${marker}"
  fi
  printf '%s\n' "${hits[0]}"
}

first_line_of() {
  # First match of $2 in the text on stdin, reported as $1. Used where the
  # marker legitimately repeats and the invariant is about the earliest one.
  local label=$1
  local marker=$2
  local -a hits=()
  local line
  while IFS= read -r line; do
    hits+=("$line")
  done < <(marker_lines "$marker")
  if ((${#hits[@]} == 0)); then
    fail "release-contract marker '${label}' resolved no lines: ${marker}"
  fi
  printf '%s\n' "${hits[0]}"
}

sole_file_line() {
  # Exactly one match of $3 inside the inclusive absolute line range
  # [${4:-1}, ${5:-end}] of file $2, reported as $1. The range is how a marker
  # that repeats verbatim in a sibling function still names one occurrence.
  local label=$1
  local file=$2
  local marker=$3
  local lo=${4:-1}
  local hi=${5:-0}
  local -a hits=()
  local line
  while IFS= read -r line; do
    hits+=("$line")
  done < <(marker_lines "$marker" <"$file" \
    | awk -v lo="$lo" -v hi="$hi" '$1 >= lo && (hi == 0 || $1 <= hi)')
  if ((${#hits[@]} != 1)); then
    fail "release-contract marker '${label}' resolved ${#hits[@]} lines in ${file} within [${lo},${hi:-end}] (expected exactly 1): ${marker}"
  fi
  printf '%s\n' "${hits[0]}"
}

block_end() {
  # Absolute line of the top-level `}` closing the block that starts at $2.
  local file=$1
  local start=$2
  local line
  line=$(awk -v s="$start" 'NR > s && $0 == "}" { print NR; exit }' "$file")
  [[ -n "$line" ]] \
    || fail "release-contract could not find the end of the block starting at ${file}:${start}"
  printf '%s\n' "$line"
}

require_text crates/mold-candle/Cargo.toml \
  'h3-private-uat = []' \
  "mold-candle does not keep the private H3 runtime behind its own feature"
require_text crates/mold-candle/Cargo.toml \
  'license = "MIT AND Apache-2.0"' \
  "mold-candle does not declare the combined license of its packaged sources"
require_text crates/mold-candle/THIRD_PARTY_NOTICES.md \
  'comfy-kitchen INT8 CUDA reference' \
  "mold-candle omits the packaged INT8 CUDA attribution notice"
package_files=$(cargo package -p mold-ai-candle --allow-dirty --no-verify --list)
for packaged_notice in LICENSE-MIT LICENSE-APACHE-2.0 THIRD_PARTY_NOTICES.md; do
  grep -Fxq "$packaged_notice" <<<"$package_files" \
    || fail "mold-candle package omits ${packaged_notice}"
done
require_text crates/mold-inference/Cargo.toml \
  'h3-private-uat = ["mold-candle/h3-private-uat"]' \
  "mold-inference does not narrowly forward the private H3 runtime feature"
require_text crates/mold-server/Cargo.toml \
  '"mold-core/h3-private-uat"' \
  "the private H3 server does not activate its validation-only core edge"
require_text crates/mold-core/src/validation.rs \
  '#[cfg(any(feature = "h3", feature = "h3-private-uat"))]' \
  "the H3 request-profile validator is not feature-gated"
require_text crates/mold-server/src/routes.rs \
  'validate_h3_private_uat_request(validation_request)' \
  "authenticated private H3 ingress re-enters the public compliance validator"
containment_rev=9b3b1bc276bc27c3e99343ee82db7f99705b9ed5
containment_patch="cudarc = { git = \"https://github.com/utensils/cudarc\", rev = \"${containment_rev}\" }"
require_text Cargo.toml "$containment_patch" \
  "the root workspace does not pin the reviewed CUDA attempt-containment revision"
require_text desktop/src-tauri/Cargo.toml "$containment_patch" \
  "the standalone desktop graph does not pin the reviewed CUDA attempt-containment revision"
inference_cuda_feature=$(sed -n '/^cuda = \[/,/^\]/p' crates/mold-inference/Cargo.toml)
grep -Fq '"dep:cudarc"' <<<"$inference_cuda_feature" \
  || fail "the inference CUDA feature does not activate the attempt-containment dependency"
require_text crates/mold-inference/Cargo.toml \
  'cudarc = { version = "0.19.8", default-features = false, features = ["driver"], optional = true }' \
  "mold-inference lacks a direct typed CUDA attempt-containment dependency"
server_cuda_feature=$(grep -E '^cuda[[:space:]]*=' crates/mold-server/Cargo.toml)
grep -Fq 'dep:cudarc' <<<"$server_cuda_feature" \
  || fail "the server CUDA feature does not activate preparation containment"
require_text crates/mold-server/Cargo.toml \
  'cudarc = { version = "0.19.8", default-features = false, features = ["driver"], optional = true }' \
  "mold-server lacks a direct typed CUDA preparation-containment dependency"
require_text flake.nix \
  '"cudarc-0.19.8" = "sha256-ARnabIhBCzahrk/kVCt5084gftGDyCBme3jxg+mvkUA=";' \
  "the standalone desktop Nix graph does not pin the CUDA containment source hash"
containment_source="source = \"git+https://github.com/utensils/cudarc?rev=${containment_rev}#${containment_rev}\""
for lockfile in Cargo.lock desktop/src-tauri/Cargo.lock; do
  locked_cudarc=$(grep -A2 '^name = "cudarc"$' "$lockfile")
  if [[ $(grep -Fc 'version = "0.19.8"' <<<"$locked_cudarc") -ne 1 ]] \
    || ! grep -Fq "$containment_source" <<<"$locked_cudarc"; then
    fail "${lockfile} does not resolve cudarc 0.19.8 exactly once from the reviewed containment revision"
  fi
done
audio_encode_checkpoint=$(sed -n \
  '/This developer-only seam lets a private Ref2VA runtime/,/fn encode_with_checkpoint/p' \
  crates/mold-candle/src/minimax_h3/audio.rs)
if ! grep -Fq '#[cfg(any(feature = "h3", feature = "h3-private-uat"))]' <<<"$audio_encode_checkpoint" \
  || ! grep -Fq 'pub fn encode_with_phase_checkpoint(' <<<"$audio_encode_checkpoint"; then
  fail "the typed H3 audio encode checkpoint is not private-feature-gated"
fi
require_text crates/mold-inference/src/minimax_h3/private_vae_adapter.rs \
  'pub(crate) fn encode_private_audio_reference(' \
  "the private H3 VAE adapter omits the reference-audio encode seam"
require_text crates/mold-inference/src/minimax_h3/private_vae_adapter.rs \
  'audio_vae.encode_with_phase_checkpoint(waveform, phase_checkpoint)' \
  "the private H3 VAE adapter bypasses the typed audio encode checkpoint"
require_text crates/mold-inference/src/minimax_h3/private_vae_adapter.rs \
  'phase: H3PipelinePhase::ReferenceAudioEncodeChunk,' \
  "the private H3 audio encode seam omits attempt-scoped phase checkpoints"
require_text crates/mold-inference/Cargo.toml \
  'required-features = ["dev-bins", "h3-private-uat"]' \
  "the private H3 artifact qualifier is reachable without both development features"
qwen_capture_bin=$(sed -n \
  '/name = "h3_qwen_layer50_capture"/,/^$/p' \
  crates/mold-inference/Cargo.toml)
if ! grep -Fq 'path = "src/bin/h3_qwen_layer50_capture.rs"' <<<"$qwen_capture_bin" \
  || ! grep -Fq 'required-features = ["dev-bins", "h3-private-uat"]' \
    <<<"$qwen_capture_bin"; then
  fail "the exact-BF16 Qwen capture adapter is reachable outside private development builds"
fi
require_text crates/mold-inference/src/bin/h3_qwen_layer50_capture.rs \
  'load_streamed_bf16_conditioner' \
  "the Qwen layer-50 capture adapter does not use the official streamed BF16 loader"
require_text crates/mold-inference/src/bin/h3_qwen_layer50_capture.rs \
  'Device::new_cuda' \
  "the exact-BF16 Qwen capture adapter does not fail closed to CUDA execution"
require_text crates/mold-inference/src/bin/h3_qwen_layer50_capture.rs \
  'mold.minimax-h3.private-uat-exact-bf16-qwen-layer50-capture.v1' \
  "the exact-BF16 Qwen capture adapter has no release-rejectable claim marker"
if grep -Eq 'load_h3_qwen_nvfp4_conditioner|private_qwen' \
  crates/mold-inference/src/bin/h3_qwen_layer50_capture.rs; then
  fail "the exact-BF16 Qwen capture adapter references the quantized deployment loader"
fi
visual_capture_bin=$(sed -n \
  '/name = "h3_visual_vae_capture"/,/^$/p' \
  crates/mold-inference/Cargo.toml)
if ! grep -Fq 'path = "src/bin/h3_visual_vae_capture.rs"' <<<"$visual_capture_bin" \
  || ! grep -Fq 'required-features = ["dev-bins", "h3-private-uat", "cuda"]' \
    <<<"$visual_capture_bin"; then
  fail "the visual-VAE capture adapter is reachable outside private CUDA development builds"
fi
require_text crates/mold-inference/src/bin/h3_visual_vae_capture.rs \
  'validate_diffusers_weight_index' \
  "the visual-VAE capture adapter does not use the official Diffusers F32 loader"
require_text crates/mold-inference/src/bin/h3_visual_vae_capture.rs \
  'DecodeComputePolicy::Reference' \
  "the visual-VAE capture adapter does not select the released decode policy"
require_text crates/mold-inference/src/bin/h3_visual_vae_capture.rs \
  'VisualAttentionBackend::Math' \
  "the visual-VAE capture adapter does not pin math attention"
require_text crates/mold-inference/src/bin/h3_visual_vae_capture.rs \
  'mold.minimax-h3.private-uat-visual-vae-f32-fp16-capture.v1' \
  "the visual-VAE capture adapter has no release-rejectable claim marker"
if grep -Eq 'validate_comfy_weight_file|OfficialComfySource' \
  crates/mold-inference/src/bin/h3_visual_vae_capture.rs; then
  fail "the exact visual-VAE capture adapter references the Comfy deployment authority"
fi
audio_capture_bin=$(sed -n \
  '/name = "h3_audio_vae_capture"/,/^$/p' \
  crates/mold-inference/Cargo.toml)
if ! grep -Fq 'path = "src/bin/h3_audio_vae_capture.rs"' <<<"$audio_capture_bin" \
  || ! grep -Fq 'required-features = ["dev-bins", "h3-private-uat"]' \
    <<<"$audio_capture_bin"; then
  fail "the exact-FP32 AudioVAE capture adapter is reachable outside private development builds"
fi
require_text crates/mold-inference/src/bin/h3_audio_vae_capture.rs \
  'load_validated_audio_vae(' \
  "the AudioVAE capture adapter does not use the validated production loader"
require_text crates/mold-inference/src/bin/h3_audio_vae_capture.rs \
  'DType::F32' \
  "the exact-FP32 AudioVAE capture adapter does not fail closed to FP32 execution"
require_text crates/mold-inference/src/bin/h3_audio_vae_capture.rs \
  'Device::new_cuda' \
  "the exact-FP32 AudioVAE capture adapter does not fail closed to CUDA execution"
require_text crates/mold-inference/src/bin/h3_audio_vae_capture.rs \
  'encode_with_phase_checkpoint' \
  "the AudioVAE capture adapter bypasses the typed encode phase checkpoints"
require_text crates/mold-inference/src/bin/h3_audio_vae_capture.rs \
  'decode_with_phase_checkpoint' \
  "the AudioVAE capture adapter bypasses the typed decode phase checkpoints"
require_text crates/mold-inference/src/bin/h3_audio_vae_capture.rs \
  'mold.minimax-h3.private-uat-exact-fp32-audio-vae-capture.v1' \
  "the exact-FP32 AudioVAE capture adapter has no release-rejectable claim marker"
transformer_capture_bin=$(sed -n \
  '/name = "h3_transformer_capture"/,/^$/p' \
  crates/mold-inference/Cargo.toml)
if ! grep -Fq 'path = "src/bin/h3_transformer_capture.rs"' <<<"$transformer_capture_bin" \
  || ! grep -Fq 'required-features = ["dev-bins", "h3-private-uat"]' \
    <<<"$transformer_capture_bin"; then
  fail "the transformer capture adapter is reachable outside private development builds"
fi
require_text crates/mold-inference/src/bin/h3_transformer_capture.rs \
  'capture_h3_private_transformer_primitives(' \
  "the transformer capture adapter bypasses the production observer seam"
require_text crates/mold-inference/src/bin/h3_transformer_capture.rs \
  'mold.minimax-h3.private-uat-transformer-capture.v1' \
  "the transformer capture adapter has no release-rejectable claim marker"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  '"mold.minimax-h3.private-uat-artifact-reader.v1"' \
  "the private H3 artifact reader has no release-rejectable claim marker"
require_text crates/mold-inference/src/minimax_h3/private_qwen_support.rs \
  '"mold.minimax-h3.private-uat-qwen-support-loader.v1"' \
  "the private H3 Qwen support loader has no release-rejectable claim marker"
require_text crates/mold-inference/Cargo.toml \
  'name = "h3_runtime_qualification_record"' \
  "the private H3 runtime-record producer is not an explicit development binary"
require_text crates/mold-inference/Cargo.toml \
  'required-features = ["dev-bins", "h3-cuda"]' \
  "the runtime-record producer does not bind the exact public H3 runtime identity"
require_text crates/mold-inference/src/minimax_h3/private_runtime_qualification.rs \
  '"mold.minimax-h3.private-runtime-record-producer.v1"' \
  "the private H3 runtime-record producer has no release-rejectable claim marker"
require_text crates/mold-inference/src/minimax_h3/private_runtime_qualification.rs \
  'qualify_private_artifacts(' \
  "the private H3 runtime-record producer does not re-authenticate the full artifact set"
require_text crates/mold-inference/src/minimax_h3/private_runtime_qualification.rs \
  'fn entries(&self) -> [(&' \
  "the private H3 runtime-record producer has no fixed bound-cardinality contract"
require_text crates/mold-inference/src/minimax_h3/private_runtime_qualification.rs \
  'H3PrivateRuntimeQualificationCandidate' \
  "the private H3 runtime-record producer does not emit a review-only candidate"
require_text crates/mold-inference/src/minimax_h3/private_runtime_observer.rs \
  'mold.minimax-h3.private-uat-runtime-bound-observation.v5' \
  "the private H3 runtime-bound observer has no release-rejectable claim marker"
require_text scripts/verify-h3-release-exclusion.sh \
  'mold.minimax-h3.private-uat-runtime-bound-observation.v1' \
  "shipping verification forgot the prior private H3 runtime-bound observer marker"
require_text scripts/verify-h3-release-exclusion.sh \
  'mold.minimax-h3.private-uat-runtime-bound-observation.v2' \
  "shipping verification forgot the prior v2 private H3 runtime-bound observer marker"
require_text scripts/verify-h3-release-exclusion.sh \
  'mold.minimax-h3.private-uat-runtime-bound-observation.v3' \
  "shipping verification forgot the prior v3 private H3 runtime-bound observer marker"
require_text scripts/verify-h3-release-exclusion.sh \
  'mold.minimax-h3.private-uat-runtime-bound-observation.v4' \
  "shipping verification forgot the prior v4 private H3 runtime-bound observer marker"
require_text scripts/verify-h3-release-exclusion.sh \
  'mold.minimax-h3.private-uat-runtime-bound-observation.v5' \
  "shipping verification does not reject the current private H3 runtime-bound observer marker"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'H3PrivateRuntimeBoundCapture::begin(' \
  "the private H3 server does not synchronously own one runtime-bound capture"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'runtime_bound_capture.finish()' \
  "the private H3 server does not validate runtime bounds before returning"
require_text scripts/verify-h3-release-exclusion.sh \
  'private_runtime_observation_markers=(' \
  "shipping verification does not reject the private H3 runtime-bound observer"
require_text crates/mold-inference/src/minimax_h3/private_runtime_qualification.rs \
  'hash_measured_server_executable(' \
  "the private H3 runtime-record producer does not authenticate its measured server"
require_text crates/mold-inference/src/lib.rs \
  'h3_private_runtime_code_identity_sha256' \
  "the private H3 inference facade does not expose its measured code identity"
require_text crates/mold-server/src/main.rs \
  'mold_inference::h3_private_runtime_code_identity_sha256()' \
  "the private H3 server ELF does not retain its measured code identity"
if grep -Fq 'NoopH3PipelineObserver' \
  crates/mold-inference/src/minimax_h3/private_server.rs; then
  fail "the private H3 server suppresses timestampable pipeline progress"
fi
require_text crates/mold-inference/src/minimax_h3/private_runtime_qualification.rs \
  'MAX_RUNTIME_QUALIFICATION_BYTES' \
  "the private H3 runtime-record producer does not share the activation record limit"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'campaign_runtime_code_identity_sha256: String,' \
  "the private H3 runtime record omits its stable campaign code identity"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'measured_server_executable_sha256: String,' \
  "the private H3 runtime record omits its measured server identity"
require_text crates/mold-inference/build_support/h3_runtime_code_identity.rs \
  'const LOCAL_RUNTIME_CRATES: &[&str] = &[' \
  "the private H3 runtime code identity omits its reviewed local dependency closure"
require_text crates/mold-inference/build_support/h3_runtime_code_identity.rs \
  'runtime identity entry {} is a symbolic link' \
  "the private H3 runtime code identity does not reject linked traversal entries"
require_text crates/mold-inference/build_support/h3_runtime_code_identity.rs \
  'RUSTC_VERBOSE_VERSION' \
  "the private H3 runtime code identity omits the compiler identity"
require_text crates/mold-inference/build_support/h3_runtime_code_identity.rs \
  'key.starts_with("CARGO_FEATURE_")' \
  "the private H3 runtime code identity omits enabled Cargo features"
require_text crates/mold-inference/build.rs \
  'cargo:rerun-if-env-changed={key}' \
  "the private H3 runtime code identity does not track build-environment changes"
require_text crates/mold-inference/build.rs \
  '&& std::env::var_os("CARGO_FEATURE_H3_PRIVATE_UAT").is_none()' \
  "ordinary non-H3 builds are not isolated from the H3 identity collector"
require_text crates/mold-server/Cargo.toml \
  '"mold-inference/dev-bins",' \
  "the measured private server does not match the producer inference feature set"
require_text crates/mold-server/build.rs \
  'validate_canonical_h3_server_features()' \
  "the measured private server does not reject non-canonical campaign features"
require_text crates/mold-inference/build_support/h3_runtime_code_identity.rs \
  '"MOLD_H3_CANONICAL_SERVER_FEATURES"' \
  "the private H3 runtime identity omits the canonical server feature set"
require_text crates/mold-inference/build_support/h3_runtime_code_identity.rs \
  '"MOLD_H3_NVCC_VERSION"' \
  "the private H3 runtime identity cannot distinguish an in-place CUDA toolkit upgrade"
require_text crates/mold-inference/build.rs \
  'collect_native_toolchain_identity()' \
  "the private H3 identity does not invalidate when a native compiler is replaced"
# mold-ai-server is published to crates.io, and a published `.crate` carries
# only its own crate directory, so its build script cannot read the inference
# copy of this list. The two are separate files on purpose; they must not drift,
# because one enforces the campaign feature set and the other hashes it into the
# runtime-code identity.
canonical_features_of() {
  sed -n '/CANONICAL_H3_SERVER_FEATURES: &\[&str\] = &\[/,/^\];/p' "$1" \
    | sed -n 's/^ *"\(CARGO_FEATURE_[A-Z0-9_]*\)",$/\1/p'
}
inference_canonical_features=$(
  canonical_features_of crates/mold-inference/build_support/h3_runtime_code_identity.rs
)
server_canonical_features=$(
  canonical_features_of crates/mold-server/build_support/h3_server_features.rs
)
if [[ -z "$inference_canonical_features" || -z "$server_canonical_features" ]]; then
  fail "the canonical private H3 server feature set could not be read from both build-support files"
fi
if [[ "$inference_canonical_features" != "$server_canonical_features" ]]; then
  fail "the canonical private H3 server feature set drifted between mold-inference and mold-server"
fi
require_text crates/mold-server/build_support/h3_server_features.rs \
  'CARGO_FEATURE_H3_PRIVATE_UAT' \
  "the measured private server build support omits its own campaign gate"
require_text scripts/verify-h3-release-exclusion.sh \
  'private_runtime_record_marker=' \
  "published-binary verification does not reject the private runtime-record producer"
require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
  '_activation: admitted_overlap_seal::Token,' \
  "private H3 FL2VA overlap authority no longer contains its private issuer token"
require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
  'b"mold.minimax-h3.private-fl2va-overlap.v3\0"' \
  "private H3 FL2VA overlap identity does not use its scheduler-ledger-bound version-three domain"
if grep -Eq 'private-fl2va-overlap\.v(1|2)' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs; then
  fail "private H3 FL2VA overlap identity still accepts a pre-ledger serializer domain"
fi
require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
  'pub(crate) fn issue_private_fl2va_memory_overlap(' \
  "private H3 FL2VA overlap has no scheduler-ledger-bound production issuer"
require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
  'scheduler_ledger_identity_sha256: ledger.identity_sha256().into(),' \
  "private H3 FL2VA overlap does not retain the issuing scheduler ledger identity"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'f624f71ce1eba7ebb75a13801da855a92f5eec0fccbcb9783f547479c7abfce5' \
  "private H3 runtime qualification omits the independently reviewed compact FL2VA candidate"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'pub const fn reviewed_h3_private_runtime_available() -> bool {' \
  "private H3 ingress has no zero-I/O reviewed-runtime availability probe"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'pub fn prepare_h3_private_fl2va_admission(' \
  "private H3 scheduler has no allocation-free inference-owned admission facade"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'pub fn validate_for(' \
  "private H3 admission DTO cannot revalidate its exact request, route, and fit facts"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'b"mold.minimax-h3.private-admission-evidence.v2\0"' \
  "private H3 admission evidence does not bind canonical host headroom"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'prepared_attempt_identity_sha256: String,' \
  "private H3 owner fence no longer binds the exact prepared-attempt identity"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'prepared.prepared_attempt_identity_sha256()' \
  "private H3 owner rebuild no longer compares the prepared-attempt identity"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'prepare_reviewed_h3_private_fl2va_attempt(input, progress)' \
  "private H3 reviewed record cannot enter the complete atomic preparation path"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'let phase_owner = bind_private_comfy_fl2va_phase_owner(' \
  "private H3 server facade does not bind its prepared evidence into the singular phase owner"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'commit_private_h3_allocation_then(&mut allocation_commit, || {' \
  "private H3 server facade does not commit the scheduler allocation before execution-device construction"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'let mut attempt = CudaExecutionAttempt::begin_unbound()' \
  "private H3 server facade does not install containment before CUDA context construction"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));' \
  "private H3 server facade does not catch the complete CUDA attempt"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'attempt.mark_panicked();' \
  "private H3 server facade does not mark caught panics before attempt teardown"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'CUDA execution attempt retained resources; server restart required' \
  "private H3 server facade does not convert retained resources into restart-required failure"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  '.context("private H3 completion synchronization failed")?;' \
  "private H3 server facade does not synchronize successful work before teardown"
runner_source=$(awk '
  /^impl H3PrivateFl2VaPreparedRunner for H3PrivateConcretePreparedRunner/ { capture = 1 }
  capture { print }
  capture && /^fn private_run_output/ { exit }
' crates/mold-inference/src/minimax_h3/private_server.rs)
boundary_line=$(sole_line_of 'runner containment boundary' \
  'with_private_h3_cuda_execution_attempt(||' <<<"$runner_source")
commit_line=$(sole_line_of 'runner allocation commit' \
  'commit_private_h3_allocation_then' <<<"$runner_source")
device_line=$(sole_line_of 'runner CUDA construction' \
  'Device::new_cuda' <<<"$runner_source")
execute_line=$(sole_line_of 'runner FL2VA execution' \
  'run_private_comfy_fl2va_attempt' <<<"$runner_source")
sync_line=$(sole_line_of 'runner completion synchronization' \
  '.synchronize()' <<<"$runner_source")
if ((boundary_line >= commit_line || commit_line >= device_line \
  || device_line >= execute_line || execute_line >= sync_line)); then
  fail "private H3 containment, allocation, construction, execution, and completion ordering changed"
fi
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  '.runner' \
  "private H3 prepared attempt no longer retains its one-shot runner"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'private H3 prepared attempt was already consumed' \
  "private H3 prepared attempt no longer rejects replay"
prepare_source=$(sed -n \
  '/fn prepare_reviewed_h3_private_fl2va_attempt/,/fn validate_base_factory/p' \
  crates/mold-inference/src/minimax_h3/private_server.rs)
if grep -Fq 'Device::new_cuda' <<<"$prepare_source"; then
  fail "private H3 preparation constructs a CUDA device before the owner allocation commit"
fi
reviewed_line=$(first_line_of 'prepared reviewed-record gate' \
  'private_runtime_qualification_source(' <<<"$prepare_source")
artifact_line=$(first_line_of 'prepared artifact qualification' \
  'qualify_private_artifacts_with_control' <<<"$prepare_source")
if ((reviewed_line >= artifact_line)); then
  fail "private H3 reviewed-record hash gate no longer precedes bulk artifact qualification"
fi
admission_source=$(sed -n \
  '/fn prepare_reviewed_h3_private_fl2va_admission/,/struct H3PrivateComponentDigest/p' \
  crates/mold-inference/src/minimax_h3/private_server.rs)
if grep -Fq 'Device::new_cuda' <<<"$admission_source"; then
  fail "private H3 admission constructs a CUDA device"
fi
admission_reviewed_line=$(first_line_of 'admission reviewed-record gate' \
  'private_runtime_qualification_source(' <<<"$admission_source")
admission_artifact_line=$(first_line_of 'admission artifact qualification' \
  'qualify_private_artifacts_with_control' <<<"$admission_source")
if ((admission_reviewed_line >= admission_artifact_line)); then
  fail "private H3 admission no longer checks reviewed evidence before bulk artifact I/O"
fi
require_text crates/mold-inference/src/minimax_h3/private_runtime.rs \
  'pub(crate) struct H3PrivateBoundComfyStream' \
  "private H3 checkpoint load lacks a non-cloneable pre-allocation binding"
require_text crates/mold-inference/src/minimax_h3/private_runtime.rs \
  'bound: H3PrivateBoundComfyStream,' \
  "private H3 resident loader still accepts unbound checkpoint and attention values"
require_text crates/mold-inference/src/minimax_h3/private_runtime.rs \
  'device: Device,' \
  "private H3 bound checkpoint does not retain the validated Candle device"
attempt_owner=$(sed -n \
  '/struct H3PrivateAttemptOwner/,/^}/p' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs)
if ! grep -Fq 'device: Device,' <<<"$attempt_owner"; then
  fail "private H3 singular attempt owner does not retain the validated Candle device"
fi
require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
  'struct H3PrivateBoundExecution<E>' \
  "private H3 route capture returns forgeable lease/device parts"
require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
  'fn new(bound: H3PrivateBoundExecution<E>, artifacts: A)' \
  "private H3 attempt construction can mix an execution lease with another device"
execution_projection=$(sed -n \
  '/impl<E, A> H3BackendExecutionLease for H3PrivateExecutionProjection/,/^}/p' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs)
if ! grep -Fq '&self.owner.device' <<<"$execution_projection" \
  || grep -Fq 'execution.device()' <<<"$execution_projection"; then
  fail "private H3 execution projections can repoll a safe lease for a different device"
fi
require_text crates/mold-inference/src/minimax_h3/private_runtime.rs \
  '.verify_current_release_candidate_dispatch(device)' \
  "private H3 binding does not verify the actual compiled attention candidate"
for method in \
  transformer_task \
  transformer_checkpoint_content_sha256 \
  transformer_checkpoint_layout_identity_sha256 \
  transformer_checkpoint_identity_sha256 \
  transformer_policy_identity_sha256; do
  require_text crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs \
    "fn ${method}(&self)" \
    "private H3 artifact lease omits ${method}"
done
owner_bind=$(sed -n \
  '/pub(crate) fn bind_private_comfy_fl2va_phase_owner/,/^}/p' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs)
prepared_line=$(sole_line_of 'owner prepared revalidation' \
  'prepared.revalidate()?;' <<<"$owner_bind")
qwen_capture_line=$(sole_line_of 'owner Qwen artifact capture' \
  'H3PrivateQwenArtifactAuthority::capture(&qwen_support, &opened_qwen)?;' <<<"$owner_bind")
overlap_line=$(sole_line_of 'owner overlap binding' \
  'validate_prepared_overlap_binding(' <<<"$owner_bind")
bind_line=$(sole_line_of 'owner checkpoint binding' \
  'let bound_transformer = bind_private_comfy_stream(' <<<"$owner_bind")
attempt_line=$(sole_line_of 'owner singular attempt' \
  'let attempt = H3PrivateAttemptAuthority::new' <<<"$owner_bind")
if ((prepared_line >= qwen_capture_line || qwen_capture_line >= overlap_line \
  || overlap_line >= bind_line || bind_line >= attempt_line)); then
  fail "private H3 prepared, Qwen, overlap, checkpoint, and singular-attempt binding order changed"
fi
if grep -Fq 'load_authorized_from_opened' <<<"$owner_bind"; then
  fail "private H3 owner binding eagerly allocates Qwen before the VAE phase"
fi
if [[ $(grep -Fc 'capture_private_execution_route(execution_lease' <<<"$owner_bind") -ne 1 ]] \
  || grep -Fq 'execution_lease.device()' <<<"$owner_bind"; then
  fail "private H3 owner binding must consume and capture the execution lease route exactly once"
fi
if [[ $(grep -Fc 'bound_execution.device()' <<<"$owner_bind") -ne 1 ]] \
  || [[ $(grep -Fc 'H3PrivateAttemptAuthority::new(bound_execution, artifact_lease)' <<<"$owner_bind") -ne 1 ]]; then
  fail "private H3 bound execution token is not shared with checkpoint binding and consumed by the attempt"
fi
runtime_source=crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs
# The Ref2VA backend and its attempt are verbatim twins of the FL2VA phase
# markers, so each anchor is resolved inside the block that identifies the
# FL2VA occurrence rather than by a whole-file grep. `H3PrivatePhaseBackend`'s
# `H3Fl2VaBackend` owns the encode/park/denoise/decode phases;
# `run_private_comfy_fl2va_attempt` owns the terminal identity echo and mux.
fl2va_impl_line=$(sole_file_line 'FL2VA backend impl' "$runtime_source" \
  'impl<C, E, A> H3Fl2VaBackend for H3PrivatePhaseBackend<C, E, A>')
fl2va_impl_end=$(block_end "$runtime_source" "$fl2va_impl_line")
fl2va_encode_text_line=$(sole_file_line 'FL2VA text encode' "$runtime_source" \
  'fn encode_text(' "$fl2va_impl_line" "$fl2va_impl_end")
fl2va_encode_visual_line=$(sole_file_line 'FL2VA visual condition encode' "$runtime_source" \
  'fn encode_visual_condition(' "$fl2va_impl_line" "$fl2va_impl_end")
fl2va_attempt_line=$(sole_file_line 'FL2VA attempt' "$runtime_source" \
  'pub(crate) fn run_private_comfy_fl2va_attempt<C, E, A>(')
fl2va_attempt_end=$(block_end "$runtime_source" "$fl2va_attempt_line")
fl2va_decode_audio_line=$(sole_file_line 'FL2VA audio decode' "$runtime_source" \
  'fn decode_audio(' "$fl2va_impl_line" "$fl2va_impl_end")
vae_load_line=$(sole_file_line 'VAE construction' "$runtime_source" \
  'self.construct_vaes(opened_vae, checkpoint)?;' \
  "$fl2va_encode_text_line" "$((fl2va_encode_visual_line - 1))")
qwen_load_line=$(sole_file_line 'FL2VA Qwen load' "$runtime_source" \
  'let qwen = H3PrivateQwenAdapter::load_authorized_from_opened(' \
  "$fl2va_impl_line" "$fl2va_impl_end")
vae_park_line=$(sole_file_line 'FL2VA VAE park' "$runtime_source" \
  'drop(self.vae.take());' "$fl2va_impl_line" "$((fl2va_decode_audio_line - 1))")
transformer_load_line=$(sole_file_line 'FL2VA transformer load' "$runtime_source" \
  'let stream = load_and_pair_private_comfy_stream(' "$fl2va_impl_line" "$fl2va_impl_end")
transformer_drop_line=$(sole_file_line 'FL2VA transformer drop' "$runtime_source" \
  'drop(self.denoiser.take());' "$fl2va_impl_line" "$fl2va_impl_end")
vae_reload_line=$(sole_file_line 'FL2VA VAE reload' "$runtime_source" \
  'self.construct_vaes(reload, checkpoint)?;' "$fl2va_impl_line" "$fl2va_impl_end")
vae_drop_line=$(sole_file_line 'FL2VA VAE drop' "$runtime_source" \
  'drop(self.vae.take());' "$fl2va_decode_audio_line" "$fl2va_impl_end")
terminal_line=$(sole_file_line 'FL2VA terminal identity echo' "$runtime_source" \
  'let identity_echo = backend.terminal_identity_echo()?;' \
  "$fl2va_attempt_line" "$fl2va_attempt_end")
mux_line=$(sole_file_line 'FL2VA AV mux' "$runtime_source" \
  'let output = super::pipeline::finalize_av(' "$fl2va_attempt_line" "$fl2va_attempt_end")
if ((qwen_load_line >= vae_load_line || vae_load_line >= vae_park_line \
  || vae_park_line >= transformer_load_line \
  || transformer_load_line >= transformer_drop_line \
  || transformer_drop_line >= vae_reload_line || vae_reload_line >= vae_drop_line \
  || vae_drop_line >= terminal_line || terminal_line >= mux_line)); then
  fail "private H3 phase runtime no longer drops Qwen before VAE construction or parks, reloads, and drops components before terminal-only mux"
fi
# The developer-only Ref2VA attempt owns the same terminal-before-mux rule.
ref2va_attempt_line=$(sole_file_line 'Ref2VA attempt' "$runtime_source" \
  'pub(crate) fn run_private_comfy_ref2va_attempt<C, E, A>(')
ref2va_attempt_end=$(block_end "$runtime_source" "$ref2va_attempt_line")
ref2va_terminal_line=$(sole_file_line 'Ref2VA terminal identity echo' "$runtime_source" \
  'let identity_echo = backend.terminal_identity_echo()?;' \
  "$ref2va_attempt_line" "$ref2va_attempt_end")
ref2va_mux_line=$(sole_file_line 'Ref2VA AV mux' "$runtime_source" \
  'let output = super::pipeline::finalize_av(' "$ref2va_attempt_line" "$ref2va_attempt_end")
if ((ref2va_terminal_line >= ref2va_mux_line)); then
  fail "private H3 Ref2VA attempt no longer captures its terminal identity before the mux"
fi
require_text "$runtime_source" \
  'pub(crate) struct H3PrivatePhaseIdentityEcho {' \
  "private H3 runtime output omits its server bridge identity echo"
require_text "$runtime_source" \
  'with_contained_private_cuda_resources(backend, |backend| {' \
  "private H3 runtime does not retain its concrete backend across panic and fatal CUDA errors"
require_text "$runtime_source" \
  'let mut resources = ManuallyDrop::new(resources);' \
  "private H3 runtime containment does not suppress suspect CUDA destructors"
require_text "$runtime_source" \
  'Ok(Err(error)) if is_fatal_private_cuda_error(&error) => Err(error),' \
  "private H3 runtime containment does not retain concrete resources on fatal CUDA errors"
for marker in \
  CUDA_ERROR_EXTERNAL_DEVICE \
  CUDA_ERROR_MPS_CLIENT_TERMINATED \
  CUDA_ERROR_CONTAINED \
  CUDA_ERROR_TENSOR_MEMORY_LEAK \
  CUBLAS_STATUS_MAPPING_ERROR \
  CUBLAS_STATUS_EXECUTION_FAILED \
  CUBLAS_STATUS_INTERNAL_ERROR \
  CURAND_STATUS_LAUNCH_FAILURE \
  CURAND_STATUS_PREEXISTING_FAILURE \
  CURAND_STATUS_INTERNAL_ERROR; do
  require_text "$runtime_source" "$marker" \
    "private H3 runtime fatal classifier omits ${marker}"
  require_text crates/mold-server/src/gpu_worker.rs "$marker" \
    "server restart classifier omits ${marker}"
  if [[ "$marker" == CUDA_ERROR_* ]]; then
    require_text crates/mold-inference/src/device.rs "$marker" \
      "typed device-memory fatal classifier omits ${marker}"
  fi
done
require_text crates/mold-server/src/gpu_worker.rs \
  'CUDA execution attempt retained resources' \
  "server restart classifier omits the generic retained-resource marker"
require_text crates/mold-server/src/gpu_worker.rs \
  'with_private_h3_cuda_preparation_attempt(worker, || {' \
  "server owner preparation is not inside a separate CUDA containment scope"
require_text crates/mold-server/src/gpu_worker.rs \
  'request_lease_host_headroom(' \
  "private H3 live owner does not request ledger-aware host headroom"
require_text crates/mold-server/src/scheduler/mod.rs \
  'WorkerEvent::HostMemoryRecheck { fence, reply }' \
  "Scheduler V2 does not reconcile the private H3 live host-memory fence"
require_text crates/mold-server/src/scheduler/mod.rs \
  '.filter(|(candidate, _)| candidate.as_str() != work_id)' \
  "the private H3 live host-memory fence does not retain every peer reservation"
require_text crates/mold-server/src/scheduler/mod.rs \
  'charge_until_release: bool,' \
  "Scheduler V2 cannot keep gradual private H3 allocations reserved through lease release"
require_text crates/mold-server/src/scheduler/mod.rs \
  '!reservation.charge_until_release' \
  "Scheduler V2 can absorb a gradual private H3 reservation into a premature host sample"
require_text crates/mold-server/src/scheduler/mod.rs \
  'charge_until_release: item.host_ram_bytes > 0,' \
  "Scheduler V2 does not keep host-memory leases charged through release"
require_text crates/mold-server/src/scheduler/mod.rs \
  'fn h3_reservation_stays_charged_until_release_and_blocks_ordinary_work()' \
  "Scheduler V2 lacks a regression for private H3 host-memory isolation"
metrics_endpoint=$(sed -n \
  '/pub async fn metrics_endpoint(/,/HTTP metrics middleware/p' \
  crates/mold-server/src/metrics.rs)
if grep -Fq 'vram_in_use_bytes' <<<"$metrics_endpoint"; then
  fail "the HTTP metrics task can enter CUDA outside the serialized owner"
fi
device_raw_source=crates/mold-inference/src/device.rs
if [[ $(grep -Fc 'context.preflight_raw_call()' "$device_raw_source") -lt 5 ]]; then
  fail "direct CUDA device and memory-pool calls are not all attempt-preflighted"
fi
pinned_source=crates/mold-inference/src/flux/pinned.rs
if [[ $(grep -Fc 'preflight_current_execution_attempt_latch()' "$pinned_source") -ne 2 ]] \
  || [[ $(grep -Fc '.result()' "$pinned_source") -lt 2 ]]; then
  fail "raw pinned-memory registration does not preflight and observe the attempt latch"
fi
require_text "$runtime_source" \
  'let identity_echo = backend.terminal_identity_echo()?;' \
  "private H3 runtime does not derive its identity echo from the retained terminal authority"
require_text "$runtime_source" \
  'if stream.authority != self.stream_authority {' \
  "private H3 loaded transformer discards its retained bound authority"
for identity in \
  support_identity_sha256 \
  weight_identity_sha256 \
  weight_header_identity_sha256 \
  weight_policy_identity_sha256; do
  require_text "$runtime_source" \
    "facts.${identity} != qwen.${identity}" \
    "private H3 continuing artifact validation omits retained Qwen ${identity}"
done
terminal_attempt=$(sed -n '/struct H3PrivateTerminalAttempt/,/^}/p' "$runtime_source")
if ! grep -Fq 'qwen_artifact_authority: H3PrivateQwenArtifactAuthority,' \
  <<<"$terminal_attempt"; then
  fail "private H3 terminal mux proof does not retain exact Qwen artifact authority"
fi
require_text "$runtime_source" \
  'device_id: live.device_id,' \
  "private H3 bridge echo copies planned rather than validated live route identity"
if grep -Fq 'unsafe impl<T> H3PrivateQwenArtifactLease for &T' \
  crates/mold-inference/src/minimax_h3/private_qwen.rs; then
  fail "private H3 Qwen artifact authority escapes through a borrowed blanket implementation"
fi
capture=$(sed -n \
  '/fn capture_private_execution_route/,/^}/p' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs)
if [[ $(grep -Fc 'execution.device()' <<<"$capture") -ne 1 ]] \
  || [[ $(grep -Fc 'lease_id: execution.lease_id().into()' <<<"$capture") -ne 1 ]]; then
  fail "private H3 route capture must bind one device and a concrete lease identity"
fi
loader=$(sed -n \
  '/pub(crate) fn load_and_pair_private_comfy_stream/,/^[[:space:]]*{/p' \
  crates/mold-inference/src/minimax_h3/private_runtime.rs)
if grep -Fq 'device:' <<<"$loader"; then
  fail "private H3 resident loader still accepts an external device after binding"
fi
overlap_constructor=$(sed -n \
  '/impl H3PrivateFl2VaMemoryOverlapAuthority {/,/fn validate(&self)/p' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs)
if ! grep -Fq '#[cfg(test)]' <<<"$overlap_constructor"; then
  fail "private H3 FL2VA overlap authority exposes a non-test constructor"
fi
if [[ $(grep -Fc 'pub(crate) fn new(' <<<"$overlap_constructor") -ne 1 ]] \
  || [[ $(grep -Fc 'Self {' <<<"$overlap_constructor") -ne 1 ]]; then
  fail "private H3 FL2VA overlap authority gained an unguarded construction path"
fi
if grep -Eq '^[[:space:]]*pub([[:space:]]*\([^)]*\))?[[:space:]]+(type[[:space:]]+H3PrivateComfyPipelineRuntime|fn[[:space:]]+compose_private_comfy_fl2va_runtime)' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs; then
  fail "private H3 FL2VA composition exposes an into-runtime escape"
fi
if grep -Eq '(transmute|MaybeUninit|::zeroed|unsafe[[:space:]]*\{).*(H3PrivateFl2VaMemoryOverlapAuthority|admitted_overlap_seal)' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs; then
  fail "private H3 FL2VA overlap authority has an unsafe fabrication path"
fi
if [[ $(grep -Fxc 'mod vae_free_inner_seal {' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs) -ne 1 ]] \
  || [[ $(grep -Fxc 'pub(crate) mod vae_free_inner_seal {' \
  crates/mold-inference/src/minimax_h3/private_fl2va_runtime.rs) -ne 1 ]]; then
  fail "private H3 VAE-free seal is not private in production and test-visible only"
fi
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'pub const H3_PRIVATE_AUTHORIZATION_SCOPE: &str = "private-h3-uat";' \
  "private H3 qualification does not use the generic private authorization scope"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'const AUTHORIZATION_SCHEMA: &str = "mold.minimax-h3.authorization.v1";' \
  "private H3 qualification does not require the reviewed authorization schema"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'const REVIEWED_AUTHORIZATION_EVIDENCE_SHA256: &str =' \
  "private H3 qualification does not pin the accepted authorization evidence"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  '8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d' \
  "private H3 qualification pins a different authorization evidence identity"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'if record.source_document_sha256 != reviewed_evidence_sha256 {' \
  "private H3 qualification does not reject self-declared authorization evidence"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'fn production_scope_rejects_unreviewed_evidence() {' \
  "private H3 qualification lacks a regression test for unreviewed evidence"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'pub authorization_record_sha256: String,' \
  "private H3 qualification report does not bind the authorization record identity"
require_text crates/mold-inference/src/minimax_h3/private_qualification.rs \
  'pub authorization_source_document_sha256: String,' \
  "private H3 qualification report does not bind the authorization source identity"
require_text crates/mold-inference/src/bin/h3_artifact_qualification.rs \
  'remove("--authorization-record")' \
  "private H3 qualifier CLI does not require the external authorization record"
if grep -Eq 'H3_PRIVATE_HOST_AUTHORITY_SHA256|H3_PRIVATE_MODELS_ROOT|/etc/hostname|/storage/.*/minimax-h3' \
  crates/mold-inference/src/minimax_h3/private_qualification.rs \
  crates/mold-inference/src/bin/h3_artifact_qualification.rs; then
  fail "private H3 qualification regressed to host-name or hardcoded-path authority"
fi
if grep -Fq -- '--authorization-scope' \
  crates/mold-inference/src/bin/h3_artifact_qualification.rs; then
  fail "private H3 qualifier accepts a caller-asserted authorization scope"
fi

if grep -Eq '^h3-private-uat[[:space:]]*=' crates/mold-cli/Cargo.toml; then
  fail "mold-ai forwards the private H3 UAT feature into a runnable product binary"
fi
server_dev_dependencies=$(sed -n '/^\[dev-dependencies\]/,/^\[/p' crates/mold-server/Cargo.toml)
if ! grep -Eq '^mold-inference[[:space:]]*=.*path[[:space:]]*=[[:space:]]*"\.\./mold-inference".*package[[:space:]]*=[[:space:]]*"mold-ai-inference".*features[[:space:]]*=[[:space:]]*\["h3-private-uat"\]' \
  <<<"$server_dev_dependencies"; then
  fail "mold-server tests do not unify the private H3 loader for bridge verification"
fi
server_dependencies=$(sed -n '/^\[dependencies\]/,/^\[/p' crates/mold-server/Cargo.toml)
if grep -Eq '^mold-inference[[:space:]]*=.*h3-private-uat' <<<"$server_dependencies"; then
  fail "mold-ai-server activates the private H3 UAT path through an ordinary dependency"
fi
server_features=$(sed -n '/^\[features\]/,/^\[/p' crates/mold-server/Cargo.toml)
if ! grep -Fq 'h3-private-bridge = []' <<<"$server_features"; then
  fail "mold-ai-server does not expose the payload-free private H3 bridge seam"
fi
h3_uat_feature=$(sed -n '/^h3-private-uat = \[/,/^\]/p' crates/mold-server/Cargo.toml)
for edge in \
  h3-private-bridge \
  mold-core/h3-private-uat \
  mold-inference/h3-private-uat \
  mold-inference/dev-bins \
  mold-inference/h3-attention-rc \
  mp4 \
  cuda; do
  grep -Fq "\"${edge}\"" <<<"$h3_uat_feature" \
    || fail "mold-ai-server private H3 UAT feature omits ${edge}"
done
if [[ $(grep -Ec '^[[:space:]]*"[^"]+",?$' <<<"$h3_uat_feature") -ne 7 ]]; then
  fail "mold-ai-server private H3 UAT feature contains an unexpected runtime edge"
fi

release_feature_sources="$({
  grep -E -- '--features|releaseFeatures[[:space:]]*=|buildFeatures[[:space:]]*=' \
    .github/workflows/release.yml \
    .github/workflows/desktop.yml \
    Dockerfile \
    flake.nix \
    packaging/aur/mold-ai/PKGBUILD \
    packaging/aur/mold-ai-git/PKGBUILD
  sed -n '/^[[:space:]]*releaseFeatures =/,/^[[:space:]]*completionFeatures =/p' flake.nix
} || true)"
if LC_ALL=C tr -cs '[:alnum:]_-' '\n' <<<"$release_feature_sources" \
  | grep -Fxq h3-private-uat; then
  fail "a published release feature set compiles the private H3 UAT path"
fi

require_text scripts/verify-h3-release-exclusion.sh \
  "private_uat_marker='mold.minimax-h3.private-uat-artifact-reader.v1'" \
  "published binary verification does not reject the private H3 UAT marker"
require_text scripts/verify-h3-release-exclusion.sh \
  "private_qwen_support_marker='mold.minimax-h3.private-uat-qwen-support-loader.v1'" \
  "published binary verification does not reject the private H3 Qwen support marker"
require_text scripts/verify-h3-release-exclusion.sh \
  "private_qwen_capture_marker='mold.minimax-h3.private-uat-exact-bf16-qwen-layer50-capture.v1'" \
  "published binary verification does not reject the exact-BF16 Qwen capture marker"
require_text scripts/verify-h3-release-exclusion.sh \
  "private_visual_vae_capture_marker='mold.minimax-h3.private-uat-visual-vae-f32-fp16-capture.v1'" \
  "published binary verification does not reject the visual-VAE capture marker"
require_text scripts/verify-h3-release-exclusion.sh \
  "private_audio_capture_marker='mold.minimax-h3.private-uat-exact-fp32-audio-vae-capture.v1'" \
  "published binary verification does not reject the exact-FP32 AudioVAE capture marker"
require_text scripts/verify-h3-release-exclusion.sh \
  "private_transformer_capture_marker='mold.minimax-h3.private-uat-transformer-capture.v1'" \
  "published binary verification does not reject the transformer capture marker"
require_text .github/workflows/ci.yml \
  'cargo clippy -p mold-ai-inference --features dev-bins,h3-private-uat --bin h3_artifact_qualification -- -D warnings' \
  "CI does not compile the authorization-bound artifact qualifier"
require_text .github/workflows/ci.yml \
  'python3 scripts/tests/minimax-h3-qwen-layer50-capture-contract.py' \
  "CI does not run the exact-BF16 Qwen capture contract"
require_text .github/workflows/ci.yml \
  'python3 scripts/tests/minimax-h3-visual-vae-capture-contract.py' \
  "CI does not run the visual-VAE capture contract"
require_text .github/workflows/ci.yml \
  'python3 scripts/tests/minimax-h3-audio-vae-capture-contract.py' \
  "CI does not run the exact-FP32 AudioVAE capture contract"
require_text .github/workflows/ci.yml \
  'python3 scripts/tests/minimax-h3-transformer-capture-contract.py' \
  "CI does not run the paired transformer capture contract"
require_text .github/workflows/ci.yml \
  'cargo test -p mold-ai-inference --lib --features h3-private-uat minimax_h3' \
  "CI does not execute the private H3 foundation tests"
require_text .github/workflows/ci.yml \
  "'crates/mold-inference/src/minimax_h3/private_*.rs'" \
  "private H3 runtime source changes do not trigger release exclusion checks"
require_text .github/workflows/ci.yml \
  "'crates/mold-inference/src/minimax_h3/mod.rs'" \
  "private H3 module-gating changes do not trigger release exclusion checks"
require_text .github/workflows/ci.yml \
  'run: bash scripts/tests/minimax-h3-private-uat-release-contract.sh' \
  "release CI does not run the private H3 exclusion contract"
require_text docs/qualification/minimax-h3.md \
  'h3_artifact_qualification' \
  "the private artifact qualifier has no operator runbook"
require_text docs/qualification/minimax-h3.md \
  '--authorization-record' \
  "the private artifact qualifier runbook omits its external authorization record"
require_text docs/qualification/minimax-h3-conformance.md \
  'capture-minimax-h3-qwen-layer50.py' \
  "the exact-BF16 Qwen capture adapter has no operator runbook"
require_text docs/qualification/minimax-h3-conformance.md \
  'capture-minimax-h3-visual-vae.py' \
  "the visual-VAE capture adapter has no operator runbook"
require_text docs/qualification/minimax-h3-conformance.md \
  'capture-minimax-h3-audio-vae.py' \
  "the exact-FP32 AudioVAE capture adapter has no operator runbook"
require_text docs/qualification/minimax-h3-conformance.md \
  'capture-minimax-h3-transformer.py' \
  "the paired transformer capture producer has no operator runbook"

# Ref2VA may share the one-shot owner protocol, but it must retain a distinct
# runtime qualification, ordered-reference authority, and publication fence.
# Until a reviewed campaign lands, production ingress remains closed.
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'pub const fn reviewed_h3_private_runtime_available_for_task(task: Task) -> bool {' \
  "private H3 runtime availability is not scoped to an exact task"
require_text crates/mold-inference/src/minimax_h3/private_server.rs \
  'Task::Ref2va => false,' \
  "Ref2VA can inherit authority without its own reviewed runtime qualification"
require_text crates/mold-server/src/h3_private_bridge.rs \
  'b"ordered-heterogeneous-references"' \
  "the private ingress partition does not bind ordered Ref2VA conditioning"
require_text crates/mold-server/src/h3_private_bridge.rs \
  'pub(crate) reference_fingerprint_sha256: Option<String>' \
  "the one-shot owner contract omits target-duration reference order identity"
require_text crates/mold-server/src/h3_private_bridge.rs \
  'pub(crate) resolved_reference_fingerprint_sha256: Option<String>' \
  "the one-shot owner contract omits resolved reference artifact identity"
require_text crates/mold-server/src/gpu_worker.rs \
  'durable_reference_fingerprint.as_deref()' \
  "terminal publication does not cross-check durable Ref2VA order metadata"
require_text crates/mold-server/src/gpu_worker.rs \
  '== contract.reference_fingerprint_sha256.as_deref()' \
  "terminal publication does not bind durable Ref2VA order metadata to the prepared contract"
require_text crates/mold-server/src/h3_attempt.rs \
  'mold_core::minimax_h3::Task::Ref2va => {' \
  "the one-shot attempt fence does not validate the Ref2VA task partition"

scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT
ordinary_marker='mold.minimax-h3.attention-release-provenance.v2:h3-rc=omitted:global-flash=omitted'
private_marker='mold.minimax-h3.private-uat-artifact-reader.v1'
private_qwen_support_marker='mold.minimax-h3.private-uat-qwen-support-loader.v1'
private_runtime_record_marker='mold.minimax-h3.private-runtime-record-producer.v1'
private_qwen_capture_marker='mold.minimax-h3.private-uat-exact-bf16-qwen-layer50-capture.v1'
private_visual_vae_capture_marker='mold.minimax-h3.private-uat-visual-vae-f32-fp16-capture.v1'
private_audio_capture_marker='mold.minimax-h3.private-uat-exact-fp32-audio-vae-capture.v1'
private_transformer_capture_marker='mold.minimax-h3.private-uat-transformer-capture.v1'
printf '%s\n' "$ordinary_marker" >"$scratch_dir/ordinary"
scripts/verify-h3-release-exclusion.sh "$scratch_dir/ordinary" >/dev/null
printf '%s\n%s\n' "$ordinary_marker" "$private_marker" >"$scratch_dir/private"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the private H3 artifact reader"
fi
printf '%s\n%s\n' "$ordinary_marker" "$private_qwen_support_marker" \
  >"$scratch_dir/private-qwen-support"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private-qwen-support" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the private H3 Qwen support loader"
fi
printf '%s\n%s\n' "$ordinary_marker" "$private_runtime_record_marker" \
  >"$scratch_dir/private-runtime-record"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private-runtime-record" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the private H3 runtime-record producer"
fi
printf '%s\n%s\n' "$ordinary_marker" "$private_qwen_capture_marker" \
  >"$scratch_dir/private-qwen-capture"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private-qwen-capture" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the exact-BF16 Qwen capture adapter"
fi
printf '%s\n%s\n' "$ordinary_marker" "$private_visual_vae_capture_marker" \
  >"$scratch_dir/private-visual-vae-capture"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private-visual-vae-capture" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the visual-VAE capture adapter"
fi
printf '%s\n%s\n' "$ordinary_marker" "$private_audio_capture_marker" \
  >"$scratch_dir/private-audio-capture"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private-audio-capture" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the exact-FP32 AudioVAE capture adapter"
fi
printf '%s\n%s\n' "$ordinary_marker" "$private_transformer_capture_marker" \
  >"$scratch_dir/private-transformer-capture"
if scripts/verify-h3-release-exclusion.sh "$scratch_dir/private-transformer-capture" >/dev/null 2>&1; then
  fail "release exclusion verifier accepted the transformer capture adapter"
fi

echo "PASS: MiniMax H3 private-UAT release contract"
