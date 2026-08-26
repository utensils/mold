#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mold_home="${MOLD_HOME:-/Volumes/ExternalStorage/mold2}"
references_root="${LTX25_REFERENCES_ROOT:-$repo_root/tmp}"
verification_root="${LTX25_VERIFICATION_ROOT:-$mold_home/output/verification/ltx-2.5}"
audio_video="${LTX25_AUDIO_VIDEO:-$mold_home/output/ltx25-final-int8-metal-audio-seed-25026.mp4}"
silent_video="${LTX25_SILENT_VIDEO:-$verification_root/phase2-int8-metal-smoke-seed-25025.apng}"
timestamp="${LTX25_CAPTURE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
report="${LTX25_REPORT:-$verification_root/ltx25-metal-int8-verification-$timestamp.json}"
skip_gates="${LTX25_SKIP_GATES:-0}"
contract_test="${LTX25_CONTRACT_TEST:-0}"
database="${MOLD_DB_PATH:-$mold_home/mold.db}"
comfy_manifest="${LTX25_COMFY_MANIFEST:-}"

fail() {
  echo "LTX-2.5 Metal verification failed: $*" >&2
  exit 1
}

for command in ffprobe git jq shasum sqlite3; do
  command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
done

if [[ "$contract_test" == 1 ]]; then
  [[ "${LTX25_ALLOW_TEST_HOME:-0}" == 1 && "$skip_gates" == 1 ]] \
    || fail "contract test mode requires isolated home and skipped gates"
else
  [[ "${LTX25_ALLOW_TEST_HOME:-0}" != 1 && "$skip_gates" != 1 ]] \
    || fail "test overrides require LTX25_CONTRACT_TEST=1"
  [[ -z "${LTX25_HOST_JSON:-}" ]] \
    || fail "injected host JSON requires LTX25_CONTRACT_TEST=1"
  [[ "$mold_home" == /Volumes/ExternalStorage/mold2 ]] \
    || fail "MOLD_HOME must be /Volumes/ExternalStorage/mold2"
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] \
    || fail "runtime capture is restricted to Apple Silicon Metal"
  [[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=normal)" ]] \
    || fail "qualification requires a clean source tree; commit the exact code before capture"
fi

mkdir -p "$verification_root" "$(dirname "$report")"
evidence_dir="${report%.json}.d"
mkdir -p "$evidence_dir"

file_size() {
  stat -c '%s' "$1" 2>/dev/null || stat -f '%z' "$1"
}

file_sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

assets='[]'
record_asset() {
  local role="$1" relative="$2" expected_sha="$3"
  local path
  path="$mold_home/$relative"
  local marker="$path.sha256-verified"
  [[ -s "$path" ]] || fail "missing $role asset: $path"
  [[ -f "$marker" ]] || fail "missing verified SHA marker: $marker"
  local marker_sha
  marker_sha="$(tr -d '[:space:]' <"$marker")"
  [[ "$marker_sha" == "$expected_sha" ]] \
    || fail "$role marker mismatch: expected $expected_sha, found $marker_sha"
  local actual_sha=""
  local identity_proof="contract test: current bytes not qualified"
  if [[ "$contract_test" != 1 ]]; then
    actual_sha="$(file_sha256 "$path")"
    [[ "$actual_sha" == "$expected_sha" ]] \
      || fail "$role byte hash mismatch: expected $expected_sha, found $actual_sha"
    identity_proof="current bytes rehashed and matched manifest"
  fi
  assets="$(jq -c \
    --arg role "$role" --arg path "$path" --arg expected_sha "$expected_sha" \
    --arg actual_sha "$actual_sha" --arg identity_proof "$identity_proof" \
    --argjson bytes "$(file_size "$path")" \
    '. + [{role: $role, path: $path, bytes: $bytes,
      expected_sha256: $expected_sha,
      actual_sha256: (if $actual_sha == "" then null else $actual_sha end),
      identity_proof: $identity_proof}]' <<<"$assets")"
}

record_asset transformer \
  models/ltx-2.5-22b-distilled-int8-conv/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  c4279eeff115cbeaca494bd2183e7d768c38fe85a184dc6afbb7159157c44334
record_asset gemma4 \
  models/shared/ltx2/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f
record_asset video_vae_conv \
  models/shared/ltx2/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d
record_asset audio_vae \
  models/shared/ltx2/vae/ltx-2.5-audio-vae-bf16.safetensors \
  c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5
record_asset duration_head \
  models/shared/ltx2/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  2ec71e4206ed365d015f00c05a48caccfb0ee862986809d06ae376c09f5d9190
record_asset spatial_upscaler \
  models/shared/ltx2/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8
record_asset temporal_upscaler \
  models/shared/ltx2/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors \
  2bc3300f2b3c3c1834d72164fbf13a3b9fd73e5a741e8a2c3f4035f89a75c3fe

references='[]'
record_reference() {
  local name="$1" expected="$2" path
  path="$references_root/$1"
  [[ -d "$path/.git" ]] || fail "missing pinned reference clone: $path"
  local actual
  actual="$(git -C "$path" rev-parse HEAD)"
  [[ "$actual" == "$expected" ]] \
    || fail "$name is at $actual, expected $expected"
  [[ -z "$(git -C "$path" status --porcelain)" ]] \
    || fail "$name reference clone has uncommitted or untracked changes"
  references="$(jq -c --arg name "$name" --arg path "$path" --arg commit "$actual" \
    '. + [{name: $name, path: $path, commit: $commit, status: "pinned_clean"}]' \
    <<<"$references")"
}

record_reference ltx-2-upstream 400fd31054597515f47125691032c04b1c3ee24e
record_reference comfyui-ltxvideo-upstream 15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d
record_reference comfyui-upstream a1079ba16f2674734b065eb036fbfdddaa321a4d
record_reference diffusers-upstream 95c0d467cc2a4770b71fa25a117320377e6eb08f

media='[]'
record_media() {
  local label="$1" path="$2" seed="$3" prompt="$4" expects_audio="$5"
  [[ -s "$path" ]] || fail "missing retained media: $path"
  local probe="$evidence_dir/$label.ffprobe.json"
  ffprobe -v error -count_frames -show_entries \
    format=filename,size,duration:stream=index,codec_name,profile,codec_type,width,height,pix_fmt,r_frame_rate,nb_frames,nb_read_frames,sample_rate,channels \
    -of json "$path" >"$probe"
  jq -e '.streams[] | select(.codec_type == "video" and .width == 256 and .height == 256)' \
    "$probe" >/dev/null || fail "$label is not decoded 256x256 video"
  jq -e '.streams[] | select(.codec_type == "video" and .r_frame_rate == "24/1"
    and ((.nb_frames // .nb_read_frames) | tonumber) == 9)' "$probe" >/dev/null \
    || fail "$label is not exactly 9 frames at 24 fps"
  if [[ "$expects_audio" == true ]]; then
    jq -e '.streams[] | select(.codec_type == "audio" and .codec_name == "aac"
      and .sample_rate == "48000" and .channels == 2)' "$probe" >/dev/null \
      || fail "$label is missing stereo 48 kHz AAC"
  else
    jq -e '[.streams[] | select(.codec_type == "audio")] | length == 0' "$probe" >/dev/null \
      || fail "$label unexpectedly contains audio"
  fi
  local filename output_dir escaped_filename rows row
  filename="$(basename "$path")"
  output_dir="$(dirname "$path")"
  escaped_filename="${filename//\'/\'\'}"
  rows="$(sqlite3 -json "$database" \
    "SELECT id, filename, output_dir, format, title, prompt, model, seed, steps,
      guidance, width, height, frames, fps, generation_time_ms, backend, hostname,
      source, metadata_synthetic, file_size_bytes, metadata_json
     FROM generations WHERE filename = '$escaped_filename'")"
  row="$(jq -c --arg output_dir "$output_dir" \
    '[.[] | select(.output_dir == $output_dir)] | if length == 1 then .[0] else empty end' \
    <<<"$rows")"
  [[ -n "$row" ]] || fail "$label has no unique matching generation database row"
  jq -e --arg prompt "$prompt" --arg model "ltx-2.5-22b-distilled:int8-conv" \
    --arg format "${path##*.}" --argjson seed "$seed" --argjson audio "$expects_audio" \
    --argjson bytes "$(file_size "$path")" '
      .prompt == $prompt and .model == $model and .seed == $seed
      and .width == 256 and .height == 256 and .steps == 1 and .guidance == 1
      and .frames == 9 and .fps == 24 and .format == $format
      and .backend == "metal" and .source == "cli" and .metadata_synthetic == 0
      and .file_size_bytes == $bytes
      and ((.metadata_json | fromjson).enable_audio == $audio)
      and ((.metadata_json | fromjson).output_format == $format)
    ' <<<"$row" >/dev/null || fail "$label generation database provenance mismatch"
  media="$(jq -c \
    --arg label "$label" --arg path "$path" --arg sha "$(file_sha256 "$path")" \
    --arg prompt "$prompt" --argjson seed "$seed" --argjson audio "$expects_audio" \
    --argjson generation "$row" --slurpfile probe "$probe" \
    '. + [{label: $label, path: $path, sha256: $sha, prompt: $prompt, seed: $seed,
      model: "ltx-2.5-22b-distilled:int8-conv", settings: {
        width: 256, height: 256, frames: 9, fps: 24, steps: 1, guidance: 1,
        audio: $audio}, generation: $generation, ffprobe: $probe[0],
      retained_in_library: true}]' <<<"$media")"
}

record_media audio_video "$audio_video" 25026 \
  'A small brass automaton drummer performing in gentle rain, locked camera, cinematic reflections' true
record_media silent_video "$silent_video" 25025 \
  'A red fox walking through sunlit desert grass, cinematic natural motion' false

if [[ -z "$comfy_manifest" && "$contract_test" != 1 ]]; then
  shopt -s nullglob
  comfy_manifests=("$verification_root"/comfyui/reference-*/manifest.json)
  shopt -u nullglob
  for candidate in "${comfy_manifests[@]}"; do
    if [[ -z "$comfy_manifest" || "$candidate" -nt "$comfy_manifest" ]]; then
      comfy_manifest="$candidate"
    fi
  done
fi
[[ -n "$comfy_manifest" && -s "$comfy_manifest" ]] \
  || fail "missing retained ComfyUI Metal reference manifest; run capture-ltx25-comfy-metal-reference.sh"
comfy_graph="$(jq -er '.graph.path' "$comfy_manifest")"
[[ -s "$comfy_graph" ]] || fail "missing retained ComfyUI graph: $comfy_graph"
comfy_status="$(jq -er '.status' "$comfy_manifest")"
if [[ "$contract_test" != 1 ]]; then
  [[ "$(file_sha256 "$comfy_graph")" == "$(jq -er '.graph.sha256' "$comfy_manifest")" ]] \
    || fail "ComfyUI retained graph hash does not match its manifest"
fi
jq -e '
  .schema_version == "mold.ltx25.comfy-metal-reference.v1"
  and (.status == "passed" or .status == "operator_deferred")
  and .implementation == "ComfyUI" and .backend == "MPS"
  and .checkpoint == "distilled INT8 ConvRot"
  and .settings == {width:256,height:256,frames:9,fps:24,stage1_seed:25026,
    stage2_seed:42,video_cfg:1,audio_cfg:1}
  and (if .status == "passed" then
    .retained_in_library == true
    and (.video.ffprobe.streams[] | select(.codec_type == "video" and .width == 256
      and .height == 256 and .r_frame_rate == "24/1"
      and ((.nb_frames // .nb_read_frames) | tonumber) == 9))
    and (.video.ffprobe.streams[] | select(.codec_type == "audio"
      and .sample_rate == "48000" and .channels == 2))
  else
    .video == null and .retained_in_library == false
    and .preservation.downloaded_models_deleted == false
    and .preservation.rendered_media_deleted == false
  end)
' "$comfy_manifest" >/dev/null || fail "ComfyUI reference manifest contract mismatch"
if [[ "$comfy_status" == passed ]]; then
  comfy_video="$(jq -er '.video.path' "$comfy_manifest")"
  [[ -s "$comfy_video" ]] || fail "missing retained ComfyUI video: $comfy_video"
  if [[ "$contract_test" != 1 ]]; then
    [[ "$(file_sha256 "$comfy_video")" == "$(jq -er '.video.sha256' "$comfy_manifest")" ]] \
      || fail "ComfyUI retained video hash does not match its manifest"
  fi
else
  guard_marker="$(jq -er '.deferred.resource_guard_marker' "$comfy_manifest")"
  [[ -s "$guard_marker" ]] || fail "missing retained ComfyUI resource-guard evidence: $guard_marker"
  server_log="$(jq -er '.server_log_path' "$comfy_manifest")"
  [[ -f "$server_log" ]] || fail "missing retained ComfyUI server log: $server_log"
  attestation='null'
  if jq -e '.deferred.guard_cause and .deferred.resource_guard_marker_sha256
    and .server_log_sha256' "$comfy_manifest" >/dev/null; then
    guard_cause="$(jq -er '.deferred.guard_cause' "$comfy_manifest")"
    marker_sha="$(jq -er '.deferred.resource_guard_marker_sha256' "$comfy_manifest")"
    log_sha="$(jq -er '.server_log_sha256' "$comfy_manifest")"
    blocking_operator="$(jq -r '.deferred.blocking_operator // empty' "$comfy_manifest")"
    upstream_progress="$(jq -r '.deferred.upstream_progress // empty' "$comfy_manifest")"
    [[ "$(jq -er '.cause' "$guard_marker")" == "$guard_cause" ]] \
      || fail "ComfyUI guard cause does not match its retained marker"
  else
    attestation_path="$(dirname "$comfy_manifest")/resource-guard-attestation.json"
    [[ -s "$attestation_path" ]] \
      || fail "legacy ComfyUI evidence requires a separate resource-guard attestation"
    jq -e '
      .schema_version == "mold.ltx25.comfy-metal-legacy-attestation.v1"
      and (.guard.cause | IN("pressure_unreadable","memory_pressure","server_rss","timeout"))
      and (.resource_guard_marker_sha256 | test("^[0-9a-f]{64}$"))
      and (.server_log_sha256 | test("^[0-9a-f]{64}$"))
      and .preservation.source_evidence_restored_to_original_content == true
    ' "$attestation_path" >/dev/null || fail "legacy ComfyUI attestation contract mismatch"
    [[ "$(file_sha256 "$comfy_manifest")" == "$(jq -er '.source_manifest_sha256' "$attestation_path")" ]] \
      || fail "legacy ComfyUI manifest hash does not match its attestation"
    guard_cause="$(jq -er '.guard.cause' "$attestation_path")"
    marker_sha="$(jq -er '.resource_guard_marker_sha256' "$attestation_path")"
    log_sha="$(jq -er '.server_log_sha256' "$attestation_path")"
    blocking_operator="$(jq -r '.observed_log_evidence.blocking_operator // empty' "$attestation_path")"
    upstream_progress="$(jq -r '.observed_log_evidence.upstream_progress // empty' "$attestation_path")"
    attestation="$(jq -c --arg path "$attestation_path" '. + {path:$path}' "$attestation_path")"
  fi
  [[ "$guard_cause" =~ ^(pressure_unreadable|memory_pressure|server_rss|timeout)$ ]] \
    || fail "unknown ComfyUI guard cause: $guard_cause"
  [[ "$(file_sha256 "$guard_marker")" == "$marker_sha" ]] \
    || fail "ComfyUI resource-guard marker hash does not match its evidence seal"
  [[ "$(file_sha256 "$server_log")" == "$log_sha" ]] \
    || fail "ComfyUI server log hash does not match its evidence seal"
  if [[ "$blocking_operator" == "aten::_int_mm fell back from MPS to CPU" ]]; then
    if ! grep -Fq "aten::_int_mm" "$server_log" \
      || ! grep -Eq 'not currently (implemented for the MPS device|supported on the MPS backend)' "$server_log"; then
      fail "ComfyUI blocking operator is not present in its retained log"
    fi
  fi
  if [[ "$upstream_progress" == "official sampler reached 0/8 after model load" ]]; then
    grep -Eq '0%.*0/8' "$server_log" \
      || fail "ComfyUI sampler progress is not present in its retained log"
  fi
fi
comfy_reference="$(jq -c --arg manifest_path "$comfy_manifest" --argjson attestation "${attestation:-null}" \
  '. + {manifest_path:$manifest_path}
    + (if $attestation == null then {} else {verification_attestation:$attestation} end)' \
  "$comfy_manifest")"

gates='[]'
run_gate() {
  local label="$1"
  shift
  local log="$evidence_dir/$label.log" started ended status=passed exit_code=0
  started="$(date +%s)"
  if [[ "$skip_gates" == 1 ]]; then
    status=skipped_contract_test
    : >"$log"
  else
    set +e
    (cd "$repo_root" && "$@") >"$log" 2>&1
    exit_code=$?
    set -e
    [[ "$exit_code" -eq 0 ]] || status=failed
  fi
  ended="$(date +%s)"
  gates="$(jq -c --arg label "$label" --arg status "$status" \
    --arg command "$(printf '%q ' "$@")" --arg log "$log" \
    --arg log_sha "$(file_sha256 "$log")" --argjson exit_code "$exit_code" \
    --argjson seconds "$((ended - started))" \
    '. + [{label: $label, status: $status, exit_code: $exit_code, seconds: $seconds,
      command: $command, log_path: $log, log_sha256: $log_sha}]' <<<"$gates")"
  [[ "$status" != failed ]] || fail "$label failed; see $log"
}

run_gate core_ltx25_manifest nix develop -c cargo test -p mold-ai-core ltx25_manifest
run_gate inference_ltx25_oracles nix develop -c cargo test -p mold-ai-inference ltx25
run_gate inference_ltx2_regression nix develop -c cargo test -p mold-ai-inference ltx2

host='{}'
if [[ -n "${LTX25_HOST_JSON:-}" ]]; then
  host="$LTX25_HOST_JSON"
else
  displays="$(system_profiler SPDisplaysDataType -json)"
  host="$(jq -c --arg os "$(sw_vers -productVersion)" --arg arch "$(uname -m)" \
    '{os: "macOS", os_version: $os, arch: $arch,
      metal_devices: [.SPDisplaysDataType[] | {name: .sppci_model,
        metal_support: .spdisplays_metal, vram: .spdisplays_vram_shared}]}' <<<"$displays")"
fi

source_commit="$(git -C "$repo_root" rev-parse HEAD)"
qualification_status=passed
source_tree_state=clean
if [[ "$contract_test" == 1 ]]; then
  qualification_status=not_qualified_contract_test
  source_tree_state=contract_test
fi
comfy_evidence="retained exact-weight API graph, history, decoded audio-video, and manifest"
if [[ "$comfy_status" == operator_deferred ]]; then
  comfy_evidence="resource guard cause: $guard_cause"
  [[ -z "$upstream_progress" ]] || comfy_evidence="$comfy_evidence; $upstream_progress"
  [[ -z "$blocking_operator" ]] || comfy_evidence="$comfy_evidence; $blocking_operator"
fi
tmp_report="$report.tmp.$$"
jq -n \
  --arg captured_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_commit "$source_commit" --arg mold_home "$mold_home" \
  --arg source_tree_state "$source_tree_state" --arg comfy_evidence "$comfy_evidence" \
  --arg qualification_status "$qualification_status" --arg comfy_status "$comfy_status" \
  --argjson host "$host" --argjson assets "$assets" --argjson references "$references" \
  --argjson media "$media" --argjson gates "$gates" --argjson comfy_reference "$comfy_reference" \
  '{schema_version: "mold.ltx25.metal-int8.verification.v1", captured_at: $captured_at,
    source_commit: $source_commit, source_tree_state: $source_tree_state,
    mold_home: $mold_home, backend_scope: "metal",
    default_model: "ltx-2.5-22b-distilled:int8-conv", host: $host,
    assets: $assets, references: $references, gates: $gates, media: $media,
    comfy_reference: $comfy_reference,
    comparison_matrix: [
      {implementation: "Mold", checkpoint: "distilled INT8 ConvRot", backend: "Metal",
        status: $qualification_status,
        evidence: "database-bound retained decoded audio and silent media"},
      {implementation: "ComfyUI", checkpoint: "distilled INT8 ConvRot", backend: "MPS",
        status: (if $qualification_status == "not_qualified_contract_test" then
          $qualification_status else $comfy_status end),
        evidence: $comfy_evidence},
      {implementation: "official PyTorch", checkpoint: "BF16", backend: "Metal",
        status: "static_oracle_only", evidence: "compact Comfy INT8 is not directly executable"},
      {implementation: "Diffusers", checkpoint: "BF16", backend: "Metal",
        status: "static_oracle_only", evidence: "compact Comfy INT8 is not directly executable"},
      {implementation: "Mold", checkpoint: "BF16", backend: "Metal",
        status: "operator_deferred", evidence: "runtime stopped; downloaded assets retained"},
      {implementation: "Mold", checkpoint: "all", backend: "CUDA",
        status: "separate_host_deferred", evidence: "outside this Apple Metal qualification"}
    ], preservation: {downloaded_models_deleted: false, rendered_media_deleted: false}}' \
  >"$tmp_report"
mv "$tmp_report" "$report"
echo "$report"
