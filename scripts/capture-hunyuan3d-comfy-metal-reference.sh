#!/usr/bin/env bash
set -euo pipefail

# ComfyUI Metal (MPS) reference capture for the Hunyuan3D image-to-3D family.
#
# Renders the SAME checkpoint, the SAME source image and the SAME seed that
# `capture-hunyuan3d-metal-uat.sh` hands to mold, once per octree rung, and
# retains every byte it produced under the evidence directory. Nothing is ever
# deleted: the graph, the queue response, the history, the server log, the
# produced `.glb` files and their SHA-256 sidecars all stay.
#
# The graph itself is `scripts/fixtures/hunyuan3d-comfy-metal-api-prompt.json`,
# a valid API-format prompt carrying default values that this runner overrides
# per rung (checkpoint name, input image filename, seed, octree resolution,
# filename prefix). Node ids and input names come from the ComfyUI source:
#   comfy_extras/nodes_video_model.py    ImageOnlyCheckpointLoader
#   nodes.py                             LoadImage, CLIPVisionEncode, KSampler
#   comfy_extras/nodes_model_advanced.py ModelSamplingAuraFlow
#   comfy_extras/nodes_hunyuan3d.py      EmptyLatentHunyuan3Dv2,
#                                        Hunyuan3Dv2Conditioning,
#                                        VAEDecodeHunyuan3D, VoxelToMesh
#   comfy_extras/nodes_save_3d.py        SaveGLB
#
# The oracle clone defaults to `$repo_root/tmp/ComfyUI` — the repo convention
# of a gitignored `tmp/` per checkout — and is PINNED to
# `HUNYUAN3D_COMFY_COMMIT`, because a ComfyUI that has moved is a different
# reference and its meshes are not comparable with the retained evidence. A
# git worktree has its own empty `tmp/`, so run this from the main checkout or
# point `HUNYUAN3D_COMFY_ROOT` at an existing clone (or give the worktree one).
#
# The source image is PRE-FRAMED first, by `scripts/hunyuan3d-frame-source.py`,
# and the framed copy is what ComfyUI is given. mold applies Tencent's
# `recenter` letterbox to a raw cutout — the alpha bounding box rescaled to
# fill 85 % of a square — while ComfyUI's `CLIPVisionEncode` with `crop:
# center` just centre-crops the picture as handed over. Comparing the two
# without framing therefore measures the conditioning policy, not the port:
# the 2026-09-01 Metal captures of the same armchair scored a normalised
# Chamfer of 0.030 with the raw cutout and 0.0103 once both sides were fed the
# same pre-framed picture. After framing, mold's own letterbox is (very
# nearly) the identity and ComfyUI's centre crop of a square is a no-op, so
# what is left to measure is the networks. `HUNYUAN3D_FRAME_SOURCE=0` bypasses
# it for a deliberate raw-cutout capture.
#
# Long captures can outlive a transient `nix develop -c` environment. Prefer
# the user's GC-rooted profile and macOS system tools so Bash never caches a
# devshell shim whose target may be collected while ComfyUI is still running.
PATH="${HOME}/.nix-profile/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH}"
export PATH
hash -r

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mold_home="${MOLD_HOME:-/Volumes/ExternalStorage/mold2}"
comfy_root="${HUNYUAN3D_COMFY_ROOT:-$repo_root/tmp/ComfyUI}"
comfy_commit_expected="${HUNYUAN3D_COMFY_COMMIT:-7fe8a6138504f90ff7be82f3babf416da32876b1}"
comfy_allow_dirty="${HUNYUAN3D_COMFY_ALLOW_DIRTY:-0}"
python="${HUNYUAN3D_COMFY_PYTHON:-$mold_home/comfyui-venv/bin/python}"
graph="${HUNYUAN3D_COMFY_GRAPH:-$repo_root/scripts/fixtures/hunyuan3d-comfy-metal-api-prompt.json}"
model_dir="${HUNYUAN3D_MODEL_DIR:-$mold_home/models/hunyuan3d-mini-turbo-fp16}"
ckpt_file="${HUNYUAN3D_CKPT_FILE:-}"
source_image="${HUNYUAN3D_SOURCE_IMAGE:-}"
frame_source="${HUNYUAN3D_FRAME_SOURCE:-1}"
frame_script="${HUNYUAN3D_FRAME_SCRIPT:-$repo_root/scripts/hunyuan3d-frame-source.py}"
frame_python="${HUNYUAN3D_FRAME_PYTHON:-$mold_home/comfyui-venv/bin/python}"
octrees="${HUNYUAN3D_OCTREES:-192 256 320}"
seed="${HUNYUAN3D_SEED:-25026}"
steps="${HUNYUAN3D_STEPS:-5}"
cfg="${HUNYUAN3D_CFG:-1.0}"
shift_value="${HUNYUAN3D_SHIFT:-1.0}"
latent_resolution="${HUNYUAN3D_LATENT_RESOLUTION:-3072}"
num_chunks="${HUNYUAN3D_NUM_CHUNKS:-8000}"
threshold="${HUNYUAN3D_THRESHOLD:-0.6}"
verification_root="${HUNYUAN3D_VERIFICATION_ROOT:-$mold_home/output/verification/hunyuan3d}"
timestamp="${HUNYUAN3D_CAPTURE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
evidence_dir="${HUNYUAN3D_EVIDENCE_DIR:-$verification_root/$timestamp}"
port="${HUNYUAN3D_COMFY_PORT:-8188}"
rung_timeout="${HUNYUAN3D_COMFY_RUNG_TIMEOUT:-3600}"
base_url="http://127.0.0.1:$port"
manifest="$evidence_dir/comfy-manifest.json"
server_log="$evidence_dir/comfyui-server.log"
comfy_output_dir="$evidence_dir/comfyui-output"
comfy_input_dir="$evidence_dir/comfyui-input"
comfy_temp_dir="$evidence_dir/comfyui-temp"
comfy_user_dir="$evidence_dir/comfyui-user"
extra_paths="$evidence_dir/extra_model_paths.yaml"
server_pid=""

fail() {
  echo "Hunyuan3D ComfyUI Metal reference failed: $*" >&2
  exit 1
}

file_sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

# The graph is checked before anything is started so a typo in a node id or an
# input name is a fast failure rather than an hour of wasted GPU time. The
# expected values mirror the ComfyUI sources cited in the header comment.
validate_graph() {
  jq -e '
    length == 10
    and .["1"].class_type == "ImageOnlyCheckpointLoader"
    and (.["1"].inputs | keys == ["ckpt_name"])
    and .["2"].class_type == "LoadImage"
    and (.["2"].inputs | keys == ["image"])
    and .["3"].class_type == "CLIPVisionEncode"
    and .["3"].inputs.clip_vision == ["1", 1]
    and .["3"].inputs.image == ["2", 0]
    and .["3"].inputs.crop == "center"
    and .["4"].class_type == "Hunyuan3Dv2Conditioning"
    and .["4"].inputs.clip_vision_output == ["3", 0]
    and .["5"].class_type == "ModelSamplingAuraFlow"
    and .["5"].inputs.model == ["1", 0]
    and .["5"].inputs.shift == 1.0
    and .["6"].class_type == "EmptyLatentHunyuan3Dv2"
    and .["6"].inputs.resolution == 3072
    and .["6"].inputs.batch_size == 1
    and .["7"].class_type == "KSampler"
    and .["7"].inputs.model == ["5", 0]
    and .["7"].inputs.steps == 5
    and .["7"].inputs.cfg == 1.0
    and .["7"].inputs.sampler_name == "euler"
    and .["7"].inputs.scheduler == "normal"
    and .["7"].inputs.positive == ["4", 0]
    and .["7"].inputs.negative == ["4", 1]
    and .["7"].inputs.latent_image == ["6", 0]
    and .["7"].inputs.denoise == 1.0
    and .["8"].class_type == "VAEDecodeHunyuan3D"
    and .["8"].inputs.samples == ["7", 0]
    and .["8"].inputs.vae == ["1", 2]
    and .["8"].inputs.num_chunks == 8000
    and .["9"].class_type == "VoxelToMesh"
    and .["9"].inputs.voxel == ["8", 0]
    and .["9"].inputs.algorithm == "surface net"
    and .["9"].inputs.threshold == 0.6
    and .["10"].class_type == "SaveGLB"
    and .["10"].inputs.mesh == ["9", 0]
  ' "$graph" >/dev/null || fail "unsafe or unexpected API graph: $graph"
}

# Substitution is done with jq rather than sed so the result is JSON by
# construction and numeric inputs stay numbers.
render_graph() {
  local octree="$1" ckpt_name="$2" image_name="$3" prefix="$4"
  jq \
    --arg ckpt "$ckpt_name" --arg image "$image_name" --arg prefix "$prefix" \
    --argjson seed "$seed" --argjson steps "$steps" --argjson cfg "$cfg" \
    --argjson shift_value "$shift_value" --argjson resolution "$latent_resolution" \
    --argjson chunks "$num_chunks" --argjson threshold "$threshold" \
    --argjson octree "$octree" '
      .["1"].inputs.ckpt_name = $ckpt
      | .["2"].inputs.image = $image
      | .["5"].inputs.shift = $shift_value
      | .["6"].inputs.resolution = $resolution
      | .["7"].inputs.seed = $seed
      | .["7"].inputs.steps = $steps
      | .["7"].inputs.cfg = $cfg
      | .["8"].inputs.num_chunks = $chunks
      | .["8"].inputs.octree_resolution = $octree
      | .["9"].inputs.threshold = $threshold
      | .["10"].inputs.filename_prefix = $prefix
    ' "$graph"
}

# The oracle is PINNED, like the ltx25 reference runner's. A ComfyUI that has
# moved is a different reference, and a mesh captured against it is not
# comparable with the retained evidence — so a mismatch stops the capture
# rather than quietly re-baselining it. Sets `comfy_commit` and `comfy_dirty`.
check_comfy_pin() {
  comfy_commit="$( (cd "$comfy_root" && git rev-parse HEAD) )"
  [[ "$comfy_commit" == "$comfy_commit_expected" ]] || fail \
    "ComfyUI reference checkout is at $comfy_commit, expected $comfy_commit_expected; \
check out the pin or set HUNYUAN3D_COMFY_COMMIT to re-pin deliberately"
  comfy_dirty=false
  if [[ -n "$( (cd "$comfy_root" && git status --porcelain) )" ]]; then
    comfy_dirty=true
    [[ "$comfy_allow_dirty" == 1 ]] || fail \
      "ComfyUI reference checkout has uncommitted or untracked changes; \
commit or clean them, or set HUNYUAN3D_COMFY_ALLOW_DIRTY=1 to capture anyway"
  fi
}

validate_graph
# Exercises the pin gate alone, so the contract test can cover both directions
# without a checkpoint, a server, or an Apple Silicon host.
if [[ "${HUNYUAN3D_COMFY_TEST_PIN:-0}" == 1 ]]; then
  check_comfy_pin
  printf '%s %s\n' "$comfy_commit" "$comfy_dirty"
  exit 0
fi
if [[ "${HUNYUAN3D_COMFY_VALIDATE_ONLY:-0}" == 1 ]]; then
  echo "$graph"
  exit 0
fi
if [[ "${HUNYUAN3D_COMFY_RENDER_GRAPH_ONLY:-0}" == 1 ]]; then
  render_graph "${HUNYUAN3D_RENDER_OCTREE:-256}" \
    "${HUNYUAN3D_RENDER_CKPT:-model.fp16.safetensors}" \
    "${HUNYUAN3D_RENDER_IMAGE:-source.png}" \
    "${HUNYUAN3D_RENDER_PREFIX:-3d/hunyuan3d}"
  exit 0
fi

for command in curl git jq shasum; do
  command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
done
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] \
  || fail "runtime capture is restricted to Apple Silicon Metal"
[[ -n "$source_image" ]] || fail "HUNYUAN3D_SOURCE_IMAGE is required"
[[ -s "$source_image" ]] || fail "missing source image: $source_image"
[[ -x "$python" ]] || fail "missing retained ComfyUI Python environment: $python"
[[ -f "$comfy_root/main.py" ]] || fail "missing ComfyUI reference checkout: $comfy_root"
check_comfy_pin
[[ -d "$model_dir" ]] || fail "missing Hunyuan3D checkpoint directory: $model_dir"

# mold stores the checkpoint under the manifest's HF-relative path
# (`hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors`), and ComfyUI's
# `folder_paths` scans a checkpoints root recursively, so the discovered name
# is kept RELATIVE to the model directory and handed to the loader as-is.
if [[ -z "$ckpt_file" ]]; then
  candidates=()
  while IFS= read -r candidate; do
    candidates+=("${candidate#"$model_dir"/}")
  done < <(find "$model_dir" -type f -name '*.safetensors' | LC_ALL=C sort)
  (( ${#candidates[@]} == 1 )) \
    || fail "expected exactly one .safetensors under $model_dir, found ${#candidates[@]}; set HUNYUAN3D_CKPT_FILE"
  ckpt_file="${candidates[0]}"
fi
[[ -s "$model_dir/$ckpt_file" ]] || fail "missing checkpoint: $model_dir/$ckpt_file"

for octree in $octrees; do
  [[ "$octree" =~ ^[0-9]+$ ]] || fail "octree rungs must be integers, found: $octree"
done

curl --fail --silent --max-time 2 "$base_url/system_stats" >/dev/null 2>&1 \
  && fail "port $port already has a ComfyUI server; refusing to control an unrelated process"

mkdir -p "$evidence_dir" "$comfy_output_dir" "$comfy_input_dir" "$comfy_temp_dir" "$comfy_user_dir"

# The weights are never copied or symlinked into the ComfyUI tree: the server
# is pointed at the retained model directory instead.
cat >"$extra_paths" <<YAML
mold_hunyuan3d:
  base_path: $model_dir
  checkpoints: |
    $model_dir
YAML

# Pre-framing (see the header): both engines must be handed the SAME picture,
# or the comparison measures the two conditioning policies instead of the two
# networks. The framed copy is retained beside the original; neither is
# modified in place.
effective_source="$source_image"
framed_source=""
framed_sha256=""
if [[ "$frame_source" == 1 ]]; then
  [[ -f "$frame_script" ]] || fail "missing source framing script: $frame_script"
  [[ -x "$frame_python" ]] || command -v "$frame_python" >/dev/null 2>&1 \
    || fail "missing framing interpreter: $frame_python"
  framed_source="$evidence_dir/source-framed.png"
  "$frame_python" "$frame_script" --input "$source_image" --output "$framed_source" \
    >"$evidence_dir/source-framing.log" 2>&1 \
    || fail "source framing failed; see $evidence_dir/source-framing.log"
  [[ -s "$framed_source" ]] || fail "source framing produced nothing: $framed_source"
  framed_sha256="$(file_sha256 "$framed_source")"
  effective_source="$framed_source"
fi

image_extension=png
[[ "$(basename "$effective_source")" == *.* ]] && image_extension="${effective_source##*.}"
image_name="hunyuan3d-source.$image_extension"
cp "$effective_source" "$comfy_input_dir/$image_name"
image_sha256="$(file_sha256 "$source_image")"
encoded_sha256="$(file_sha256 "$comfy_input_dir/$image_name")"
cp "$graph" "$evidence_dir/comfy-graph-template.json"


python_version="$("$python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
torch_version="$("$python" -c 'import torch; print(torch.__version__)')"

cleanup() {
  local status=$?
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

(
  cd "$comfy_root"
  env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 "$python" main.py \
      --listen 127.0.0.1 --port "$port" \
      --extra-model-paths-config "$extra_paths" \
      --input-directory "$comfy_input_dir" \
      --output-directory "$comfy_output_dir" --temp-directory "$comfy_temp_dir" \
      --user-directory "$comfy_user_dir" --disable-auto-launch \
      --cache-none --preview-method none --disable-smart-memory --verbose INFO
) >"$server_log" 2>&1 &
server_pid=$!

for _ in $(seq 1 180); do
  if ! kill -0 "$server_pid" >/dev/null 2>&1; then
    fail "ComfyUI exited during startup; see $server_log"
  fi
  if curl --fail --silent --max-time 2 "$base_url/system_stats" >"$evidence_dir/system-stats.json"; then
    break
  fi
  sleep 1
done
[[ -s "$evidence_dir/system-stats.json" ]] || fail "ComfyUI did not start within 180 seconds"
jq -e '.devices[] | select((.type | ascii_downcase) == "mps")' \
  "$evidence_dir/system-stats.json" >/dev/null || fail "ComfyUI did not select MPS"
device="$(jq -r '[.devices[] | .type] | join(",")' "$evidence_dir/system-stats.json")"

runs='[]'
for octree in $octrees; do
  prefix="3d/hunyuan3d-comfy-mps-seed-$seed-oct-$octree"
  rung_graph="$evidence_dir/comfy-prompt-$octree.json"
  queue_response="$evidence_dir/comfy-queue-response-$octree.json"
  history_json="$evidence_dir/comfy-history-$octree.json"
  render_graph "$octree" "$ckpt_file" "$image_name" "$prefix" >"$rung_graph"

  client_id="mold-hunyuan3d-$timestamp-$octree"
  started="$(date +%s)"
  jq -n --slurpfile prompt "$rung_graph" --arg client_id "$client_id" \
    '{prompt:$prompt[0], client_id:$client_id}' \
    | curl --fail --silent --show-error -H 'Content-Type: application/json' \
        --data-binary @- "$base_url/prompt" >"$queue_response"
  prompt_id="$(jq -er '.prompt_id' "$queue_response")"

  settled=false
  for _ in $(seq 1 "$rung_timeout"); do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      fail "ComfyUI exited during inference at octree $octree; see $server_log"
    fi
    if curl --fail --silent --max-time 5 "$base_url/history/$prompt_id" >"$history_json" \
      && jq -e --arg id "$prompt_id" 'has($id)' "$history_json" >/dev/null; then
      settled=true
      break
    fi
    sleep 1
  done
  [[ "$settled" == true ]] \
    || fail "octree $octree did not settle within $rung_timeout seconds; see $server_log"
  jq -e --arg id "$prompt_id" '.[$id].status.status_str == "success"' \
    "$history_json" >/dev/null \
    || fail "ComfyUI job for octree $octree did not complete successfully; see $history_json"
  ended="$(date +%s)"

  # SaveGLB reports through `ui={"3d": [...]}`; older builds surfaced the same
  # rows under `images`, so both keys are read before giving up.
  relative_output="$(jq -er --arg id "$prompt_id" '
    [.[$id].outputs[]? | ((.["3d"] // [])[]), ((.images // [])[])]
    | map(select(.type == "output" and (.filename | endswith(".glb"))))
    | first | ((if (.subfolder // "") == "" then "" else .subfolder + "/" end) + .filename)
  ' "$history_json")"
  produced="$comfy_output_dir/$relative_output"
  [[ -s "$produced" ]] || fail "ComfyUI reported output is missing: $produced"
  retained="$evidence_dir/comfy-$octree.glb"
  cp "$produced" "$retained"
  retained_sha="$(file_sha256 "$retained")"
  printf '%s  %s\n' "$retained_sha" "$(basename "$retained")" >"$retained.sha256"

  runs="$(jq -c \
    --arg octree "$octree" --arg prompt_id "$prompt_id" --arg glb "$retained" \
    --arg glb_sha "$retained_sha" --arg produced "$produced" \
    --arg graph "$rung_graph" --arg graph_sha "$(file_sha256 "$rung_graph")" \
    --arg history "$history_json" --arg queue_response "$queue_response" \
    --argjson bytes "$(wc -c <"$retained" | tr -d ' ')" \
    --argjson wall_seconds "$((ended - started))" \
    '. + [{octree_resolution: ($octree | tonumber), prompt_id: $prompt_id,
      glb_path: $glb, glb_sha256: $glb_sha, glb_bytes: $bytes,
      comfyui_output_path: $produced, graph_path: $graph, graph_sha256: $graph_sha,
      history_path: $history, queue_response_path: $queue_response,
      wall_seconds: $wall_seconds}]' <<<"$runs")"
done

tmp_manifest="$manifest.tmp.$$"
jq -n \
  --arg captured_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg evidence_dir "$evidence_dir" --arg mold_home "$mold_home" \
  --arg comfy_root "$comfy_root" --arg comfy_commit "$comfy_commit" \
  --arg comfy_commit_expected "$comfy_commit_expected" \
  --arg python "$python" --arg python_version "$python_version" \
  --arg torch_version "$torch_version" --arg device "$device" \
  --arg checkpoint "$model_dir/$ckpt_file" --arg ckpt_name "$ckpt_file" \
  --arg source_image "$source_image" --arg image_sha256 "$image_sha256" \
  --arg framed_source "$framed_source" --arg framed_sha256 "$framed_sha256" \
  --arg encoded_sha256 "$encoded_sha256" --arg frame_script "$frame_script" \
  --argjson framed "$([[ "$frame_source" == 1 ]] && echo true || echo false)" \
  --arg server_log "$server_log" --arg server_log_sha256 "$(file_sha256 "$server_log")" \
  --arg extra_paths "$extra_paths" \
  --argjson comfy_dirty "$comfy_dirty" --argjson seed "$seed" \
  --argjson steps "$steps" --argjson cfg "$cfg" --argjson shift_value "$shift_value" \
  --argjson resolution "$latent_resolution" --argjson chunks "$num_chunks" \
  --argjson threshold "$threshold" --argjson runs "$runs" \
  --slurpfile system "$evidence_dir/system-stats.json" \
  '{schema_version: "mold.hunyuan3d.comfy-metal-reference.v1", status: "passed",
    captured_at: $captured_at, implementation: "ComfyUI", backend: "MPS",
    device: $device, evidence_dir: $evidence_dir, mold_home: $mold_home,
    comfyui: {root: $comfy_root, commit: $comfy_commit,
      commit_expected: $comfy_commit_expected, dirty: $comfy_dirty,
      python: $python, python_version: $python_version, torch_version: $torch_version,
      extra_model_paths: $extra_paths, server_log_path: $server_log,
      server_log_sha256: $server_log_sha256},
    checkpoint: {path: $checkpoint, ckpt_name: $ckpt_name},
    source_image: {path: $source_image, sha256: $image_sha256, framed: $framed,
      framing_script: (if $framed then $frame_script else null end),
      framed_path: (if $framed_source == "" then null else $framed_source end),
      framed_sha256: (if $framed_sha256 == "" then null else $framed_sha256 end),
      encoded_sha256: $encoded_sha256},
    settings: {seed: $seed, steps: $steps, cfg: $cfg, sampler_name: "euler",
      scheduler: "normal", model_sampling_shift: $shift_value,
      latent_resolution: $resolution, num_chunks: $chunks,
      voxel_algorithm: "surface net", threshold: $threshold},
    runs: $runs,
    preservation: {downloaded_models_deleted: false, rendered_media_deleted: false}}' \
  >"$tmp_manifest"
mv "$tmp_manifest" "$manifest"

echo "$manifest"
