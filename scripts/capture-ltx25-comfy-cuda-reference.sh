#!/usr/bin/env bash
set -euo pipefail

# ComfyUI CUDA oracle for LTX-2.5 (#1398 INT8 ConvRot, #1414 GGUF Q4_K_M).
#
# UAT only; nothing here ships. This is the Linux/CUDA sibling of
# capture-ltx25-comfy-metal-reference.sh: same 29-node API graph, same seeds,
# same 256x256x9 shape, run through the pinned upstream ComfyUI checkout so a
# Mold render can be compared against the official implementation on the
# same hardware. The resource guard reads `free -b` and `nvidia-smi` instead of the
# macOS pressure probe, and a torch build that cannot initialise CUDA is a
# deferral cause rather than a crash.

# Long captures can outlive a transient `nix develop -c` environment. Keep the
# system tool paths ahead of any devshell shim so Bash never caches a target
# that may be garbage-collected while ComfyUI is still running.
PATH="/run/current-system/sw/bin:/usr/bin:/bin:${PATH}"
export PATH
hash -r

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mold_home="${MOLD_HOME:-/mnt/storage20tb/AI/mold}"
models_dir="${MOLD_MODELS_DIR:-}"
comfy_root="${LTX25_COMFY_ROOT:-$repo_root/tmp/comfyui-upstream}"
python="${LTX25_COMFY_PYTHON:-$repo_root/tmp/comfyui-venv/bin/python}"
extra_paths="${LTX25_COMFY_EXTRA_PATHS:-$repo_root/tmp/comfyui-extra-model-paths.yaml}"
output_dir="${LTX25_COMFY_OUTPUT_DIR:-$mold_home/output/verification/ltx-2.5/cuda/comfyui}"
temp_dir="${LTX25_COMFY_TEMP_DIR:-$repo_root/tmp/comfyui-temp}"
user_dir="${LTX25_COMFY_USER_DIR:-$repo_root/tmp/comfyui-user}"
port="${LTX25_COMFY_PORT:-8188}"
base_url="http://127.0.0.1:$port"
timestamp="${LTX25_COMFY_CAPTURE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
graph_selector=""

fail() {
  echo "LTX-2.5 ComfyUI CUDA reference failed: $*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
usage: capture-ltx25-comfy-cuda-reference.sh --graph int8|gguf-q4

Runs the pinned 29-node LTX-2.5 ComfyUI workflow on CUDA and seals a
mold.ltx25.comfy-cuda-reference.v1 manifest under
$MOLD_HOME/output/verification/ltx-2.5/cuda/comfyui/reference-<UTC>/.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --graph)
      graph_selector="${2:-}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail "unknown argument: $1"
      ;;
  esac
done

case "$graph_selector" in
  int8)
    default_graph="$repo_root/scripts/fixtures/ltx25-comfy-cuda-int8-api-prompt.json"
    checkpoint="distilled INT8 ConvRot"
    filename_prefix="ltx25-comfy-int8-cuda-seed-25026"
    ;;
  gguf-q4)
    default_graph="$repo_root/scripts/fixtures/ltx25-comfy-cuda-gguf-q4-api-prompt.json"
    checkpoint="distilled GGUF Q4_K_M"
    filename_prefix="ltx25-comfy-gguf-q4-cuda-seed-25026"
    ;;
  *)
    usage >&2
    fail "--graph must be int8 or gguf-q4"
    ;;
esac
graph="${LTX25_COMFY_GRAPH:-$default_graph}"
evidence_dir="$output_dir/reference-$timestamp"
server_log="$evidence_dir/comfyui-server.log"
history_json="$evidence_dir/history.json"
queued_graph="$evidence_dir/prompt.json"
manifest="$evidence_dir/manifest.json"
abort_marker="$evidence_dir/resource-guard-aborted"
server_pid=""
monitor_pid=""

# Resource guard thresholds. Host memory is `MemAvailable / MemTotal` from
# `free -b`; ZFS ARC is not counted as available, so the floor is lower than
# the Metal script's 70% preflight. The RSS ceiling is 48 GiB on a 64 GB host.
readonly preflight_avail_percent=50
readonly abort_avail_percent=20
readonly max_server_rss_kib=50331648
readonly max_seconds=3600

resource_guard_cause() {
  local avail_percent="$1"
  local rss_kib="$2"
  local elapsed="$3"
  local gpu_used_mib="$4"
  if [[ ! "$avail_percent" =~ ^[0-9]+$ ]]; then
    echo "pressure_unreadable"
  elif [[ ! "$gpu_used_mib" =~ ^[0-9]+$ ]]; then
    echo "gpu_unreadable"
  elif ((avail_percent < abort_avail_percent)); then
    echo "host_memory"
  elif [[ "$rss_kib" =~ ^[0-9]+$ ]] && ((rss_kib > max_server_rss_kib)); then
    echo "server_rss"
  elif ((elapsed > max_seconds)); then
    echo "timeout"
  fi
}

host_avail_percent() {
  free -b | awk '/^Mem:/ { if ($2 > 0) printf "%d\n", ($7 * 100) / $2 }'
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -d '[:space:]'
}

stop_owned_server_and_wait() {
  local pid="$1"
  local state
  kill -TERM "$pid" >/dev/null 2>&1 || true
  for _ in $(seq 1 50); do
    state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    if [[ -z "$state" || "$state" == Z* ]]; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill -KILL "$pid" >/dev/null 2>&1 || true
  fi
  wait "$pid" >/dev/null 2>&1 || true
}

if [[ "${LTX25_COMFY_TEST_GUARD:-0}" == 1 ]]; then
  resource_guard_cause \
    "${LTX25_TEST_AVAIL_PERCENT-50}" "${LTX25_TEST_RSS_KIB-0}" \
    "${LTX25_TEST_ELAPSED-0}" "${LTX25_TEST_GPU_USED_MIB-0}"
  exit 0
fi

if [[ "${LTX25_COMFY_TEST_STABLE_SEAL:-0}" == 1 ]]; then
  seal_test_log="$(mktemp)"
  bash -c 'trap '\''printf "shutdown complete\n" >>"$1"; exit 0'\'' TERM; while :; do sleep 0.1; done' \
    _ "$seal_test_log" &
  seal_test_pid=$!
  sleep 0.2
  stop_owned_server_and_wait "$seal_test_pid"
  grep -Fq "shutdown complete" "$seal_test_log"
  seal_test_hash="$(sha256sum "$seal_test_log" | awk '{print $1}')"
  sleep 0.2
  [[ "$seal_test_hash" == "$(sha256sum "$seal_test_log" | awk '{print $1}')" ]]
  rm -f "$seal_test_log"
  exit 0
fi

validate_graph() {
  local loader
  case "$graph_selector" in
    int8)
      loader='.["1"].class_type == "UNETLoader"
        and .["1"].inputs.unet_name == "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"'
      ;;
    gguf-q4)
      loader='.["1"].class_type == "UnetLoaderGGUF"
        and .["1"].inputs == {unet_name: "LTX-2.5-Distilled-Q4_K_M.gguf"}'
      ;;
  esac
  jq -e --arg prefix "$filename_prefix" "
    length == 29
    and ($loader)
    and .[\"2\"].inputs.vae_name == \"ltx-2.5-video-vae-conv-bf16.safetensors\"
    and .[\"3\"].inputs.vae_name == \"ltx-2.5-audio-vae-bf16.safetensors\"
    and .[\"4\"].inputs.clip_name == \"gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors\"
    and .[\"8\"].inputs == {width:128,height:128,length:9,batch_size:1}
    and .[\"9\"].inputs.frames_number == 9
    and .[\"11\"].inputs.noise_seed == 25026
    and .[\"13\"].inputs.sigmas == \"1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0\"
    and .[\"20\"].inputs.noise_seed == 42
    and .[\"22\"].inputs.sigmas == \"0.909375, 0.7250, 0.421875, 0.0\"
    and .[\"26\"].inputs.tile_size == 256
    and .[\"28\"].inputs.fps == 24
    and .[\"29\"].inputs.filename_prefix == \$prefix
  " "$graph" >/dev/null || fail "unsafe or unexpected API graph: $graph"
}

validate_graph
if [[ "${LTX25_COMFY_VALIDATE_ONLY:-0}" == 1 ]]; then
  echo "$graph"
  exit 0
fi

for command in curl ffprobe free git jq nvidia-smi ps sha256sum; do
  command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] \
  || fail "runtime capture is restricted to Linux x86_64 CUDA"
[[ "$mold_home" == /mnt/storage20tb/AI/mold ]] \
  || fail "MOLD_HOME must be /mnt/storage20tb/AI/mold"
[[ -n "$models_dir" ]] || fail "MOLD_MODELS_DIR is required (the model store is separate from MOLD_HOME)"
[[ -x "$python" ]] || fail "missing retained ComfyUI Python environment: $python"
[[ -f "$comfy_root/main.py" ]] || fail "missing ComfyUI reference checkout: $comfy_root"
[[ -f "$extra_paths" ]] || fail "missing ComfyUI model path configuration: $extra_paths"
[[ "$(git -C "$comfy_root" rev-parse HEAD)" == a1079ba16f2674734b065eb036fbfdddaa321a4d ]] \
  || fail "ComfyUI reference checkout is not at the pinned commit"
[[ -z "$(git -C "$comfy_root" status --porcelain)" ]] \
  || fail "ComfyUI reference checkout is not clean"
if [[ "$graph_selector" == gguf-q4 ]]; then
  [[ -f "$comfy_root/custom_nodes/ComfyUI-GGUF/nodes.py" ]] \
    || fail "missing city96/ComfyUI-GGUF custom node under $comfy_root/custom_nodes"
fi

required_models=(
  "$models_dir/shared/ltx2/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
  "$models_dir/shared/ltx2/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
  "$models_dir/shared/ltx2/vae/ltx-2.5-audio-vae-bf16.safetensors"
  "$models_dir/shared/ltx2/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
)
case "$graph_selector" in
  int8)
    required_models+=("$models_dir/ltx-2.5-22b-distilled-int8-conv/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors")
    ;;
  gguf-q4)
    required_models+=("$models_dir/ltx-2.5-22b-distilled-q4/LTX-2.5-Distilled-Q4_K_M.gguf")
    ;;
esac
for model in "${required_models[@]}"; do
  [[ -s "$model" && -f "$model.sha256-verified" ]] \
    || fail "missing retained model or SHA marker: $model"
done

avail_percent="$(host_avail_percent)"
[[ "$avail_percent" =~ ^[0-9]+$ ]] || fail "could not read host memory from free -b"
((avail_percent >= preflight_avail_percent)) \
  || fail "resource preflight requires at least ${preflight_avail_percent}% available host memory; found $avail_percent%"
[[ "$(gpu_used_mib)" =~ ^[0-9]+$ ]] || fail "could not read GPU memory from nvidia-smi"
curl --fail --silent --max-time 2 "$base_url/system_stats" >/dev/null 2>&1 \
  && fail "port $port already has a ComfyUI server; refusing to control an unrelated process"

mkdir -p "$output_dir" "$temp_dir" "$user_dir" "$evidence_dir"
cp "$graph" "$queued_graph"

# Every torch call in this script runs with the NixOS driver libraries in
# scope; a wheel that still cannot see the device is a deferral, not a crash.
export LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

seal_operator_deferred() {
  local cause="$1" reason blocking_operator upstream_progress prompt_id="${2:-}"
  case "$cause" in
    pressure_unreadable) reason="host memory became unreadable from free -b" ;;
    host_memory) reason="available host memory fell below the ${abort_avail_percent}% safety floor" ;;
    server_rss) reason="ComfyUI server RSS exceeded the 48 GiB safety ceiling" ;;
    timeout) reason="official ComfyUI CUDA workflow exceeded the 60-minute resource budget" ;;
    gpu_unreadable) reason="GPU memory became unreadable from nvidia-smi" ;;
    torch_cuda_unavailable) reason="the retained torch build cannot initialise CUDA on this host" ;;
    *) fail "unknown resource guard cause: $cause" ;;
  esac
  [[ -f "$server_log" ]] || : >"$server_log"
  blocking_operator=""
  upstream_progress=""
  if grep -Eq '0%.*0/8' "$server_log"; then
    upstream_progress="official sampler reached 0/8 after model load"
  fi
  jq -n \
    --arg captured_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg prompt_id "$prompt_id" \
    --arg reason "$reason" --arg graph "$queued_graph" --arg checkpoint "$checkpoint" \
    --arg graph_sha256 "$(sha256sum "$queued_graph" | awk '{print $1}')" \
    --arg history "$history_json" --arg server_log "$server_log" \
    --arg resource_guard_marker "$abort_marker" \
    --arg resource_guard_marker_sha256 "$(sha256sum "$abort_marker" | awk '{print $1}')" \
    --arg server_log_sha256 "$(sha256sum "$server_log" | awk '{print $1}')" \
    --arg guard_cause "$cause" --arg blocking_operator "$blocking_operator" \
    --arg upstream_progress "$upstream_progress" \
    --argjson preflight "$preflight_avail_percent" --argjson abort_below "$abort_avail_percent" \
    --argjson max_rss "$max_server_rss_kib" --argjson max_seconds "$max_seconds" \
    --slurpfile system "$evidence_dir/system-stats.json" \
    '{schema_version:"mold.ltx25.comfy-cuda-reference.v1", status:"operator_deferred",
      captured_at:$captured_at, implementation:"ComfyUI", backend:"CUDA",
      checkpoint:$checkpoint, prompt_id:(if $prompt_id == "" then null else $prompt_id end),
      settings:{width:256,height:256,frames:9,fps:24,stage1_seed:25026,
        stage2_seed:42,video_cfg:1,audio_cfg:1},
      graph:{path:$graph,sha256:$graph_sha256}, history_path:$history,
      server_log_path:$server_log, server_log_sha256:$server_log_sha256,
      video:null, system_stats:($system[0] // null), retained_in_library:false,
      deferred:{reason:$reason, guard_cause:$guard_cause,
        resource_guard_marker:$resource_guard_marker,
        resource_guard_marker_sha256:$resource_guard_marker_sha256,
        upstream_progress:(if $upstream_progress == "" then null else $upstream_progress end),
        blocking_operator:(if $blocking_operator == "" then null else $blocking_operator end)},
      preservation:{downloaded_models_deleted:false,rendered_media_deleted:false},
      resource_guard:{minimum_preflight_percent:$preflight,abort_below_percent:$abort_below,
        max_server_rss_kib:$max_rss,max_seconds:$max_seconds}}' >"$manifest"
  echo "$manifest"
}

# torch must see the device before ComfyUI is started at all; otherwise the
# server would silently fall back to CPU and the oracle would be meaningless.
printf '[]\n' >"$evidence_dir/system-stats.json"
if ! "$python" - >"$evidence_dir/torch-cuda-probe.json" 2>"$evidence_dir/torch-cuda-probe.log" <<'PY'
import json, sys
import torch
ok = torch.cuda.is_available()
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                  "cuda_available": ok,
                  "device": torch.cuda.get_device_name(0) if ok else None}))
sys.exit(0 if ok else 3)
PY
then
  jq -n --arg stopped_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{cause:"torch_cuda_unavailable",elapsed_seconds:0,stopped_at:$stopped_at}' >"$abort_marker"
  seal_operator_deferred torch_cuda_unavailable
  exit 0
fi

cleanup() {
  local status=$?
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
  fi
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

(
  cd "$comfy_root"
  env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$python" main.py \
    --listen 127.0.0.1 --port "$port" \
    --extra-model-paths-config "$extra_paths" \
    --output-directory "$output_dir" --temp-directory "$temp_dir" \
    --user-directory "$user_dir" --disable-auto-launch \
    --cache-none --preview-method none \
    --disable-smart-memory --verbose INFO
) >"$server_log" 2>&1 &
server_pid=$!

for _ in $(seq 1 120); do
  if ! kill -0 "$server_pid" >/dev/null 2>&1; then
    fail "ComfyUI exited during startup; see $server_log"
  fi
  if curl --fail --silent --max-time 2 "$base_url/system_stats" >"$evidence_dir/system-stats.json"; then
    break
  fi
  sleep 1
done
[[ -s "$evidence_dir/system-stats.json" ]] || fail "ComfyUI did not start within 120 seconds"
jq -e '.devices[] | select((.type | ascii_downcase) == "cuda")' \
  "$evidence_dir/system-stats.json" >/dev/null || fail "ComfyUI did not select CUDA"

(
  started="$(date +%s)"
  while kill -0 "$server_pid" >/dev/null 2>&1; do
    avail="$(host_avail_percent)"
    rss_kib="$(ps -o rss= -p "$server_pid" | tr -d ' ')"
    elapsed="$(($(date +%s) - started))"
    gpu_mib="$(gpu_used_mib)"
    cause="$(resource_guard_cause "$avail" "$rss_kib" "$elapsed" "$gpu_mib")"
    if [[ -n "$cause" ]]; then
      jq -n --arg cause "$cause" --arg avail "$avail" --arg rss_kib "$rss_kib" \
        --arg elapsed "$elapsed" --arg gpu_mib "$gpu_mib" \
        --arg stopped_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{cause:$cause,available_percent:($avail | tonumber?),
          server_rss_kib:($rss_kib | tonumber?),elapsed_seconds:($elapsed | tonumber),
          gpu_memory_used_mib:($gpu_mib | tonumber?),stopped_at:$stopped_at}' >"$abort_marker"
      kill -TERM "$server_pid" >/dev/null 2>&1 || true
      exit 0
    fi
    sleep 2
  done
) &
monitor_pid=$!

client_id="mold-ltx25-cuda-$timestamp"
queue_response="$evidence_dir/queue-response.json"
jq -n --slurpfile prompt "$queued_graph" --arg client_id "$client_id" \
  '{prompt:$prompt[0], client_id:$client_id}' \
  | curl --fail --silent --show-error -H 'Content-Type: application/json' \
    --data-binary @- "$base_url/prompt" >"$queue_response"
prompt_id="$(jq -er '.prompt_id' "$queue_response")"

for _ in $(seq 1 1800); do
  if [[ -f "$abort_marker" ]]; then
    stop_owned_server_and_wait "$server_pid"
    server_pid=""
    seal_operator_deferred "$(jq -er '.cause' "$abort_marker")" "$prompt_id"
    exit 0
  fi
  if ! kill -0 "$server_pid" >/dev/null 2>&1; then
    fail "ComfyUI exited during inference; see $server_log"
  fi
  if curl --fail --silent --max-time 5 "$base_url/history/$prompt_id" >"$history_json" \
    && jq -e --arg id "$prompt_id" 'has($id)' "$history_json" >/dev/null; then
    break
  fi
  sleep 2
done
jq -e --arg id "$prompt_id" '.[$id].status.status_str == "success"' \
  "$history_json" >/dev/null || fail "ComfyUI job did not complete successfully; see $history_json"

relative_output="$(jq -er --arg id "$prompt_id" '
  [.[$id].outputs[]? | .videos[]?, .images[]?]
  | map(select(.type == "output" and (.filename | endswith(".mp4"))))
  | first | ((if (.subfolder // "") == "" then "" else .subfolder + "/" end) + .filename)
' "$history_json")"
video="$output_dir/$relative_output"
[[ -s "$video" ]] || fail "ComfyUI reported output is missing: $video"
ffprobe -v error -count_frames -show_entries \
  format=filename,size,duration:stream=codec_name,codec_type,width,height,r_frame_rate,nb_frames,nb_read_frames,sample_rate,channels \
  -of json "$video" >"$evidence_dir/ffprobe.json"
jq -e '.streams[] | select(.codec_type == "video" and .width == 256 and .height == 256
  and .r_frame_rate == "24/1" and ((.nb_frames // .nb_read_frames) | tonumber) == 9)' \
  "$evidence_dir/ffprobe.json" >/dev/null || fail "ComfyUI output is not 256x256, 9 frames at 24 fps"
jq -e '.streams[] | select(.codec_type == "audio" and .sample_rate == "48000" and .channels == 2)' \
  "$evidence_dir/ffprobe.json" >/dev/null || fail "ComfyUI output is missing stereo 48 kHz audio"

jq -n \
  --arg captured_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg prompt_id "$prompt_id" \
  --arg checkpoint "$checkpoint" \
  --arg video "$video" --arg video_sha256 "$(sha256sum "$video" | awk '{print $1}')" \
  --arg graph "$queued_graph" --arg graph_sha256 "$(sha256sum "$queued_graph" | awk '{print $1}')" \
  --arg history "$history_json" --arg server_log "$server_log" \
  --argjson preflight "$preflight_avail_percent" --argjson abort_below "$abort_avail_percent" \
  --argjson max_rss "$max_server_rss_kib" --argjson max_seconds "$max_seconds" \
  --slurpfile system "$evidence_dir/system-stats.json" --slurpfile probe "$evidence_dir/ffprobe.json" \
  --slurpfile torch "$evidence_dir/torch-cuda-probe.json" \
  '{schema_version:"mold.ltx25.comfy-cuda-reference.v1", status:"passed", captured_at:$captured_at,
    implementation:"ComfyUI", backend:"CUDA", checkpoint:$checkpoint,
    prompt_id:$prompt_id, settings:{width:256,height:256,frames:9,fps:24,
      stage1_seed:25026,stage2_seed:42,video_cfg:1,audio_cfg:1},
    graph:{path:$graph,sha256:$graph_sha256}, history_path:$history,
    server_log_path:$server_log, video:{path:$video,sha256:$video_sha256,ffprobe:$probe[0]},
    system_stats:$system[0], torch:$torch[0], retained_in_library:true,
    resource_guard:{minimum_preflight_percent:$preflight,abort_below_percent:$abort_below,
      max_server_rss_kib:$max_rss,max_seconds:$max_seconds}}' >"$manifest"

echo "$manifest"
