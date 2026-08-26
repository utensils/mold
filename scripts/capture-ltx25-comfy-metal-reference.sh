#!/usr/bin/env bash
set -euo pipefail

# Long captures can outlive a transient `nix develop -c` environment. Prefer
# the user's GC-rooted profile and macOS system tools so Bash never caches a
# devshell shim whose target may be collected while ComfyUI is still running.
PATH="${HOME}/.nix-profile/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH}"
export PATH
hash -r

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mold_home="${MOLD_HOME:-/Volumes/ExternalStorage/mold2}"
comfy_root="${LTX25_COMFY_ROOT:-$repo_root/tmp/comfyui-upstream}"
python="${LTX25_COMFY_PYTHON:-$mold_home/comfyui-venv/bin/python}"
graph="${LTX25_COMFY_GRAPH:-$repo_root/scripts/fixtures/ltx25-comfy-metal-api-prompt.json}"
extra_paths="${LTX25_COMFY_EXTRA_PATHS:-$mold_home/comfyui-extra-model-paths.yaml}"
output_dir="${LTX25_COMFY_OUTPUT_DIR:-$mold_home/output/verification/ltx-2.5/comfyui}"
temp_dir="${LTX25_COMFY_TEMP_DIR:-$mold_home/comfyui-temp}"
user_dir="${LTX25_COMFY_USER_DIR:-$mold_home/comfyui-user}"
port="${LTX25_COMFY_PORT:-8188}"
base_url="http://127.0.0.1:$port"
timestamp="${LTX25_COMFY_CAPTURE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
evidence_dir="$output_dir/reference-$timestamp"
server_log="$evidence_dir/comfyui-server.log"
history_json="$evidence_dir/history.json"
queued_graph="$evidence_dir/prompt.json"
manifest="$evidence_dir/manifest.json"
abort_marker="$evidence_dir/resource-guard-aborted"
server_pid=""
monitor_pid=""

fail() {
  echo "LTX-2.5 ComfyUI Metal reference failed: $*" >&2
  exit 1
}

resource_guard_cause() {
  local pressure="$1"
  local rss_kib="$2"
  local elapsed="$3"
  if [[ ! "$pressure" =~ ^[0-9]+$ ]]; then
    echo "pressure_unreadable"
  elif ((pressure < 20)); then
    echo "memory_pressure"
  elif [[ "$rss_kib" =~ ^[0-9]+$ ]] && ((rss_kib > 37748736)); then
    echo "server_rss"
  elif ((elapsed > 3600)); then
    echo "timeout"
  fi
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
    "${LTX25_TEST_PRESSURE:-50}" "${LTX25_TEST_RSS_KIB:-0}" "${LTX25_TEST_ELAPSED:-0}"
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
  seal_test_hash="$(shasum -a 256 "$seal_test_log" | awk '{print $1}')"
  sleep 0.2
  [[ "$seal_test_hash" == "$(shasum -a 256 "$seal_test_log" | awk '{print $1}')" ]]
  rm -f "$seal_test_log"
  exit 0
fi

validate_graph() {
  jq -e '
    length == 29
    and .["1"].class_type == "UNETLoader"
    and .["1"].inputs.unet_name == "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    and .["2"].inputs.vae_name == "ltx-2.5-video-vae-conv-bf16.safetensors"
    and .["3"].inputs.vae_name == "ltx-2.5-audio-vae-bf16.safetensors"
    and .["4"].inputs.clip_name == "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
    and .["8"].inputs == {width:128,height:128,length:9,batch_size:1}
    and .["9"].inputs.frames_number == 9
    and .["11"].inputs.noise_seed == 25026
    and .["13"].inputs.sigmas == "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    and .["20"].inputs.noise_seed == 42
    and .["22"].inputs.sigmas == "0.909375, 0.7250, 0.421875, 0.0"
    and .["26"].inputs.tile_size == 256
    and .["28"].inputs.fps == 24
    and .["29"].inputs.filename_prefix == "ltx25-comfy-int8-mps-seed-25026"
  ' "$graph" >/dev/null || fail "unsafe or unexpected API graph: $graph"
}

validate_graph
if [[ "${LTX25_COMFY_VALIDATE_ONLY:-0}" == 1 ]]; then
  echo "$graph"
  exit 0
fi

for command in curl ffprobe git jq memory_pressure ps shasum; do
  command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
done
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] \
  || fail "runtime capture is restricted to Apple Silicon Metal"
[[ "$mold_home" == /Volumes/ExternalStorage/mold2 ]] \
  || fail "MOLD_HOME must be /Volumes/ExternalStorage/mold2"
[[ -x "$python" ]] || fail "missing retained ComfyUI Python environment: $python"
[[ -f "$comfy_root/main.py" ]] || fail "missing ComfyUI reference checkout: $comfy_root"
[[ -f "$extra_paths" ]] || fail "missing ComfyUI model path configuration: $extra_paths"
[[ "$(git -C "$comfy_root" rev-parse HEAD)" == a1079ba16f2674734b065eb036fbfdddaa321a4d ]] \
  || fail "ComfyUI reference checkout is not at the pinned commit"
[[ -z "$(git -C "$comfy_root" status --porcelain)" ]] \
  || fail "ComfyUI reference checkout is not clean"

required_models=(
  "$mold_home/models/ltx-2.5-22b-distilled-int8-conv/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
  "$mold_home/models/shared/ltx2/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
  "$mold_home/models/shared/ltx2/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
  "$mold_home/models/shared/ltx2/vae/ltx-2.5-audio-vae-bf16.safetensors"
  "$mold_home/models/shared/ltx2/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
)
for model in "${required_models[@]}"; do
  [[ -s "$model" && -f "$model.sha256-verified" ]] \
    || fail "missing retained model or SHA marker: $model"
done

free_percent="$(memory_pressure -Q | awk '/System-wide memory free percentage/ {gsub(/%/, "", $5); print $5}')"
[[ "$free_percent" =~ ^[0-9]+$ ]] || fail "could not read macOS memory pressure"
(( free_percent >= 70 )) \
  || fail "resource preflight requires at least 70% reclaimable memory; found $free_percent%"
curl --fail --silent --max-time 2 "$base_url/system_stats" >/dev/null 2>&1 \
  && fail "port $port already has a ComfyUI server; refusing to control an unrelated process"

mkdir -p "$output_dir" "$temp_dir" "$user_dir" "$evidence_dir"
cp "$graph" "$queued_graph"

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
  env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 "$python" main.py \
      --listen 127.0.0.1 --port "$port" \
      --extra-model-paths-config "$extra_paths" \
      --output-directory "$output_dir" --temp-directory "$temp_dir" \
      --user-directory "$user_dir" --disable-auto-launch --lowvram \
      --cache-none --preview-method none --reserve-vram 6 \
      --disable-pinned-memory --disable-smart-memory --verbose INFO
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
jq -e '.devices[] | select((.type | ascii_downcase) == "mps")' \
  "$evidence_dir/system-stats.json" >/dev/null || fail "ComfyUI did not select MPS"

(
  started="$(date +%s)"
  while kill -0 "$server_pid" >/dev/null 2>&1; do
    pressure="$(memory_pressure -Q | awk '/System-wide memory free percentage/ {gsub(/%/, "", $5); print $5}')"
    rss_kib="$(ps -o rss= -p "$server_pid" | tr -d ' ')"
    elapsed="$(( $(date +%s) - started ))"
    cause="$(resource_guard_cause "$pressure" "$rss_kib" "$elapsed")"
    if [[ -n "$cause" ]]; then
      jq -n --arg cause "$cause" --arg pressure "$pressure" --arg rss_kib "$rss_kib" \
        --arg elapsed "$elapsed" --arg stopped_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{cause:$cause,pressure_percent:($pressure | tonumber?),
          server_rss_kib:($rss_kib | tonumber?),elapsed_seconds:($elapsed | tonumber),
          stopped_at:$stopped_at}' >"$abort_marker"
      kill -TERM "$server_pid" >/dev/null 2>&1 || true
      exit 0
    fi
    sleep 2
  done
) &
monitor_pid=$!

client_id="mold-ltx25-$timestamp"
queue_response="$evidence_dir/queue-response.json"
jq -n --slurpfile prompt "$queued_graph" --arg client_id "$client_id" \
  '{prompt:$prompt[0], client_id:$client_id}' \
  | curl --fail --silent --show-error -H 'Content-Type: application/json' \
      --data-binary @- "$base_url/prompt" >"$queue_response"
prompt_id="$(jq -er '.prompt_id' "$queue_response")"

seal_operator_deferred() {
  local cause reason blocking_operator upstream_progress
  cause="$(jq -er '.cause' "$abort_marker")" || fail "invalid resource guard marker"
  case "$cause" in
    pressure_unreadable) reason="macOS memory pressure became unreadable" ;;
    memory_pressure) reason="macOS reclaimable memory fell below the 20% safety floor" ;;
    server_rss) reason="ComfyUI server RSS exceeded the 36 GiB safety ceiling" ;;
    timeout) reason="official ComfyUI MPS workflow exceeded the 60-minute resource budget" ;;
    *) fail "unknown resource guard cause: $cause" ;;
  esac
  blocking_operator=""
  upstream_progress=""
  if grep -Fq "aten::_int_mm" "$server_log" \
    && grep -Eq 'not currently (implemented for the MPS device|supported on the MPS backend)' "$server_log"; then
    blocking_operator="aten::_int_mm fell back from MPS to CPU"
  fi
  if grep -Eq '0%.*0/8' "$server_log"; then
    upstream_progress="official sampler reached 0/8 after model load"
  fi
  jq -n \
    --arg captured_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg prompt_id "$prompt_id" \
    --arg reason "$reason" --arg graph "$queued_graph" \
    --arg graph_sha256 "$(shasum -a 256 "$queued_graph" | awk '{print $1}')" \
    --arg history "$history_json" --arg server_log "$server_log" \
    --arg queue_response "$queue_response" --arg resource_guard_marker "$abort_marker" \
    --arg resource_guard_marker_sha256 "$(shasum -a 256 "$abort_marker" | awk '{print $1}')" \
    --arg server_log_sha256 "$(shasum -a 256 "$server_log" | awk '{print $1}')" \
    --arg guard_cause "$cause" --arg blocking_operator "$blocking_operator" \
    --arg upstream_progress "$upstream_progress" \
    --slurpfile system "$evidence_dir/system-stats.json" \
    '{schema_version:"mold.ltx25.comfy-metal-reference.v1", status:"operator_deferred",
      captured_at:$captured_at, implementation:"ComfyUI", backend:"MPS",
      checkpoint:"distilled INT8 ConvRot", prompt_id:$prompt_id,
      settings:{width:256,height:256,frames:9,fps:24,stage1_seed:25026,
        stage2_seed:42,video_cfg:1,audio_cfg:1},
      graph:{path:$graph,sha256:$graph_sha256}, history_path:$history,
      server_log_path:$server_log, server_log_sha256:$server_log_sha256,
      queue_response_path:$queue_response,
      video:null, system_stats:$system[0], retained_in_library:false,
      deferred:{reason:$reason, guard_cause:$guard_cause,
        resource_guard_marker:$resource_guard_marker,
        resource_guard_marker_sha256:$resource_guard_marker_sha256,
        upstream_progress:(if $upstream_progress == "" then null else $upstream_progress end),
        blocking_operator:(if $blocking_operator == "" then null else $blocking_operator end)},
      preservation:{downloaded_models_deleted:false,rendered_media_deleted:false},
      resource_guard:{minimum_preflight_percent:70,abort_below_percent:20,
        max_server_rss_kib:37748736,max_seconds:3600}}' >"$manifest"
  echo "$manifest"
}

for _ in $(seq 1 1800); do
  if [[ -f "$abort_marker" ]]; then
    stop_owned_server_and_wait "$server_pid"
    server_pid=""
    seal_operator_deferred
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
  --arg video "$video" --arg video_sha256 "$(shasum -a 256 "$video" | awk '{print $1}')" \
  --arg graph "$queued_graph" --arg graph_sha256 "$(shasum -a 256 "$queued_graph" | awk '{print $1}')" \
  --arg history "$history_json" --arg server_log "$server_log" \
  --slurpfile system "$evidence_dir/system-stats.json" --slurpfile probe "$evidence_dir/ffprobe.json" \
  '{schema_version:"mold.ltx25.comfy-metal-reference.v1", status:"passed", captured_at:$captured_at,
    implementation:"ComfyUI", backend:"MPS", checkpoint:"distilled INT8 ConvRot",
    prompt_id:$prompt_id, settings:{width:256,height:256,frames:9,fps:24,
      stage1_seed:25026,stage2_seed:42,video_cfg:1,audio_cfg:1},
    graph:{path:$graph,sha256:$graph_sha256}, history_path:$history,
    server_log_path:$server_log, video:{path:$video,sha256:$video_sha256,ffprobe:$probe[0]},
    system_stats:$system[0], retained_in_library:true, resource_guard:{minimum_preflight_percent:70,
      abort_below_percent:20,max_server_rss_kib:37748736,max_seconds:3600}}' >"$manifest"

echo "$manifest"
