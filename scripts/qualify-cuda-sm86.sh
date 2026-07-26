#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  qualify-cuda-sm86.sh \
    --release-tag <vMAJOR.MINOR.PATCH> \
    --sm86-binary <mold-x86_64-unknown-linux-gnu-cuda-sm86> \
    --image-model <installed-flux-model-id> \
    --video-model <installed-video-model-id> \
    --chain-script <mold.chain.v1.toml> \
    --report <output.json> [--timeout-seconds <n>]

Runs the deferred real RTX 3090 distribution qualification matrix. The runner
downloads the exact official release provenance from GitHub; caller-supplied
hashes are deliberately unsupported. A passing report requires the exact
verified sm86 Linux artifact with exact embedded PTX, matching source identity,
decoded 256x256 image/video/chained-video outputs, selected GPU UUID evidence,
observed CUDA compute work from the exact generation PID, math attention, and a
successful CUDA Driver API load of an exact embedded sm86 PTX module followed
by normal full-Mold generation.

PTX is compatible only with equal-or-higher device compute capabilities. An
sm89 artifact is expected to fail on this sm86 hardware and must be recorded as
a separate negative incompatibility regression, never as a positive smoke.
Process-wide CUDA_FORCE_PTX_JIT is not used because it also forces NVIDIA
runtime libraries through JIT and can fail before Mold's own PTX is reached.

This records evidence; it is not a cryptographic attestation service and does
not provision hardware.
EOF
}

release_tag=""
sm86_binary=""
image_model=""
video_model=""
chain_script=""
report=""
timeout_seconds=1800

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release-tag) release_tag="${2:-}"; shift 2 ;;
    --sm86-binary) sm86_binary="${2:-}"; shift 2 ;;
    --image-model|--model) image_model="${2:-}"; shift 2 ;;
    --video-model) video_model="${2:-}"; shift 2 ;;
    --chain-script) chain_script="${2:-}"; shift 2 ;;
    --report) report="${2:-}"; shift 2 ;;
    --timeout-seconds) timeout_seconds="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

for command in curl ffprobe jq nvidia-smi pgrep python3 readelf realpath sha256sum timeout; do
  command -v "$command" >/dev/null \
    || { echo "required command is unavailable: $command" >&2; exit 69; }
done
[[ "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] \
  || { echo "--release-tag must be an exact stable tag such as v0.20.2" >&2; exit 64; }
[[ -n "$image_model" ]] || { echo "--image-model is required" >&2; exit 64; }
[[ "$image_model" == flux* && "$image_model" != *:q[0-9]* ]] \
  || {
    echo "--image-model must be an installed non-GGUF FLUX-family model so --offload exercises Mold attention" >&2
    exit 64
  }
[[ -n "$video_model" ]] || { echo "--video-model is required" >&2; exit 64; }
[[ -f "$chain_script" ]] || { echo "chain script is missing: $chain_script" >&2; exit 66; }
[[ -n "$report" ]] || { echo "--report is required" >&2; exit 64; }
[[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] \
  || { echo "--timeout-seconds must be a positive integer" >&2; exit 64; }
[[ -x "$sm86_binary" ]] || { echo "sm86 binary is not executable: $sm86_binary" >&2; exit 66; }

sm86_binary="$(realpath "$sm86_binary")"

scratch_dir="$(mktemp -d)"
trap 'rm -rf "$scratch_dir"' EXIT
report_parent="$(dirname "$report")"
mkdir -p "$report_parent"
report="$(realpath -m "$report")"
evidence_dir="${report}.d"
mkdir -p "$evidence_dir"

provenance_url="https://github.com/utensils/mold/releases/download/${release_tag}/mold-release-provenance.json"
provenance_path="$scratch_dir/mold-release-provenance.json"
curl --fail --silent --show-error --location \
  --retry 3 --retry-delay 1 --retry-connrefused \
  "$provenance_url" -o "$provenance_path" \
  || { echo "failed to fetch trusted official release provenance: $provenance_url" >&2; exit 69; }

jq -e --arg tag "$release_tag" '
  .schema_version == "mold.release.provenance.v1"
  and .release_tag == $tag
  and (.source_sha | test("^[0-9a-f]{40}$"))
  and .artifacts.sm86.filename == "mold-x86_64-unknown-linux-gnu-cuda-sm86.tar.gz"
  and .artifacts.sm86.cuda_target == "sm_86"
  and (.artifacts.sm86.archive_sha256 | test("^[0-9a-f]{64}$"))
  and (.artifacts.sm86.binary_sha256 | test("^[0-9a-f]{64}$"))
' "$provenance_path" >/dev/null \
  || { echo "official release provenance has an invalid shape or target mapping" >&2; exit 65; }

source_sha="$(jq -r '.source_sha' "$provenance_path")"
sm86_expected_sha="$(jq -r '.artifacts.sm86.binary_sha256' "$provenance_path")"
sm86_actual_sha="$(sha256sum "$sm86_binary" | awk '{print $1}')"
if [[ "$sm86_actual_sha" != "$sm86_expected_sha" ]]; then
  echo "artifact checksum does not match the exact official release provenance" >&2
  exit 65
fi

"$repo_root/scripts/verify-cuda-release-binary.sh" "$sm86_binary" 86 >/dev/null
source_short="${source_sha:0:7}"
sm86_version="$("$sm86_binary" version)"
grep -Fq "$source_short" <<<"$sm86_version" \
  || { echo "sm86 binary does not identify release source $source_sha" >&2; exit 65; }

device_csv="$scratch_dir/devices.csv"
nvidia-smi \
  --query-gpu=uuid,name,compute_cap,driver_version \
  --format=csv,noheader,nounits >"$device_csv" \
  || { echo "failed to inspect RTX 3090 qualification devices" >&2; exit 69; }
devices_json="$(
  jq -Rn '
    def trim: gsub("^[[:space:]]+|[[:space:]]+$"; "");
    [inputs
      | select(length > 0)
      | split(",")
      | select(length == 4)
      | {
          uuid: (.[0] | trim),
          name: (.[1] | trim),
          compute_capability: (.[2] | trim),
          driver_version: (.[3] | trim)
        }
    ]
  ' <"$device_csv"
)"
devices_ok="$(
  jq -r '
    length > 0
    and all(.[];
      (.uuid | test("^GPU-[0-9A-Fa-f-]+$"))
      and (.name | ascii_upcase | contains("RTX 3090"))
      and .compute_capability == "8.6")
  ' <<<"$devices_json"
)"
[[ "$devices_ok" == true ]] \
  || { echo "every qualification device must be an RTX 3090 with compute capability 8.6" >&2; exit 65; }
mapfile -t device_uuids < <(jq -r '.[].uuid' <<<"$devices_json")

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
hostname_value="$(hostname)"

run_smoke() {
  local label="$1"
  local binary="$2"
  local selected_uuid="$3"
  local media_kind="$4"
  local probe_embedded_ptx="$5"
  shift 5
  local output="$1"
  shift
  local log="$evidence_dir/${label}.log"
  local result="$scratch_dir/${label}.json"
  local ptx_probe_path=""
  local ptx_probe_sha256=""
  local ptx_probe_exit=0
  local embedded_ptx_module_loaded=false
  local -a env_args=(
    CUDA_VISIBLE_DEVICES="$selected_uuid"
    MOLD_ATTN=math
    MOLD_LOG=info
    MOLD_DB_DISABLE=1
  )
  local -a command=("$binary" "$@" --output "$output")
  local rendered_command
  printf -v rendered_command '%q ' env "${env_args[@]}" "${command[@]}"

  if [[ "$probe_embedded_ptx" == true ]]; then
    ptx_probe_path="$evidence_dir/${label}-embedded-ptx.json"
    local rendered_probe
    printf -v rendered_probe '%q ' env "CUDA_VISIBLE_DEVICES=$selected_uuid" \
      "$repo_root/scripts/probe-cuda-embedded-ptx.py" "$binary" 86
    rendered_command="${rendered_probe}&& ${rendered_command}"
    set +e
    timeout "$timeout_seconds" env "CUDA_VISIBLE_DEVICES=$selected_uuid" \
      "$repo_root/scripts/probe-cuda-embedded-ptx.py" "$binary" 86 \
      >"$ptx_probe_path" 2>&1
    ptx_probe_exit=$?
    set -e
    if [[ "$ptx_probe_exit" -eq 0 ]] \
      && jq -e '.expected_target == "sm_86" and .loaded == true' \
        "$ptx_probe_path" >/dev/null; then
      embedded_ptx_module_loaded=true
    fi
    ptx_probe_sha256="$(sha256sum "$ptx_probe_path" | awk '{print $1}')"
  fi

  set +e
  timeout "$timeout_seconds" env "${env_args[@]}" "${command[@]}" >"$log" 2>&1 &
  local run_pid=$!
  local compute_observed=false
  while kill -0 "$run_pid" 2>/dev/null; do
    local observed_pids="|${run_pid}|"
    local child_pid
    while IFS= read -r child_pid; do
      [[ -n "$child_pid" ]] && observed_pids="${observed_pids}${child_pid}|"
    done < <(pgrep -P "$run_pid" 2>/dev/null || true)
    if nvidia-smi \
      --query-compute-apps=pid,gpu_uuid \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v pids="$observed_pids" -v uuid="$selected_uuid" '
          index(pids, "|" $1 "|") && $2 == uuid { found = 1 }
          END { exit !found }
        '; then
      compute_observed=true
    fi
    sleep 0.1
  done
  wait "$run_pid"
  local generation_exit_code=$?
  set -e
  local exit_code="$generation_exit_code"
  if [[ "$probe_embedded_ptx" == true && "$ptx_probe_exit" -ne 0 ]]; then
    exit_code="$ptx_probe_exit"
  fi

  local status=failed
  local media_decoded=false
  local width=0
  local height=0
  local frame_count=0
  if [[ "$exit_code" -eq 0 && -s "$output" && "$compute_observed" == true ]] \
    && grep -Fq "Using CUDA device 0" "$log"; then
    if [[ "$media_kind" == image ]]; then
      if "$repo_root/scripts/verify-png-artifact.py" "$output" 256 256 >/dev/null; then
        media_decoded=true
        width=256
        height=256
        frame_count=1
      fi
    else
      local media_json="$scratch_dir/${label}-media.json"
      if ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=width,height,nb_read_frames \
        -of json "$output" >"$media_json"; then
        width="$(jq -r '.streams[0].width // 0' "$media_json")"
        height="$(jq -r '.streams[0].height // 0' "$media_json")"
        frame_count="$(jq -r '.streams[0].nb_read_frames // 0 | tonumber? // 0' "$media_json")"
        if [[ "$width" -eq 256 && "$height" -eq 256 && "$frame_count" -ge 1 ]]; then
          media_decoded=true
        fi
      fi
    fi
  fi
  if [[ "$media_decoded" == true ]] \
    && { [[ "$media_kind" != image ]] \
      || { grep -Fq "attention backend selected" "$log" && grep -Eiq "math" "$log"; }; } \
    && { [[ "$probe_embedded_ptx" != true ]] \
      || [[ "$embedded_ptx_module_loaded" == true ]]; }; then
    status=passed
  fi

  local output_sha=""
  [[ -s "$output" ]] && output_sha="$(sha256sum "$output" | awk '{print $1}')"
  local log_sha
  log_sha="$(sha256sum "$log" | awk '{print $1}')"
  jq -n \
    --arg status "$status" \
    --argjson exit_code "$exit_code" \
    --arg command "$rendered_command" \
    --arg output_path "$output" \
    --arg output_sha256 "$output_sha" \
    --arg log_path "$log" \
    --arg log_sha256 "$log_sha" \
    --arg selected_gpu_uuid "$selected_uuid" \
    --argjson cuda_work_observed "$compute_observed" \
    --argjson media_decoded "$media_decoded" \
    --argjson width "$width" \
    --argjson height "$height" \
    --argjson frame_count "$frame_count" \
    --argjson embedded_ptx_module_loaded "$embedded_ptx_module_loaded" \
    --arg embedded_ptx_probe_path "$ptx_probe_path" \
    --arg embedded_ptx_probe_sha256 "$ptx_probe_sha256" \
    '{
      status: $status,
      exit_code: $exit_code,
      command: $command,
      output_path: $output_path,
      output_sha256: $output_sha256,
      log_path: $log_path,
      log_sha256: $log_sha256,
      selected_gpu_uuid: $selected_gpu_uuid,
      cuda_work_observed: $cuda_work_observed,
      cuda_work_evidence: "exact generation PID and selected UUID observed together via nvidia-smi, plus Mold CUDA-device log",
      media_decoded: $media_decoded,
      width: $width,
      height: $height,
      frame_count: $frame_count,
      embedded_ptx_module_loaded: $embedded_ptx_module_loaded,
      embedded_ptx_probe_path: $embedded_ptx_probe_path,
      embedded_ptx_probe_sha256: $embedded_ptx_probe_sha256
    }' >"$result"
}

uuid0="${device_uuids[0]}"
uuid1="${device_uuids[1]:-${device_uuids[0]}}"
run_smoke sm86_attention_image_smoke "$sm86_binary" "$uuid0" image false \
  "$evidence_dir/sm86_attention_image_smoke.png" \
  run "$image_model" "mold CUDA qualification calibration frame" \
  --local --offload --width 256 --height 256 --steps 12 --seed 424242
run_smoke sm86_ptx_image_smoke "$sm86_binary" "$uuid1" image true \
  "$evidence_dir/sm86_ptx_image_smoke.png" \
  run "$image_model" "mold CUDA PTX JIT qualification frame" \
  --local --offload --width 256 --height 256 --steps 12 --seed 424243
run_smoke sm86_video_smoke "$sm86_binary" "$uuid0" video false \
  "$evidence_dir/sm86_video_smoke.mp4" \
  run "$video_model" "mold CUDA qualification video" \
  --local --width 256 --height 256 --frames 9 --steps 2 --seed 424244 --format mp4
run_smoke sm86_chained_video_smoke "$sm86_binary" "$uuid1" video false \
  "$evidence_dir/sm86_chained_video_smoke.mp4" \
  run --script "$(realpath "$chain_script")" --local

tests_json="$(
  jq -n \
    --slurpfile a "$scratch_dir/sm86_attention_image_smoke.json" \
    --slurpfile b "$scratch_dir/sm86_ptx_image_smoke.json" \
    --slurpfile c "$scratch_dir/sm86_video_smoke.json" \
    --slurpfile d "$scratch_dir/sm86_chained_video_smoke.json" \
    '{
      sm86_attention_image_smoke: $a[0],
      sm86_ptx_image_smoke: $b[0],
      sm86_video_smoke: $c[0],
      sm86_chained_video_smoke: $d[0]
    }'
)"
hardware_qualified="$(
  jq -r 'all(.[]; .status == "passed")' <<<"$tests_json"
)"
finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

jq -n \
  --arg source_sha "$source_sha" \
  --arg release_tag "$release_tag" \
  --arg provenance_url "$provenance_url" \
  --arg started_at "$started_at" \
  --arg finished_at "$finished_at" \
  --arg hostname "$hostname_value" \
  --argjson devices "$devices_json" \
  --arg sm86_path "$sm86_binary" \
  --arg sm86_expected "$sm86_expected_sha" \
  --arg sm86_actual "$sm86_actual_sha" \
  --arg sm86_version "$sm86_version" \
  --arg image_model "$image_model" \
  --arg video_model "$video_model" \
  --arg chain_script "$(realpath "$chain_script")" \
  --argjson tests "$tests_json" \
  --argjson hardware_qualified "$hardware_qualified" \
  '{
    schema_version: "mold.cuda.sm86.qualification.v4",
    source_sha: $source_sha,
    release_tag: $release_tag,
    started_at: $started_at,
    finished_at: $finished_at,
    hardware_qualified: $hardware_qualified,
    provenance: {
      official_release_manifest_url: $provenance_url,
      official_release_manifest_verified: true
    },
    host: {
      hostname: $hostname,
      devices: $devices
    },
    artifacts: {
      sm86: {
        path: $sm86_path,
        expected_sha256: $sm86_expected,
        actual_sha256: $sm86_actual,
        cuda_target: "sm_86",
        trusted_checksum_verified: true,
        elf_target_verified: true,
        ptx_target_verified: true,
        source_identity_verified: true,
        version_output: $sm86_version
      }
    },
    workloads: {
      image_model: $image_model,
      video_model: $video_model,
      chain_script: $chain_script
    },
    tests: $tests
  }' >"$report"

"$repo_root/scripts/validate-cuda-qualification-report.py" "$report" >/dev/null
echo "qualification report: $report"
if [[ "$hardware_qualified" != true ]]; then
  echo "RTX 3090 qualification did not pass; report is evidence of failure only" >&2
  exit 1
fi
