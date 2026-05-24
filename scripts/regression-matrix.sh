#!/usr/bin/env bash
set -euo pipefail

HOST="${MOLD_HOST:-http://127.0.0.1:7680}"
MOLD_BIN="${MOLD_BIN:-target/release/mold}"
OUT_ROOT="${MOLD_REGRESSION_OUT:-$HOME/.mold/regression}"
RUN_ID="${MOLD_REGRESSION_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$OUT_ROOT/$RUN_ID"
LOG="$RUN_DIR/results.jsonl"
SOURCE_IMAGE="$RUN_DIR/source.png"

PROMPT_IMAGE="${MOLD_REGRESSION_PROMPT_IMAGE:-a small ceramic teapot on a wooden table, soft window light, realistic}"
PROMPT_VIDEO="${MOLD_REGRESSION_PROMPT_VIDEO:-a small ceramic teapot on a wooden table as the camera slowly pushes in, soft window light}"
STEPS_IMAGE="${MOLD_REGRESSION_STEPS_IMAGE:-}"
STEPS_VIDEO="${MOLD_REGRESSION_STEPS_VIDEO:-}"
FRAMES_VIDEO="${MOLD_REGRESSION_FRAMES_VIDEO:-17}"
FPS_VIDEO="${MOLD_REGRESSION_FPS_VIDEO:-12}"
TIMEOUT_IMAGE="${MOLD_REGRESSION_TIMEOUT_IMAGE:-1800}"
TIMEOUT_VIDEO="${MOLD_REGRESSION_TIMEOUT_VIDEO:-3600}"
MODE="${MOLD_REGRESSION_MODE:-matrix}"
ONLY_MODEL="${MOLD_REGRESSION_ONLY_MODEL:-}"
ONLY_CASE="${MOLD_REGRESSION_ONLY_CASE:-}"
START_AFTER_MODEL="${MOLD_REGRESSION_START_AFTER_MODEL:-}"
START_AFTER_CASE="${MOLD_REGRESSION_START_AFTER_CASE:-}"
MIXED_JOBS="${MOLD_REGRESSION_MIXED_JOBS:-12}"
MIXED_PARALLEL="${MOLD_REGRESSION_MIXED_PARALLEL:-6}"
INCLUDE_OVER_BUDGET="${MOLD_REGRESSION_INCLUDE_OVER_BUDGET:-0}"

if [[ -n "$START_AFTER_MODEL" || -n "$START_AFTER_CASE" ]]; then
  RESUME_READY=false
else
  RESUME_READY=true
fi

mkdir -p "$RUN_DIR"

if [[ ! -x "$MOLD_BIN" ]]; then
  echo "mold binary not executable: $MOLD_BIN" >&2
  exit 2
fi

magick -size 1024x1024 gradient:'#c8d8df-#f7ead8' \
  -fill '#5b4034' -draw 'circle 512,512 512,300' \
  -fill '#f3f0e8' -draw 'rectangle 260,585 764,745' \
  "$SOURCE_IMAGE"

models_json="$RUN_DIR/models.json"
loras_json="$RUN_DIR/loras.json"
curl -fsS "$HOST/api/models" > "$models_json"
curl -fsS "$HOST/api/catalog/installed?kind=lora" > "$loras_json"

jq -r '
  .[]
  | select(.downloaded == true)
  | select(.family | IN("flux","flux2","sd15","sdxl","sd3","z-image","qwen-image","wuerstchen","ltx-video","ltx2"))
  | select((.family != "ltx2") or (((.description // "") | ascii_downcase | contains("fp4") | not) and ((.description // "") | ascii_downcase | contains("nvfp4") | not)))
  | [.name, .family, (.default_steps|tostring), (.default_width|tostring), (.default_height|tostring)]
  | @tsv
' "$models_json" > "$RUN_DIR/models.tsv"

if [[ "$INCLUDE_OVER_BUDGET" != 1 ]]; then
  awk -F '\t' '$1 != "ltx-video-0.9.8-13b-dev:bf16"' "$RUN_DIR/models.tsv" > "$RUN_DIR/models.tsv.tmp"
  mv "$RUN_DIR/models.tsv.tmp" "$RUN_DIR/models.tsv"
fi

awk -F '\t' '$1 != "cv:2925935"' "$RUN_DIR/models.tsv" > "$RUN_DIR/models.tsv.tmp"
mv "$RUN_DIR/models.tsv.tmp" "$RUN_DIR/models.tsv"

jq -r '
  .entries[]
  | select(.primary_path != null)
  | [.family, (.sub_family // ""), .primary_path]
  | @tsv
' "$loras_json" > "$RUN_DIR/loras.tsv"

json_log() {
  jq -nc \
    --arg ts "$(date --iso-8601=seconds)" \
    --arg status "$1" \
    --arg model "$2" \
    --arg family "$3" \
    --arg case "$4" \
    --arg output "$5" \
    --arg cmd "$6" \
    '{ts:$ts,status:$status,model:$model,family:$family,case:$case,output:$output,cmd:$cmd}'
}

run_case() {
  local model="$1" family="$2" case_name="$3" output="$4" timeout_s="$5"
  shift 5
  local -a cmd=("$@")
  local cmd_text
  printf -v cmd_text '%q ' "${cmd[@]}"
  if [[ -n "$ONLY_MODEL" && "$model" != "$ONLY_MODEL" ]]; then
    return 0
  fi
  if [[ -n "$ONLY_CASE" && "$case_name" != "$ONLY_CASE" ]]; then
    return 0
  fi
  if [[ "$RESUME_READY" != true ]]; then
    if [[ "$model" == "$START_AFTER_MODEL" && ( -z "$START_AFTER_CASE" || "$case_name" == "$START_AFTER_CASE" ) ]]; then
      RESUME_READY=true
    fi
    return 0
  fi
  echo "RUN $case_name $model -> $output"
  json_log start "$model" "$family" "$case_name" "$output" "$cmd_text" >> "$LOG"
  if timeout "$timeout_s" "${cmd[@]}" > "$RUN_DIR/${case_name//[^A-Za-z0-9_.-]/_}.stdout" 2> "$RUN_DIR/${case_name//[^A-Za-z0-9_.-]/_}.stderr"; then
    if [[ ! -s "$output" ]]; then
      echo "empty output for $case_name $model: $output" >&2
      json_log empty-output "$model" "$family" "$case_name" "$output" "$cmd_text" >> "$LOG"
      exit 1
    fi
    json_log ok "$model" "$family" "$case_name" "$output" "$cmd_text" >> "$LOG"
  else
    local code=$?
    echo "FAILED $case_name $model (exit $code)" >&2
    echo "$cmd_text" > "$RUN_DIR/FAILED_COMMAND.txt"
    cp "$RUN_DIR/${case_name//[^A-Za-z0-9_.-]/_}.stderr" "$RUN_DIR/FAILED_STDERR.txt"
    json_log "failed:$code" "$model" "$family" "$case_name" "$output" "$cmd_text" >> "$LOG"
    exit "$code"
  fi
}

run_mixed_job() {
  local idx="$1" model="$2" family="$3" case_name="$4" output="$5" timeout_s="$6"
  shift 6
  local -a cmd=("$@")
  local cmd_text
  printf -v cmd_text '%q ' "${cmd[@]}"
  echo "MIXED RUN $idx $case_name $model -> $output"
  json_log start "$model" "$family" "mixed-$idx-$case_name" "$output" "$cmd_text" >> "$LOG"
  if timeout "$timeout_s" "${cmd[@]}" > "$RUN_DIR/mixed-${idx}.stdout" 2> "$RUN_DIR/mixed-${idx}.stderr"; then
    if [[ ! -s "$output" ]]; then
      echo "empty output for mixed job $idx $case_name $model: $output" >&2
      json_log empty-output "$model" "$family" "mixed-$idx-$case_name" "$output" "$cmd_text" >> "$LOG"
      return 1
    fi
    json_log ok "$model" "$family" "mixed-$idx-$case_name" "$output" "$cmd_text" >> "$LOG"
  else
    local code=$?
    echo "FAILED mixed job $idx $case_name $model (exit $code)" >&2
    echo "$cmd_text" > "$RUN_DIR/FAILED_COMMAND.txt"
    cp "$RUN_DIR/mixed-${idx}.stderr" "$RUN_DIR/FAILED_STDERR.txt"
    json_log "failed:$code" "$model" "$family" "mixed-$idx-$case_name" "$output" "$cmd_text" >> "$LOG"
    return "$code"
  fi
}

write_chain_script() {
  local path="$1" model="$2" family="$3" width="$4" height="$5" steps="$6" audio="$7" with_source="$8"
  local motion_tail_frames=0
  [[ "$family" == ltx2 ]] && motion_tail_frames=1
  {
    printf 'schema = "mold.chain.v1"\n\n'
    printf '[chain]\n'
    printf 'model = "%s"\n' "$model"
    printf 'width = %s\n' "$width"
    printf 'height = %s\n' "$height"
    printf 'fps = %s\n' "$FPS_VIDEO"
    printf 'steps = %s\n' "$steps"
    printf 'guidance = 3.0\n'
    printf 'strength = 0.75\n'
    printf 'motion_tail_frames = %s\n' "$motion_tail_frames"
    printf 'output_format = "mp4"\n'
    if [[ "$audio" == true ]]; then
      printf 'enable_audio = true\n'
    fi
    printf '\n[[stage]]\n'
    printf 'prompt = "%s, first short clip"\n' "$PROMPT_VIDEO"
    printf 'frames = 9\n'
    printf 'transition = "smooth"\n'
    if [[ "$with_source" == true && "$family" == ltx2 ]]; then
      printf 'source_image_path = "source.png"\n'
    fi
    printf '\n[[stage]]\n'
    printf 'prompt = "%s, second short clip"\n' "$PROMPT_VIDEO"
    printf 'frames = 9\n'
    printf 'transition = "smooth"\n'
    if [[ "$with_source" == true && "$family" == ltx2 ]]; then
      printf 'source_image_path = "source.png"\n'
    fi
  } > "$path"
}

lora_paths_for_model() {
  local model="$1" family="$2"
  awk -F '\t' -v family="$family" -v model="$model" '
    $1 != family { next }
    family == "flux2" && $2 == "klein-9b" && model !~ /9b|cv:2669986|cv:2650565|cv:2805234|cv:2765147|cv:2663677|cv:2759597/ { next }
    { print $3 }
  ' "$RUN_DIR/loras.tsv" | head -2
}

sanitize() {
  tr ':/ ' '---' <<< "$1"
}

model_row() {
  local model="$1"
  awk -F '\t' -v model="$model" '$1 == model { print; exit }' "$RUN_DIR/models.tsv"
}

model_installed() {
  [[ -n "$(model_row "$1")" ]]
}

supports_chain_cases() {
  local model="$1" family="$2"
  [[ "$family" != ltx2 ]] && return 0
  [[ "$model" == *distilled* ]]
}

run_mixed_queue() {
  local specs=()
  local first_lora
  first_lora="$(lora_paths_for_model "z-image-turbo:q8" "z-image" | head -1 || true)"
  model_installed "flux-dev:q8" && specs+=("flux-dev:q8|flux|base")
  model_installed "sd15:fp16" && specs+=("sd15:fp16|sd15|source")
  model_installed "sd3.5-large:q8" && specs+=("sd3.5-large:q8|sd3|base")
  [[ -n "$first_lora" ]] && model_installed "z-image-turbo:q8" && specs+=("z-image-turbo:q8|z-image|lora1")
  model_installed "flux2-klein:bf16" && specs+=("flux2-klein:bf16|flux2|base")
  model_installed "qwen-image:q8" && specs+=("qwen-image:q8|qwen-image|base")
  model_installed "ltx-video-0.9.8-2b-distilled:bf16" && specs+=("ltx-video-0.9.8-2b-distilled:bf16|ltx-video|base")
  model_installed "ltx-2-19b-distilled:fp8" && specs+=("ltx-2-19b-distilled:fp8|ltx2|audio")

  if (( ${#specs[@]} == 0 )); then
    echo "no installed models available for mixed queue" >&2
    exit 2
  fi

  local active=0
  local failures=0
  local idx=0
  while (( idx < MIXED_JOBS )); do
    local spec="${specs[$((idx % ${#specs[@]}))]}"
    IFS='|' read -r model family case_name <<< "$spec"
    local row default_steps default_width default_height width height steps safe_model prompt output timeout_s
    row="$(model_row "$model")"
    IFS=$'\t' read -r _ _ default_steps default_width default_height <<< "$row"
    width="$default_width"
    height="$default_height"
    steps="$default_steps"
    if [[ "$family" == qwen-image ]]; then
      width="${MOLD_REGRESSION_QWEN_WIDTH:-1024}"
      height="${MOLD_REGRESSION_QWEN_HEIGHT:-1024}"
    fi
    if [[ "$family" == ltx-video ]]; then
      width="${MOLD_REGRESSION_LTX_VIDEO_WIDTH:-768}"
      height="${MOLD_REGRESSION_LTX_VIDEO_HEIGHT:-512}"
    fi
    [[ -n "$STEPS_IMAGE" && "$family" != ltx-video && "$family" != ltx2 ]] && steps="$STEPS_IMAGE"
    [[ -n "$STEPS_VIDEO" && ( "$family" == ltx-video || "$family" == ltx2 ) ]] && steps="$STEPS_VIDEO"
    safe_model="$(sanitize "$model")"
    prompt="$PROMPT_IMAGE"
    timeout_s="$TIMEOUT_IMAGE"
    if [[ "$family" == ltx-video || "$family" == ltx2 ]]; then
      prompt="$PROMPT_VIDEO"
      timeout_s="$TIMEOUT_VIDEO"
      output="$RUN_DIR/mixed-${idx}.${safe_model}.${case_name}.mp4"
      local -a cmd=("$MOLD_BIN" run --host "$HOST" "$model" "$prompt" --output "$output" --format mp4 --frames "$FRAMES_VIDEO" --fps "$FPS_VIDEO" --width "$width" --height "$height" --steps "$steps")
      [[ "$case_name" == audio ]] && cmd+=(--audio)
      [[ "$case_name" == source ]] && cmd+=(--image "$SOURCE_IMAGE")
      run_mixed_job "$idx" "$model" "$family" "$case_name" "$output" "$timeout_s" "${cmd[@]}" &
    else
      output="$RUN_DIR/mixed-${idx}.${safe_model}.${case_name}.png"
      local -a cmd=("$MOLD_BIN" run --host "$HOST" "$model" "$prompt" --output "$output" --format png --width "$width" --height "$height" --steps "$steps")
      [[ "$case_name" == source ]] && cmd+=(--image "$SOURCE_IMAGE")
      [[ "$case_name" == lora1 ]] && cmd+=(--lora "$first_lora")
      run_mixed_job "$idx" "$model" "$family" "$case_name" "$output" "$timeout_s" "${cmd[@]}" &
    fi
    active=$((active + 1))
    idx=$((idx + 1))
    if (( active >= MIXED_PARALLEL )); then
      if ! wait -n; then
        failures=$((failures + 1))
      fi
      active=$((active - 1))
    fi
  done
  while (( active > 0 )); do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    active=$((active - 1))
  done
  if (( failures > 0 )); then
    echo "mixed queue failed with $failures failed job(s): $RUN_DIR" >&2
    exit 1
  fi
  echo "mixed queue passed: $RUN_DIR"
}

if [[ "$MODE" == "mixed-queue" ]]; then
  run_mixed_queue
  exit 0
fi

while IFS=$'\t' read -r model family default_steps default_width default_height; do
  safe_model="$(sanitize "$model")"
  prompt="$PROMPT_IMAGE"
  [[ "$family" == ltx-video || "$family" == ltx2 ]] && prompt="$PROMPT_VIDEO"

  width="$default_width"
  height="$default_height"
  steps="$default_steps"
  if [[ "$family" == qwen-image ]]; then
    width="${MOLD_REGRESSION_QWEN_WIDTH:-1024}"
    height="${MOLD_REGRESSION_QWEN_HEIGHT:-1024}"
  fi
  if [[ "$family" == ltx-video ]]; then
    width="${MOLD_REGRESSION_LTX_VIDEO_WIDTH:-768}"
    height="${MOLD_REGRESSION_LTX_VIDEO_HEIGHT:-512}"
  fi
  [[ -n "$STEPS_IMAGE" && "$family" != ltx-video && "$family" != ltx2 ]] && steps="$STEPS_IMAGE"
  [[ -n "$STEPS_VIDEO" && ( "$family" == ltx-video || "$family" == ltx2 ) ]] && steps="$STEPS_VIDEO"

  if [[ "$family" == ltx-video || "$family" == ltx2 ]]; then
    base_out="$RUN_DIR/${safe_model}.base.mp4"
    run_case "$model" "$family" base "$base_out" "$TIMEOUT_VIDEO" \
      "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
      --output "$base_out" --format mp4 --frames "$FRAMES_VIDEO" --fps "$FPS_VIDEO" \
      --width "$width" --height "$height" --steps "$steps"

    if [[ "$family" == ltx2 ]]; then
      audio_out="$RUN_DIR/${safe_model}.audio.mp4"
      run_case "$model" "$family" audio "$audio_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
        --output "$audio_out" --format mp4 --frames "$FRAMES_VIDEO" --fps "$FPS_VIDEO" \
        --width "$width" --height "$height" --steps "$steps" --audio

      src_out="$RUN_DIR/${safe_model}.source.mp4"
      run_case "$model" "$family" source "$src_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
        --output "$src_out" --format mp4 --frames "$FRAMES_VIDEO" --fps "$FPS_VIDEO" \
        --width "$width" --height "$height" --steps "$steps" --image "$SOURCE_IMAGE"
    fi

    if supports_chain_cases "$model" "$family"; then
      chain_script="$RUN_DIR/${safe_model}.chain.toml"
      write_chain_script "$chain_script" "$model" "$family" "$width" "$height" "$steps" false false
      chain_out="$RUN_DIR/${safe_model}.chain.mp4"
      run_case "$model" "$family" chain "$chain_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" --script "$chain_script" --output "$chain_out"
    fi

    if [[ "$family" == ltx2 ]] && supports_chain_cases "$model" "$family"; then
      chain_source_script="$RUN_DIR/${safe_model}.chain-source.toml"
      write_chain_script "$chain_source_script" "$model" "$family" "$width" "$height" "$steps" false true
      chain_source_out="$RUN_DIR/${safe_model}.chain-source.mp4"
      run_case "$model" "$family" chain-source "$chain_source_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" --script "$chain_source_script" --output "$chain_source_out"

      chain_audio_script="$RUN_DIR/${safe_model}.chain-audio.toml"
      write_chain_script "$chain_audio_script" "$model" "$family" "$width" "$height" "$steps" true false
      chain_audio_out="$RUN_DIR/${safe_model}.chain-audio.mp4"
      run_case "$model" "$family" chain-audio "$chain_audio_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" --script "$chain_audio_script" --output "$chain_audio_out"

      chain_audio_source_script="$RUN_DIR/${safe_model}.chain-audio-source.toml"
      write_chain_script "$chain_audio_source_script" "$model" "$family" "$width" "$height" "$steps" true true
      chain_audio_source_out="$RUN_DIR/${safe_model}.chain-audio-source.mp4"
      run_case "$model" "$family" chain-audio-source "$chain_audio_source_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" --script "$chain_audio_source_script" --output "$chain_audio_source_out"
    fi
  else
    base_out="$RUN_DIR/${safe_model}.base.png"
    run_case "$model" "$family" base "$base_out" "$TIMEOUT_IMAGE" \
      "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
      --output "$base_out" --format png --width "$width" --height "$height" --steps "$steps"

    mapfile -t loras < <(lora_paths_for_model "$model" "$family")
    if (( ${#loras[@]} >= 1 )); then
      lora1_out="$RUN_DIR/${safe_model}.lora1.png"
      run_case "$model" "$family" lora1 "$lora1_out" "$TIMEOUT_IMAGE" \
        "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
        --output "$lora1_out" --format png --width "$width" --height "$height" --steps "$steps" \
        --lora "${loras[0]}"
    fi
    if (( ${#loras[@]} >= 2 )); then
      lora2_out="$RUN_DIR/${safe_model}.lora2.png"
      run_case "$model" "$family" lora2 "$lora2_out" "$TIMEOUT_IMAGE" \
        "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
        --output "$lora2_out" --format png --width "$width" --height "$height" --steps "$steps" \
        --lora "${loras[0]}" --lora "${loras[1]}"
    fi

    src_out="$RUN_DIR/${safe_model}.source.png"
    run_case "$model" "$family" source "$src_out" "$TIMEOUT_IMAGE" \
      "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
      --output "$src_out" --format png --width "$width" --height "$height" --steps "$steps" \
      --image "$SOURCE_IMAGE"
  fi
done < "$RUN_DIR/models.tsv"

echo "all regression cases passed: $RUN_DIR"
