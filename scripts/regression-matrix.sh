#!/usr/bin/env bash
set -euo pipefail

HOST="${MOLD_HOST:-http://127.0.0.1:7680}"
MOLD_BIN="${MOLD_BIN:-target/release/mold}"
OUT_ROOT="${MOLD_REGRESSION_OUT:-$HOME/.mold/regression}"
RUN_ID="${MOLD_REGRESSION_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$OUT_ROOT/$RUN_ID"
LOG="$RUN_DIR/results.jsonl"
ERRORS_LOG="$RUN_DIR/errors.jsonl"
ATTEMPT_ERRORS_LOG="$RUN_DIR/attempt-errors.jsonl"
FAILED_CASES="$RUN_DIR/FAILED_CASES.tsv"
FAILURE_LOCK="$RUN_DIR/failures.lock"
LOG_LOCK="$RUN_DIR/results.lock"
ATTEMPT_LOCK="$RUN_DIR/attempt-errors.lock"
SOURCE_IMAGE="$RUN_DIR/source.png"
SOURCE_IMAGE_B="$RUN_DIR/source-b.png"

PROMPT_IMAGE="${MOLD_REGRESSION_PROMPT_IMAGE:-a small ceramic teapot on a wooden table, soft window light, realistic}"
PROMPT_VIDEO="${MOLD_REGRESSION_PROMPT_VIDEO:-a small ceramic teapot on a wooden table as the camera slowly pushes in, soft window light}"
STEPS_IMAGE="${MOLD_REGRESSION_STEPS_IMAGE:-}"
STEPS_VIDEO="${MOLD_REGRESSION_STEPS_VIDEO:-}"
FRAMES_VIDEO="${MOLD_REGRESSION_FRAMES_VIDEO:-17}"
FPS_VIDEO="${MOLD_REGRESSION_FPS_VIDEO:-12}"
TIMEOUT_IMAGE="${MOLD_REGRESSION_TIMEOUT_IMAGE:-1800}"
TIMEOUT_VIDEO="${MOLD_REGRESSION_TIMEOUT_VIDEO:-3600}"
QUEUE_TIMEOUT_BONUS="${MOLD_REGRESSION_QUEUE_TIMEOUT_BONUS:-86400}"
TRANSIENT_RETRIES="${MOLD_REGRESSION_TRANSIENT_RETRIES:-1}"
MODE="${MOLD_REGRESSION_MODE:-matrix}"
ONLY_MODEL="${MOLD_REGRESSION_ONLY_MODEL:-}"
ONLY_CASE="${MOLD_REGRESSION_ONLY_CASE:-}"
START_AFTER_MODEL="${MOLD_REGRESSION_START_AFTER_MODEL:-}"
START_AFTER_CASE="${MOLD_REGRESSION_START_AFTER_CASE:-}"
MIXED_JOBS="${MOLD_REGRESSION_MIXED_JOBS:-12}"
MIXED_PARALLEL="${MOLD_REGRESSION_MIXED_PARALLEL:-6}"
MATRIX_BATCH="${MOLD_REGRESSION_BATCH:-4}"
INCLUDE_OVER_BUDGET="${MOLD_REGRESSION_INCLUDE_OVER_BUDGET:-0}"

if [[ -n "$START_AFTER_MODEL" || -n "$START_AFTER_CASE" ]]; then
  RESUME_READY=false
else
  RESUME_READY=true
fi

mkdir -p "$RUN_DIR"
: > "$ERRORS_LOG"
: > "$ATTEMPT_ERRORS_LOG"
: > "$FAILED_CASES"

if [[ ! -x "$MOLD_BIN" ]]; then
  echo "mold binary not executable: $MOLD_BIN" >&2
  exit 2
fi

magick -size 1024x1024 gradient:'#c8d8df-#f7ead8' \
  -fill '#8b6b55' -draw 'rectangle 0,675 1024,1024' \
  -fill '#f3f0e8' -stroke '#5b4034' -strokewidth 14 \
  -draw 'ellipse 512,560 235,150 0,360' \
  -draw 'ellipse 512,425 120,46 0,360' \
  -draw 'ellipse 512,360 34,24 0,360' \
  -fill none -stroke '#5b4034' -strokewidth 28 \
  -draw 'arc 300,430 455,665 70,290' \
  -fill '#f3f0e8' -stroke '#5b4034' -strokewidth 12 \
  -draw "path 'M 690,520 C 790,455 900,485 925,565 C 840,565 790,610 700,620 Z'" \
  "$SOURCE_IMAGE"

# A visibly different second still for wan's first/last-frame cases (#779):
# reusing the same image for both endpoints would let a broken last-frame
# pin pass, since the render would look correct either way.
magick -size 1024x1024 gradient:'#1a2740'-'#4d6fa8' \
  -fill '#ffd9a0' -stroke '#3a2d1e' -strokewidth 12 \
  -draw 'roundrectangle 330,300 700,760 40,40' \
  -fill '#3a2d1e' -draw 'ellipse 515,530 90,90 0,360' \
  -fill '#ffd9a0' -draw 'ellipse 515,530 42,42 0,360' \
  "$SOURCE_IMAGE_B"

models_json="$RUN_DIR/models.json"
loras_json="$RUN_DIR/loras.json"
curl -fsS "$HOST/api/models" > "$models_json"
curl -fsS "$HOST/api/catalog/installed?kind=lora" > "$loras_json"

jq -r '
  [
    .[]
    | select(.downloaded == true)
    | select(.family | IN("flux","flux2","sd15","sdxl","sd3","z-image","qwen-image","wuerstchen","ltx-video","ltx2","wan"))
    | select((.family != "ltx2") or (((.description // "") | ascii_downcase | contains("fp4") | not) and ((.description // "") | ascii_downcase | contains("nvfp4") | not)))
  ]
  | sort_by(.name)
  | group_by(.name)
  | .[]
  | select(length == 1)
  | .[0]
  | [.name, .family, (.default_steps|tostring), (.default_width|tostring), (.default_height|tostring), ((.dimension_alignment // 0)|tostring), ((.source_image // "")|tostring)]
  | @tsv
' "$models_json" > "$RUN_DIR/models.tsv"

jq -r '
  [
    .[]
    | select(.downloaded == true)
    | select(.family | IN("flux","flux2","sd15","sdxl","sd3","z-image","qwen-image","wuerstchen","ltx-video","ltx2","wan"))
    | select((.family != "ltx2") or (((.description // "") | ascii_downcase | contains("fp4") | not) and ((.description // "") | ascii_downcase | contains("nvfp4") | not)))
  ]
  | sort_by(.name)
  | group_by(.name)
  | .[]
  | select(length > 1)
  | .[] | [.name, .family] | @tsv
' "$models_json" > "$RUN_DIR/ambiguous-models.tsv"

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

json_error_log() {
  jq -nc \
    --arg ts "$(date --iso-8601=seconds)" \
    --arg status "$1" \
    --arg model "$2" \
    --arg family "$3" \
    --arg case "$4" \
    --arg output "$5" \
    --arg cmd "$6" \
    --arg stdout_path "$7" \
    --arg stderr_path "$8" \
    --arg stderr_tail "$9" \
    '{ts:$ts,status:$status,model:$model,family:$family,case:$case,output:$output,cmd:$cmd,stdout_path:$stdout_path,stderr_path:$stderr_path,stderr_tail:$stderr_tail}'
}

append_log() {
  (
    flock 9
    json_log "$@" >> "$LOG"
  ) 9>"$LOG_LOCK"
}

append_attempt_error_log() {
  (
    flock 9
    json_error_log "$@" >> "$ATTEMPT_ERRORS_LOG"
  ) 9>"$ATTEMPT_LOCK"
}

record_failure() {
  local status="$1" model="$2" family="$3" case_name="$4" output="$5" cmd_text="$6" stdout_path="$7" stderr_path="$8" artifact="$9"
  local command_path="$RUN_DIR/$artifact.command"
  local stderr_tail
  stderr_tail="$(tail -n 80 "$stderr_path" 2> /dev/null || true)"
  printf '%s\n' "$cmd_text" > "$command_path"
  (
    flock 9
    json_error_log "$status" "$model" "$family" "$case_name" "$output" "$cmd_text" "$stdout_path" "$stderr_path" "$stderr_tail" >> "$ERRORS_LOG"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$model" "$family" "$case_name" "$status" "$output" "$command_path" "$stderr_path" >> "$FAILED_CASES"
    if [[ ! -e "$RUN_DIR/FAILED_COMMAND.txt" ]]; then
      cp "$command_path" "$RUN_DIR/FAILED_COMMAND.txt"
      cp "$stderr_path" "$RUN_DIR/FAILED_STDERR.txt"
    fi
  ) 9>"$FAILURE_LOCK"
}

is_transient_attempt_failure() {
  local stderr_path="$1"
  grep -Fq 'CUDA_ERROR_INVALID_VALUE' "$stderr_path"
}

case_selected() {
  local model="$1" case_name="$2"
  if [[ -n "$ONLY_MODEL" && "$model" != "$ONLY_MODEL" ]]; then
    return 1
  fi
  if [[ -n "$ONLY_CASE" && "$case_name" != "$ONLY_CASE" ]]; then
    return 1
  fi
  if [[ "$RESUME_READY" != true ]]; then
    if [[ "$model" == "$START_AFTER_MODEL" && ( -z "$START_AFTER_CASE" || "$case_name" == "$START_AFTER_CASE" ) ]]; then
      RESUME_READY=true
    fi
    return 1
  fi
  return 0
}

execute_case() {
  local model="$1" family="$2" case_name="$3" output="$4" timeout_s="$5"
  shift 5
  local -a cmd=("$@")
  local cmd_text artifact stdout_path stderr_path effective_timeout_s code
  printf -v cmd_text '%q ' "${cmd[@]}"
  artifact="$(case_stem "$family" "$model").$case_name"
  stdout_path="$RUN_DIR/$artifact.stdout"
  stderr_path="$RUN_DIR/$artifact.stderr"
  effective_timeout_s=$((timeout_s + QUEUE_TIMEOUT_BONUS))
  echo "RUN $case_name $model -> $output"
  append_log start "$model" "$family" "$case_name" "$output" "$cmd_text"
  local attempt=0
  while true; do
    if timeout "$effective_timeout_s" "${cmd[@]}" > "$stdout_path" 2> "$stderr_path"; then
      if [[ ! -s "$output" ]]; then
        echo "empty output for $case_name $model: $output" >&2
        append_log empty-output "$model" "$family" "$case_name" "$output" "$cmd_text"
        record_failure empty-output "$model" "$family" "$case_name" "$output" "$cmd_text" "$stdout_path" "$stderr_path" "$artifact"
        return 1
      fi
      append_log ok "$model" "$family" "$case_name" "$output" "$cmd_text"
      return 0
    else
      code=$?
      local stderr_tail
      stderr_tail="$(tail -n 80 "$stderr_path" 2> /dev/null || true)"
      append_attempt_error_log "attempt-failed:$code" "$model" "$family" "$case_name" "$output" "$cmd_text" "$stdout_path" "$stderr_path" "$stderr_tail"
      if (( attempt < TRANSIENT_RETRIES )) && is_transient_attempt_failure "$stderr_path"; then
        attempt=$((attempt + 1))
        echo "RETRY $case_name $model after transient CUDA failure (attempt $attempt/$TRANSIENT_RETRIES)" >&2
        append_log "retry:$code" "$model" "$family" "$case_name" "$output" "$cmd_text"
        sleep 2
        continue
      fi

      echo "FAILED $case_name $model (exit $code)" >&2
      append_log "failed:$code" "$model" "$family" "$case_name" "$output" "$cmd_text"
      record_failure "failed:$code" "$model" "$family" "$case_name" "$output" "$cmd_text" "$stdout_path" "$stderr_path" "$artifact"
      return "$code"
    fi
  done
}

queue_case() {
  local model="$1" family="$2" case_name="$3"
  if ! case_selected "$model" "$case_name"; then
    return 0
  fi
  execute_case "$@" &
  MATRIX_ACTIVE=$((MATRIX_ACTIVE + 1))
}

MATRIX_ACTIVE=0

wait_for_matrix() {
  local failures=0
  while (( MATRIX_ACTIVE > 0 )); do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    MATRIX_ACTIVE=$((MATRIX_ACTIVE - 1))
  done
  if (( failures > 0 )); then
    echo "matrix failed with $failures failed case(s): $RUN_DIR" >&2
    echo "failed cases: $FAILED_CASES" >&2
    echo "error details: $ERRORS_LOG" >&2
    exit 1
  fi
}

validate_run_log_complete() {
  local starts terminals duplicate_outputs missing_outputs
  starts="$(jq -s 'map(select(.status == "start")) | length' "$LOG")"
  terminals="$(jq -s 'map(select(.status == "ok" or .status == "empty-output" or (.status | startswith("failed:")))) | length' "$LOG")"
  if [[ "$starts" != "$terminals" ]]; then
    echo "matrix log incomplete: starts=$starts terminal=$terminals ($LOG)" >&2
    exit 1
  fi
  duplicate_outputs="$(jq -r 'select(.status == "ok") | .output' "$LOG" | sort | uniq -d)"
  if [[ -n "$duplicate_outputs" ]]; then
    echo "matrix produced duplicate output paths:" >&2
    printf '%s\n' "$duplicate_outputs" >&2
    exit 1
  fi
  missing_outputs="$(jq -r 'select(.status == "ok") | .output' "$LOG" | while IFS= read -r output; do test -s "$output" || printf '%s\n' "$output"; done)"
  if [[ -n "$missing_outputs" ]]; then
    echo "matrix has ok rows with missing outputs:" >&2
    printf '%s\n' "$missing_outputs" >&2
    exit 1
  fi
}

run_mixed_job() {
  local idx="$1" model="$2" family="$3" case_name="$4" output="$5" timeout_s="$6"
  shift 6
  local -a cmd=("$@")
  local cmd_text effective_timeout_s
  printf -v cmd_text '%q ' "${cmd[@]}"
  effective_timeout_s=$((timeout_s + QUEUE_TIMEOUT_BONUS))
  echo "MIXED RUN $idx $case_name $model -> $output"
  append_log start "$model" "$family" "mixed-$idx-$case_name" "$output" "$cmd_text"
  if timeout "$effective_timeout_s" "${cmd[@]}" > "$RUN_DIR/mixed-${idx}.stdout" 2> "$RUN_DIR/mixed-${idx}.stderr"; then
    if [[ ! -s "$output" ]]; then
      echo "empty output for mixed job $idx $case_name $model: $output" >&2
      append_log empty-output "$model" "$family" "mixed-$idx-$case_name" "$output" "$cmd_text"
      return 1
    fi
    append_log ok "$model" "$family" "mixed-$idx-$case_name" "$output" "$cmd_text"
  else
    local code=$?
    echo "FAILED mixed job $idx $case_name $model (exit $code)" >&2
    echo "$cmd_text" > "$RUN_DIR/FAILED_COMMAND.txt"
    cp "$RUN_DIR/mixed-${idx}.stderr" "$RUN_DIR/FAILED_STDERR.txt"
    append_log "failed:$code" "$model" "$family" "mixed-$idx-$case_name" "$output" "$cmd_text"
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
  if [[ "$family" == z-image ]]; then
    return 0
  fi
  awk -F '\t' -v family="$family" -v model="$model" '
    $1 != family { next }
    family == "flux2" && $2 == "klein-9b" && model !~ /9b|cv:2669986|cv:2650565|cv:2805234|cv:2765147|cv:2663677|cv:2759597/ { next }
    { print $3 }
  ' "$RUN_DIR/loras.tsv" | head -2
}

sanitize() {
  tr ':/ ' '---' <<< "$1"
}

case_stem() {
  local family="$1" model="$2"
  sanitize "$family.$model"
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
    IFS=$'\t' read -r _ _ default_steps default_width default_height _ _ <<< "$row"
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
    safe_model="$(case_stem "$family" "$model")"
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

while IFS=$'\t' read -r model family default_steps default_width default_height dim_align source_image; do
  safe_model="$(case_stem "$family" "$model")"
  prompt="$PROMPT_IMAGE"
  [[ "$family" == ltx-video || "$family" == ltx2 || "$family" == wan ]] && prompt="$PROMPT_VIDEO"

  width="$default_width"
  height="$default_height"
  steps="$default_steps"
  if [[ "$family" == qwen-image ]]; then
    width="${MOLD_REGRESSION_QWEN_WIDTH:-1024}"
    height="${MOLD_REGRESSION_QWEN_HEIGHT:-1024}"
  fi
  if [[ "$family" == z-image ]]; then
    width="${MOLD_REGRESSION_ZIMAGE_WIDTH:-768}"
    height="${MOLD_REGRESSION_ZIMAGE_HEIGHT:-768}"
  fi
  if [[ "$family" == ltx-video ]]; then
    width="${MOLD_REGRESSION_LTX_VIDEO_WIDTH:-768}"
    height="${MOLD_REGRESSION_LTX_VIDEO_HEIGHT:-512}"
  fi
  # Wan sizes from what the checkpoint itself advertises: TI2V-5B needs a /32
  # grid where the A14B tiers take /16, and a family constant would put one
  # of them off-grid. The advertised alignment is authoritative; the width
  # and height already came from the same row.
  if [[ "$family" == wan && "${dim_align:-0}" -gt 0 ]]; then
    width=$(( (width / dim_align) * dim_align ))
    height=$(( (height / dim_align) * dim_align ))
  fi

  [[ -n "$STEPS_IMAGE" && "$family" != ltx-video && "$family" != ltx2 && "$family" != wan ]] && steps="$STEPS_IMAGE"
  [[ -n "$STEPS_VIDEO" && ( "$family" == ltx-video || "$family" == ltx2 ) ]] && steps="$STEPS_VIDEO"
  # Deliberately NOT step-flattened (#790): wan's tiers ARE the recipe — the
  # 4-step Lightning tiers and the 20-step quality tiers exercise different
  # code (distill branch vs none, CFG vs single forward). Overriding steps
  # would collapse them into one case that tests neither.

  if [[ "$family" == wan ]]; then
    # The conditioning contract the server advertises decides the cases: a
    # T2V-only checkpoint has no source case to run, and an I2V-required one
    # cannot render a bare base case at all (#772).
    #
    # An ABSENT contract means unknown — an older server, or a checkpoint
    # whose headers this build could not classify — not "unsupported".
    # Guessing either way produces a misleading result: assume T2V and an
    # I2V-required checkpoint fails for missing input; assume I2V and a
    # T2V-only one fails for supplying it. Skip the model and say so, so the
    # run reports a coverage gap rather than a fake regression.
    if [[ -z "$source_image" ]]; then
      printf 'skipping %s: server advertises no source_image contract (unknown, not unsupported)\n' \
        "$model" >&2
      printf '%s\t%s\tunclassified-source-contract\n' "$model" "$family" \
        >> "$RUN_DIR/SKIPPED_MODELS.tsv"
      continue
    fi

    if [[ "$source_image" != "required" ]]; then
      base_out="$RUN_DIR/${safe_model}.base.mp4"
      queue_case "$model" "$family" base "$base_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
        --output "$base_out" --format mp4 --frames "$FRAMES_VIDEO" --fps "$FPS_VIDEO" \
        --width "$width" --height "$height" --steps "$steps"
    fi

    if [[ "$source_image" == "required" || "$source_image" == "optional" ]]; then
      src_out="$RUN_DIR/${safe_model}.source.mp4"
      queue_case "$model" "$family" source "$src_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
        --output "$src_out" --format mp4 --frames "$FRAMES_VIDEO" --fps "$FPS_VIDEO" \
        --width "$width" --height "$height" --steps "$steps" --image "$SOURCE_IMAGE"

      # First/last-frame (#779). The nine-frame floor is TI2V's alone — it
      # pins both endpoints in latent space, so a shorter clip leaves nothing
      # to denoise. A14B concatenates its conditioning and happily takes any
      # valid 4k+1 clip, so applying the floor family-wide would silently
      # drop its endpoint case at a small FRAMES_VIDEO.
      flf_min=1
      [[ "$model" == wan22-ti2v-5b* ]] && flf_min=9
      if [[ "$FRAMES_VIDEO" -ge "$flf_min" ]]; then
        flf_out="$RUN_DIR/${safe_model}.flf.mp4"
        queue_case "$model" "$family" flf "$flf_out" "$TIMEOUT_VIDEO" \
          "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
          --output "$flf_out" --format mp4 --frames "$FRAMES_VIDEO" --fps "$FPS_VIDEO" \
          --width "$width" --height "$height" --steps "$steps" \
          --image "$SOURCE_IMAGE" --last-image "$SOURCE_IMAGE_B"
      fi
    fi

    # Single-frame still (#798): wan is the only video family that renders a
    # PNG, and the format default flips with the frame count.
    if [[ "$source_image" != "required" ]]; then
      still_out="$RUN_DIR/${safe_model}.still.png"
      queue_case "$model" "$family" still "$still_out" "$TIMEOUT_IMAGE" \
        "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
        --output "$still_out" --format png --frames 1 \
        --width "$width" --height "$height" --steps "$steps"
    fi
  fi

  if [[ "$family" == ltx-video || "$family" == ltx2 ]]; then
    base_out="$RUN_DIR/${safe_model}.base.mp4"
    queue_case "$model" "$family" base "$base_out" "$TIMEOUT_VIDEO" \
      "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
      --output "$base_out" --format mp4 --frames "$FRAMES_VIDEO" --fps "$FPS_VIDEO" \
      --width "$width" --height "$height" --steps "$steps"

    if [[ "$family" == ltx2 ]]; then
      audio_out="$RUN_DIR/${safe_model}.audio.mp4"
      queue_case "$model" "$family" audio "$audio_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
        --output "$audio_out" --format mp4 --frames "$FRAMES_VIDEO" --fps "$FPS_VIDEO" \
        --width "$width" --height "$height" --steps "$steps" --audio

      src_out="$RUN_DIR/${safe_model}.source.mp4"
      queue_case "$model" "$family" source "$src_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
        --output "$src_out" --format mp4 --frames "$FRAMES_VIDEO" --fps "$FPS_VIDEO" \
        --width "$width" --height "$height" --steps "$steps" --image "$SOURCE_IMAGE"
    fi

    if supports_chain_cases "$model" "$family"; then
      chain_script="$RUN_DIR/${safe_model}.chain.toml"
      write_chain_script "$chain_script" "$model" "$family" "$width" "$height" "$steps" false false
      chain_out="$RUN_DIR/${safe_model}.chain.mp4"
      queue_case "$model" "$family" chain "$chain_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" --script "$chain_script" --output "$chain_out"
    fi

    if [[ "$family" == ltx2 ]] && supports_chain_cases "$model" "$family"; then
      chain_source_script="$RUN_DIR/${safe_model}.chain-source.toml"
      write_chain_script "$chain_source_script" "$model" "$family" "$width" "$height" "$steps" false true
      chain_source_out="$RUN_DIR/${safe_model}.chain-source.mp4"
      queue_case "$model" "$family" chain-source "$chain_source_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" --script "$chain_source_script" --output "$chain_source_out"

      chain_audio_script="$RUN_DIR/${safe_model}.chain-audio.toml"
      write_chain_script "$chain_audio_script" "$model" "$family" "$width" "$height" "$steps" true false
      chain_audio_out="$RUN_DIR/${safe_model}.chain-audio.mp4"
      queue_case "$model" "$family" chain-audio "$chain_audio_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" --script "$chain_audio_script" --output "$chain_audio_out"

      chain_audio_source_script="$RUN_DIR/${safe_model}.chain-audio-source.toml"
      write_chain_script "$chain_audio_source_script" "$model" "$family" "$width" "$height" "$steps" true true
      chain_audio_source_out="$RUN_DIR/${safe_model}.chain-audio-source.mp4"
      queue_case "$model" "$family" chain-audio-source "$chain_audio_source_out" "$TIMEOUT_VIDEO" \
        "$MOLD_BIN" run --host "$HOST" --script "$chain_audio_source_script" --output "$chain_audio_source_out"
    fi
  elif [[ "$family" != wan ]]; then
    # wan is handled by its own arm above. Without this guard it would ALSO
    # take the image-family cases, queueing a bare PNG base case that an
    # I2V-required checkpoint rejects outright and a duplicate still.
    base_out="$RUN_DIR/${safe_model}.base.png"
    queue_case "$model" "$family" base "$base_out" "$TIMEOUT_IMAGE" \
      "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
      --output "$base_out" --format png --width "$width" --height "$height" --steps "$steps"

    mapfile -t loras < <(lora_paths_for_model "$model" "$family")
    if (( ${#loras[@]} >= 1 )); then
      lora1_out="$RUN_DIR/${safe_model}.lora1.png"
      queue_case "$model" "$family" lora1 "$lora1_out" "$TIMEOUT_IMAGE" \
        "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
        --output "$lora1_out" --format png --width "$width" --height "$height" --steps "$steps" \
        --lora "${loras[0]}"
    fi
    if (( ${#loras[@]} >= 2 )) && [[ "$family" != z-image ]]; then
      lora2_out="$RUN_DIR/${safe_model}.lora2.png"
      queue_case "$model" "$family" lora2 "$lora2_out" "$TIMEOUT_IMAGE" \
        "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
        --output "$lora2_out" --format png --width "$width" --height "$height" --steps "$steps" \
        --lora "${loras[0]}" --lora "${loras[1]}"
    fi

    src_width="$width"
    src_height="$height"
    if [[ "$family" == sdxl ]]; then
      src_width="${MOLD_REGRESSION_SDXL_SOURCE_WIDTH:-768}"
      src_height="${MOLD_REGRESSION_SDXL_SOURCE_HEIGHT:-768}"
    fi

    src_out="$RUN_DIR/${safe_model}.source.png"
    queue_case "$model" "$family" source "$src_out" "$TIMEOUT_IMAGE" \
      "$MOLD_BIN" run --host "$HOST" "$model" "$prompt" \
      --output "$src_out" --format png --width "$src_width" --height "$src_height" --steps "$steps" \
      --image "$SOURCE_IMAGE"
  fi
done < "$RUN_DIR/models.tsv"

wait_for_matrix
validate_run_log_complete

echo "all regression cases passed: $RUN_DIR"
