#!/usr/bin/env bash
set -euo pipefail

# LTX-2.5 CUDA qualification harness (#1398 official split packs, #1414 GGUF
# tiers). Two modes over one evidence directory:
#
#   --run   execute every scripts/fixtures/ltx25-cuda-matrix.json row whose
#           `profile` matches LTX25_SERVER_PROFILE against the scratch server
#           at MOLD_HOST, writing rows/<id>/{manifest.json,stdout.log,
#           server.log,request.json,ffprobe.json,vram.csv,host.csv,compute.csv}
#   --seal  (default) re-validate every retained row, model asset, reference
#           clone, ComfyUI oracle manifest, and gate, then write the combined
#           mold.ltx25.cuda.verification.v1 report and run its validator.
#
# The Metal capture (capture-ltx25-metal-verification.sh) is the shape this
# follows; the CUDA differences are structural rather than string swaps: the
# model store is a separate MOLD_MODELS_DIR, the asset list is generated from
# the manifest table (scripts/fixtures/ltx25-assets.json), provenance is read
# from the SERVER log slice each row produced plus the Library row's
# additive metadata, and every row lands in one of exactly four states.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode=seal
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) mode=run ;;
    --seal) mode=seal ;;
    -h | --help)
      sed -n '4,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 64
      ;;
  esac
  shift
done

mold_home="${MOLD_HOME:-/mnt/storage20tb/AI/mold}"
models_dir="${MOLD_MODELS_DIR:-}"
references_root="${LTX25_REFERENCES_ROOT:-$repo_root/tmp}"
campaign="${LTX25_CAMPAIGN:-current}"
verification_root="${LTX25_VERIFICATION_ROOT:-$mold_home/output/verification/ltx-2.5/cuda}"
report="${LTX25_REPORT:-$verification_root/ltx25-cuda-verification-$campaign.json}"
evidence_dir="${report%.json}.d"
rows_dir="$evidence_dir/rows"
matrix="${LTX25_MATRIX:-$repo_root/scripts/fixtures/ltx25-cuda-matrix.json}"
assets_fixture="${LTX25_ASSETS:-$repo_root/scripts/fixtures/ltx25-assets.json}"
schema="$repo_root/docs/qualification/ltx25-cuda-verification.schema.json"
validator="$repo_root/scripts/validate-ltx25-cuda-report.py"
skip_gates="${LTX25_SKIP_GATES:-0}"
contract_test="${LTX25_CONTRACT_TEST:-0}"
database="${MOLD_DB_PATH:-$mold_home/mold.db}"
mold_bin="${LTX25_MOLD_BIN:-}"
build_json="${LTX25_BUILD_JSON:-}"
runtime_env_source="${LTX25_RUNTIME_ENV_SOURCE:-$repo_root/crates/mold-inference/src/runtime_env.rs}"
row_timeout="${LTX25_ROW_TIMEOUT:-7200}"
# halcyon's sealed Metal evidence, copied to this host for the A/B rows.
metal_reference_root="${LTX25_METAL_REFERENCE_ROOT:-/storage-fast/mold/uat-1398/metal-reference}"
ab_python="${LTX25_AB_PYTHON:-python3}"
ab_script="$repo_root/scripts/ltx25-metal-ab.py"

fail() {
  echo "LTX-2.5 CUDA verification failed: $*" >&2
  exit 1
}

for command in ffprobe git jq python3 sha256sum; do
  command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
done
if [[ -z "${LTX25_HOST_JSON:-}" ]]; then
  for command in nvidia-smi nvcc free; do
    command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
  done
fi

if [[ "$contract_test" == 1 ]]; then
  [[ "${LTX25_ALLOW_TEST_HOME:-0}" == 1 && "$skip_gates" == 1 ]] \
    || fail "contract test mode requires isolated home and skipped gates"
else
  [[ "${LTX25_ALLOW_TEST_HOME:-0}" != 1 && "$skip_gates" != 1 ]] \
    || fail "test overrides require LTX25_CONTRACT_TEST=1"
  [[ -z "${LTX25_HOST_JSON:-}" ]] \
    || fail "injected host JSON requires LTX25_CONTRACT_TEST=1"
  [[ "$runtime_env_source" == "$repo_root/crates/mold-inference/src/runtime_env.rs" ]] \
    || fail "an injected engine-shaping registry requires LTX25_CONTRACT_TEST=1"
  [[ "$mold_home" == /mnt/storage20tb/AI/mold ]] \
    || fail "MOLD_HOME must be /mnt/storage20tb/AI/mold"
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] \
    || fail "runtime capture is restricted to Linux x86_64 CUDA"
  [[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=normal)" ]] \
    || fail "qualification requires a clean source tree; commit the exact code before capture"
fi
[[ -n "$models_dir" ]] || fail "MOLD_MODELS_DIR is required: the model store is separate from MOLD_HOME"
[[ -d "$models_dir" ]] || fail "MOLD_MODELS_DIR does not exist: $models_dir"
[[ "${models_dir%/}" != "${mold_home%/}/models" ]] \
  || fail "MOLD_MODELS_DIR must not be MOLD_HOME/models; the production store is a separate dataset"
[[ -s "$matrix" ]] || fail "missing matrix fixture: $matrix"
[[ -s "$assets_fixture" ]] || fail "missing asset fixture: $assets_fixture"
jq -e '.schema_version == "mold.ltx25.cuda.matrix.v1"' "$matrix" >/dev/null \
  || fail "matrix fixture schema mismatch"
jq -e '.schema_version == "mold.ltx25.assets.v1"' "$assets_fixture" >/dev/null \
  || fail "asset fixture schema mismatch"

source_commit="$(git -C "$repo_root" rev-parse HEAD)"
mkdir -p "$verification_root" "$evidence_dir" "$rows_dir"

file_size() {
  stat -c '%s' "$1"
}

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

# --- binary and build identity ----------------------------------------------
# `which mold` inside the devshell is the LINKER; the qualified binary must be
# named explicitly and must identify the exact source commit being sealed.
[[ -n "$mold_bin" ]] || fail "LTX25_MOLD_BIN is required (absolute path to the built mold binary)"
[[ "$mold_bin" == /* && -x "$mold_bin" ]] || fail "LTX25_MOLD_BIN must be an absolute executable path"
binary_version="$("$mold_bin" version 2>/dev/null || true)"
if [[ "$binary_version" =~ \(([0-9a-f]{7,40})[[:space:]] ]]; then
  binary_commit="${BASH_REMATCH[1]}"
else
  fail "$mold_bin version output carries no source commit: $binary_version"
fi
[[ "${source_commit:0:${#binary_commit}}" == "$binary_commit" ]] \
  || fail "$mold_bin was built from $binary_commit, not source commit $source_commit"
[[ -n "$build_json" && -s "$build_json" ]] \
  || fail "LTX25_BUILD_JSON is required (features, candle_rev, cargo_command, git_sha of the build)"
jq -e '(.features | type == "array" and length > 0)
  and (.candle_rev | test("^[0-9a-f]{40}$"))
  and (.cargo_command | length > 0)
  and (.git_sha | test("^[0-9a-f]{40}$"))' "$build_json" >/dev/null \
  || fail "LTX25_BUILD_JSON must carry features[], candle_rev, cargo_command, and git_sha"
[[ "$(jq -r '.git_sha' "$build_json")" == "$source_commit" ]] \
  || fail "LTX25_BUILD_JSON.git_sha is not the source commit"
cargo_lock_candle_rev="$(grep -Eo 'utensils/candle\.git\?rev=[0-9a-f]{40}' "$repo_root/Cargo.lock" \
  | head -1 | sed 's/.*rev=//')"
[[ -n "$cargo_lock_candle_rev" ]] || fail "Cargo.lock does not pin the utensils/candle fork"
[[ "$(jq -r '.candle_rev' "$build_json")" == "$cargo_lock_candle_rev" ]] \
  || fail "LTX25_BUILD_JSON.candle_rev disagrees with Cargo.lock ($cargo_lock_candle_rev)"
build="$(jq -c --arg path "$mold_bin" --arg sha "$(file_sha256 "$mold_bin")" \
  --arg version "$binary_version" --arg lock_rev "$cargo_lock_candle_rev" \
  '{binary_path: $path, binary_sha256: $sha, version: $version, features: .features,
    candle_rev: .candle_rev, cargo_lock_candle_rev: $lock_rev,
    cargo_command: .cargo_command, git_sha: .git_sha}' "$build_json")"

# --- host ---------------------------------------------------------------------
host='{}'
if [[ -n "${LTX25_HOST_JSON:-}" ]]; then
  host="$(jq -c --argjson build "$build" '. + {build: $build}' <<<"$LTX25_HOST_JSON")"
else
  gpus_csv="$(nvidia-smi --query-gpu=uuid,name,compute_cap,driver_version,memory.total \
    --format=csv,noheader,nounits)" || fail "failed to inspect CUDA devices"
  gpus="$(jq -Rn '
    def trim: gsub("^[[:space:]]+|[[:space:]]+$"; "");
    [inputs | select(length > 0) | split(",") | select(length == 5)
      | {uuid: (.[0] | trim), name: (.[1] | trim), compute_capability: (.[2] | trim),
         driver_version: (.[3] | trim), memory_total_mib: (.[4] | trim | tonumber)}]' <<<"$gpus_csv")"
  [[ "$(jq 'length' <<<"$gpus")" -gt 0 ]] || fail "nvidia-smi reported no CUDA devices"
  if [[ -n "${LTX25_EXPECT_COMPUTE_CAP:-}" ]]; then
    jq -e --arg cap "$LTX25_EXPECT_COMPUTE_CAP" 'all(.[]; .compute_capability == $cap)' <<<"$gpus" >/dev/null \
      || fail "every qualification device must report compute capability $LTX25_EXPECT_COMPUTE_CAP"
  fi
  nvcc_line="$(nvcc --version | tail -1)"
  host_ram="$(free -b | awk '/^Mem:/ {print $2}')"
  zfs_arc_max=null
  if [[ -r /sys/module/zfs/parameters/zfs_arc_max ]]; then
    zfs_arc_max="$(tr -d '[:space:]' </sys/module/zfs/parameters/zfs_arc_max)"
  fi
  host="$(jq -n --arg kernel "$(uname -r)" --arg arch "$(uname -m)" --arg hostname "$(hostname)" \
    --argjson gpus "$gpus" --arg nvcc "$nvcc_line" --argjson host_ram "$host_ram" \
    --argjson zfs_arc_max "$zfs_arc_max" --argjson build "$build" \
    '{os: "Linux", kernel: $kernel, arch: $arch, hostname: $hostname, gpus: $gpus, nvcc: $nvcc,
      driver_version: $gpus[0].driver_version, host_ram_bytes: $host_ram,
      zfs_arc_max_bytes: $zfs_arc_max, build: $build}')"
fi

# --- Library access (python sqlite3 module; this host ships no sqlite3 CLI) --
db_query_generation() {
  local filename="$1" output_dir="$2"
  [[ -s "$database" ]] || fail "missing Library database: $database"
  python3 - "$database" "$filename" "$output_dir" <<'PY'
import json, sqlite3, sys
db, filename, output_dir = sys.argv[1:4]
conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT id, filename, output_dir, format, prompt, model, seed, steps, guidance, width,"
    " height, frames, fps, generation_time_ms, backend, hostname, source, metadata_synthetic,"
    " file_size_bytes, metadata_json FROM generations WHERE filename = ? AND output_dir = ?",
    (filename, output_dir)).fetchall()
if len(rows) == 1:
    print(json.dumps(dict(rows[0])))
PY
}

db_dump_scheduler_estimates() {
  [[ -s "$database" ]] || fail "missing Library database: $database"
  python3 - "$database" <<'PY'
import json, sqlite3, sys
conn = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT estimate_key, device_class, model_family, model_fingerprint, work_kind, shape_bucket,"
    " sample_count, ewma_total_ms, ewma_load_ms, ewma_prompt_encode_ms, ewma_denoise_ms,"
    " ewma_vae_ms, ewma_upscale_ms, vram_high_water_bytes, host_high_water_bytes,"
    " last_observed_at FROM scheduler_estimates ORDER BY last_observed_at DESC LIMIT 40").fetchall()
print(json.dumps([dict(row) for row in rows], indent=2))
PY
}

# --- matrix helpers -----------------------------------------------------------
row_json() {
  jq -c --arg id "$1" '.rows[] | select(.id == $id)' "$matrix"
}

media_extension() {
  case "$1" in
    apng) echo png ;;
    mp4) echo mp4 ;;
    wav) echo wav ;;
    *) fail "unknown media format: $1" ;;
  esac
}

# ffprobe the media and check it against the row's `expect` block. Prints the
# failure reason and returns 1 on mismatch.
probe_media_against_expect() {
  local media="$1" probe="$2" row="$3"
  ffprobe -v error -count_frames -show_entries \
    format=filename,size,duration:stream=index,codec_name,profile,codec_type,width,height,pix_fmt,r_frame_rate,nb_frames,nb_read_frames,sample_rate,channels \
    -of json "$media" >"$probe" || {
    echo "ffprobe could not decode $media"
    return 1
  }
  local expect audio
  expect="$(jq -c '.expect' <<<"$row")"
  audio="$(jq -r '.expect.audio' <<<"$row")"
  if [[ "$audio" == only ]]; then
    jq -e '([.streams[] | select(.codec_type == "video")] | length == 0)
      and (.streams[] | select(.codec_type == "audio" and .sample_rate == "48000" and .channels == 2))' \
      "$probe" >/dev/null || {
      echo "audio-only output is not stereo 48 kHz without a video stream"
      return 1
    }
    return 0
  fi
  jq -e --argjson expect "$expect" '
    (.streams[] | select(.codec_type == "video")) as $video
    | (($video.nb_frames // $video.nb_read_frames) | tonumber) as $frames
    | ($expect.width == null or $video.width == $expect.width)
    and ($expect.height == null or $video.height == $expect.height)
    and ($expect.fps == null or $video.r_frame_rate == (($expect.fps | tostring) + "/1"))
    and (if ($expect.frames | type) == "number" then $frames == $expect.frames
       elif ($expect.frames | type) == "object" then
         ($frames >= $expect.frames.min and $frames <= $expect.frames.max
          and (($frames - $expect.frames.offset) % $expect.frames.grid) == 0)
       else true end)' "$probe" >/dev/null || {
    echo "decoded video does not match the expected canvas, fps, or frame count"
    return 1
  }
  if [[ "$audio" == true ]]; then
    jq -e '.streams[] | select(.codec_type == "audio" and .codec_name == "aac"
      and .sample_rate == "48000" and .channels == 2)' "$probe" >/dev/null || {
      echo "missing stereo 48 kHz AAC"
      return 1
    }
  else
    jq -e '[.streams[] | select(.codec_type == "audio")] | length == 0' "$probe" >/dev/null || {
      echo "unexpected audio stream on a silent render"
      return 1
    }
  fi
}

# Check the Library row for a passed render. Prints the row on success and
# the failure reason on stderr otherwise.
library_row_for() {
  local media="$1" row="$2"
  local filename output_dir generation
  filename="$(basename "$media")"
  output_dir="$(dirname "$media")"
  generation="$(db_query_generation "$filename" "$output_dir")"
  [[ -n "$generation" ]] || {
    echo "no unique Library row for $media" >&2
    return 1
  }
  jq -e --argjson row "$row" --argjson bytes "$(file_size "$media")" '
    .prompt == $row.prompt and .model == $row.model and .seed == $row.seed
    and .format == $row.format and .backend == "cuda" and .source == "cli"
    and .metadata_synthetic == 0 and .file_size_bytes == $bytes
    and ((.metadata_json | fromjson) as $meta
      | $meta.output_format == $row.format
      and ($row.expect.audio == "only" or $meta.enable_audio == $row.expect.audio)
      and (($row.expect.metadata // {}) | to_entries | all(. as $entry | $meta[$entry.key] == $entry.value)))
    and ($row.expect.width == null or .width == $row.expect.width)
    and ($row.expect.height == null or .height == $row.expect.height)
    and ($row.expect.fps == null or .fps == $row.expect.fps)
    and (($row.expect.frames | type) != "number" or .frames == $row.expect.frames)
  ' <<<"$generation" >/dev/null || {
    echo "Library row provenance mismatch for $media" >&2
    return 1
  }
  echo "$generation"
}

generator_commit_of() {
  local generation="$1" version commit
  version="$(jq -er '.metadata_json | fromjson | .version' <<<"$generation")" || {
    echo "generation metadata has no version" >&2
    return 1
  }
  if [[ "$version" =~ \(([0-9a-f]{7,40})[[:space:]] ]]; then
    commit="${BASH_REMATCH[1]}"
  else
    echo "generation metadata has no source commit: $version" >&2
    return 1
  fi
  [[ "${source_commit:0:${#commit}}" == "$commit" ]] || {
    echo "media was generated by $commit, not source commit $source_commit" >&2
    return 1
  }
  echo "$commit"
}

# Every expected provenance line must be in the row's server-log slice, except
# the shared dispatcher's once-per-process line, which may sit anywhere in
# that profile's full server log. Each process-scoped match is copied into the
# row's own `server-process.log` so the evidence the seal relied on is itself
# retained and hash-bound — the full server log is unhashed and mutable.
# Prints the observed list as JSON.
# The dispatcher emits its line with `backend` as a STRUCTURED tracing field
# ("message":"attention backend selected","backend":"Math"), so a JSON log
# never contains the flat pinned spelling; this regex matches both forms.
dispatcher_line_regex() {
  local line="$1"
  local backend="${line##*backend=}"
  printf '"message":"attention backend selected".*"backend":"%s"|attention backend selected backend=%s' \
    "$backend" "$backend"
}

observe_provenance() {
  local slice="$1" full="$2" expected="$3" dir="$4" observed='[]' line scope regex
  rm -f "$dir/server-process.log"
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    if [[ "$line" == "attention backend selected backend="* ]]; then
      regex="$(dispatcher_line_regex "$line")"
      if grep -Eq -- "$regex" "$slice"; then
        scope=slice
      elif [[ -n "$full" && -f "$full" ]] && grep -Eq -- "$regex" "$full"; then
        scope=process
        grep -E -- "$regex" "$full" | head -1 >>"$dir/server-process.log"
      else
        echo "expected provenance line is absent from the retained server log: $line" >&2
        return 1
      fi
    elif grep -Fq -- "$line" "$slice"; then
      scope=slice
    else
      echo "expected provenance line is absent from the retained server log: $line" >&2
      return 1
    fi
    observed="$(jq -c --arg line "$line" --arg scope "$scope" '. + [{line: $line, scope: $scope}]' <<<"$observed")"
  done < <(jq -r '.[]' <<<"$expected")
  echo "$observed"
}

# Re-hash every `<stem>_path` / `<stem>_sha256` pair a row manifest recorded.
verify_manifest_hashes() {
  local manifest="$1" key path sha stem
  while IFS= read -r key; do
    stem="${key%_path}"
    path="$(jq -r --arg key "$key" '.[$key]' "$manifest")"
    sha="$(jq -r --arg key "${stem}_sha256" '.[$key] // empty' "$manifest")"
    [[ -n "$sha" ]] || continue
    [[ -f "$path" ]] || fail "$(basename "$(dirname "$manifest")"): missing evidence $path"
    [[ "$(file_sha256 "$path")" == "$sha" ]] \
      || fail "$(basename "$(dirname "$manifest")"): $key no longer matches its sealed hash"
  done < <(jq -r 'keys[] | select(endswith("_path"))' "$manifest")
}

write_manifest() {
  local dir="$1" json="$2" tmp
  mkdir -p "$dir"
  tmp="$dir/manifest.json.tmp.$$"
  jq '.' <<<"$json" >"$tmp"
  mv "$tmp" "$dir/manifest.json"
}

# --- --run ---------------------------------------------------------------------
run_mode() {
  local profile="${LTX25_SERVER_PROFILE:-default}"
  local host_url="${MOLD_HOST:-}" server_pid="${LTX25_SERVER_PID:-}" server_log="${LTX25_SERVER_LOG:-}"
  [[ -n "$host_url" ]] || fail "--run requires MOLD_HOST (the scratch server URL)"
  [[ -n "${MOLD_API_KEY:-}" ]] || fail "--run requires MOLD_API_KEY (never printed into evidence)"
  [[ "$server_pid" =~ ^[0-9]+$ ]] || fail "--run requires LTX25_SERVER_PID (systemctl show -p MainPID)"
  [[ -n "$server_log" ]] || fail "--run requires LTX25_SERVER_LOG (the scratch server's --log-file)"
  local profile_env
  profile_env="$(jq -ce --arg profile "$profile" '.common.profiles[$profile]' "$matrix")" \
    || fail "LTX25_SERVER_PROFILE=$profile is not a profile in $matrix"
  # A profile variable the engine does not register silently reads as unset
  # (CLAUDE.md, runtime_env), so refuse to attribute rows to a profile the
  # running server cannot honour.
  local registry variable
  registry="$(awk '/ENGINE_SHAPING_VARIABLES/{flag=1} flag{print} flag && /\];/{exit}' "$runtime_env_source")"
  [[ -n "$registry" ]] || fail "could not read ENGINE_SHAPING_VARIABLES from $runtime_env_source"
  while IFS= read -r variable; do
    [[ -n "$variable" ]] || continue
    grep -Fq "\"$variable\"" <<<"$registry" \
      || fail "profile $profile sets $variable, which is not in ENGINE_SHAPING_VARIABLES; the server would read it as unset"
  done < <(jq -r 'keys[]' <<<"$profile_env")
  if [[ "${LTX25_RUN_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    echo "preflight ok: profile $profile"
    exit 0
  fi
  [[ "$contract_test" != 1 ]] || fail "contract test mode never renders; use --seal over fixture rows"
  for command in curl timeout; do
    command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
  done
  [[ -f "$server_log" ]] || fail "server log does not exist: $server_log"
  ps -p "$server_pid" >/dev/null 2>&1 || fail "LTX25_SERVER_PID $server_pid is not running"

  local models_json="$evidence_dir/models-$profile.json"
  curl --fail --silent --show-error -H "x-api-key: $MOLD_API_KEY" "$host_url/api/models" >"$models_json" \
    || fail "could not read $host_url/api/models"
  curl --fail --silent --show-error -H "x-api-key: $MOLD_API_KEY" "$host_url/api/status" \
    >"$evidence_dir/status-$profile.json" || fail "could not read $host_url/api/status"
  jq -c --arg profile "$profile" --argjson env "$profile_env" \
    '{profile: $profile, env: $env}' >"$evidence_dir/profile-$profile.json"

  local gpu_uuid
  gpu_uuid="$(jq -r '.gpus[0].uuid' <<<"$host")"

  local row_ids
  if [[ -n "${LTX25_ROWS:-}" ]]; then
    row_ids="$(jq -r --arg profile "$profile" --arg filter "$LTX25_ROWS" \
      '($filter | split(",")) as $wanted
       | .rows[] | select(.profile == $profile) | select(.id as $id | .case as $case
         | any($wanted[]; . == $id or . == $case)) | .id' "$matrix")"
  else
    row_ids="$(jq -r --arg profile "$profile" '.rows[] | select(.profile == $profile) | .id' "$matrix")"
  fi
  local id row kind
  while IFS= read -r id; do
    [[ -n "$id" ]] || continue
    row="$(row_json "$id")"
    kind="$(jq -r '.kind' <<<"$row")"
    if [[ -f "$rows_dir/$id/manifest.json" && "${LTX25_RERUN:-0}" != 1 ]]; then
      local previous
      previous="$(jq -r '.status' "$rows_dir/$id/manifest.json")"
      if [[ "$previous" == passed || "$previous" == blocked ]]; then
        echo "skip $id: already $previous (LTX25_RERUN=1 to redo)"
        continue
      fi
    fi
    case "$kind" in
      deferred)
        write_manifest "$rows_dir/$id" "$(jq -c '{schema_version: "mold.ltx25.cuda.row.v1", id, model,
          case, profile, kind, status: "not_run", reason: .deferred_reason}' <<<"$row")"
        ;;
      fatal_cuda)
        run_fatal_cuda_row "$row" "$server_log" "$host_url"
        ;;
      cancellation)
        run_cancellation_row "$row" "$server_log" "$server_pid" "$host_url" "$gpu_uuid" "$models_json"
        ;;
      render | perf)
        run_render_row "$row" "$server_log" "$server_pid" "$gpu_uuid" "$models_json"
        ;;
      *) fail "$id: unknown row kind $kind" ;;
    esac
    local status
    status="$(jq -r '.status' "$rows_dir/$id/manifest.json")"
    echo "$id: $status"
    if [[ "$status" == failed && "${LTX25_CONTINUE_ON_FAILURE:-0}" != 1 ]]; then
      fail "$id failed; fix or set LTX25_CONTINUE_ON_FAILURE=1 to keep going"
    fi
  done <<<"$row_ids"
}

# Resolve `{seed_image}` / `{models_dir}` placeholders. Prints the resolved
# argument list as JSON, or a not_run reason on stderr with exit 1.
resolve_args() {
  local row="$1" args seed_image
  args="$(jq -c '.args' <<<"$row")"
  if jq -e 'any(.[]; contains("{seed_image}"))' <<<"$args" >/dev/null; then
    seed_image="${LTX25_SEED_IMAGE:-$evidence_dir/seed.png}"
    if [[ ! -s "$seed_image" ]]; then
      local source="$rows_dir/ltx-2.5-22b-distilled-int8-conv--smoke_silent_apng"
      if [[ -f "$source/manifest.json" && "$(jq -r '.status' "$source/manifest.json")" == passed ]] \
        && command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -v error -y -i "$(jq -r '.media.path' "$source/manifest.json")" -frames:v 1 "$seed_image" \
          || {
            echo "could not extract the seed image from the passed smoke_silent_apng render" >&2
            return 1
          }
      else
        echo "seed image unavailable: ltx-2.5-22b-distilled:int8-conv smoke_silent_apng has not passed" >&2
        return 1
      fi
    fi
    args="$(jq -c --arg seed "$seed_image" 'map(gsub("\\{seed_image\\}"; $seed))' <<<"$args")"
  fi
  jq -c --arg models "$models_dir" 'map(gsub("\\{models_dir\\}"; $models))' <<<"$args"
}

# Classify a non-zero `mold run` exit from its retained output. Prints
# `<reason_source>\t<reason>` for a blocked outcome and nothing for a plain
# failure. The order is deliberate: an OOM envelope refusal names the numbers
# and is the most specific; readiness and admission are typed server refusals;
# a bare HTTP status is the least informative and comes last.
classify_refusal() {
  local log="$1" line
  line="$(grep -Ei 'predicted (vram|host)|insufficient (vram|host|memory)|out of memory|CUDA_ERROR_OUT_OF_MEMORY|OOM' "$log" | tail -1 || true)"
  if [[ -n "$line" ]]; then
    printf 'oom_envelope\t%s\n' "$line"
    return 0
  fi
  line="$(grep -Ei 'RUNTIME_UNAVAILABLE|runtime_readiness|not implemented|not runnable|Download only' "$log" | tail -1 || true)"
  if [[ -n "$line" ]]; then
    printf 'runtime_readiness\t%s\n' "$line"
    return 0
  fi
  line="$(grep -Ei 'admission|cannot be admitted|not admitted|refused|GENERATION_.*_REJECTED|422' "$log" | tail -1 || true)"
  if [[ -n "$line" ]]; then
    printf 'admission\t%s\n' "$line"
    return 0
  fi
  line="$(grep -Ei 'HTTP (5[0-9][0-9]|4[0-9][0-9])|status code [45][0-9][0-9]' "$log" | tail -1 || true)"
  if [[ -n "$line" ]]; then
    printf 'http_status\t%s\n' "$line"
    return 0
  fi
}

# Sample nvidia-smi (1 Hz) into vram.csv, the server's VmHWM into host.csv,
# and PID-bound compute observations into compute.csv until the PID file's
# process exits. Runs in the background; the caller kills it.
sample_gpu_and_host() {
  local watched_pid="$1" server_pid="$2" gpu_uuid="$3" dir="$4"
  printf 'polled_at_utc,memory_used_mib,utilization_gpu\n' >"$dir/vram.csv"
  printf 'polled_at_utc,vmhwm_kib\n' >"$dir/host.csv"
  printf 'polled_at_utc,generation_root_pid,observed_pid,gpu_uuid\n' >"$dir/compute.csv"
  while kill -0 "$watched_pid" 2>/dev/null; do
    local now sample hwm apps
    now="$(utc_now)"
    sample="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      | head -1 | tr -d ' ' || true)"
    [[ -z "$sample" ]] || printf '%s,%s\n' "$now" "$sample" >>"$dir/vram.csv"
    hwm="$(awk '/^VmHWM:/ {print $2}' "/proc/$server_pid/status" 2>/dev/null || true)"
    [[ -z "$hwm" ]] || printf '%s,%s\n' "$now" "$hwm" >>"$dir/host.csv"
    apps="$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null || true)"
    while IFS= read -r line; do
      [[ -n "$line" ]] || continue
      local pid="${line%%,*}" uuid="${line#*,}"
      pid="${pid//[[:space:]]/}"
      uuid="${uuid//[[:space:]]/}"
      [[ "$pid" == "$server_pid" && "$uuid" == "$gpu_uuid" ]] || continue
      printf '%s,%s,%s,%s\n' "$now" "$server_pid" "$pid" "$uuid" >>"$dir/compute.csv"
    done <<<"$apps"
    sleep 1
  done
}

# Readiness of the row's model on the running server. Prints
# `<status>\t<reason_source>\t<reason>` (status ok|not_run|blocked).
model_readiness() {
  local model="$1" models_json="$2" entry
  entry="$(jq -c --arg model "$model" '(.models? // .) | .[] | select(.name == $model)' "$models_json")"
  if [[ -z "$entry" ]]; then
    printf 'not_run\t\tmodel %s is not registered by this server build\n' "$model"
    return 0
  fi
  if [[ "$(jq -r '.downloaded // false' <<<"$entry")" != true ]]; then
    printf 'not_run\t\tmodel %s is not downloaded (remaining %s bytes)\n' "$model" \
      "$(jq -r '.remaining_download_bytes // "unknown"' <<<"$entry")"
    return 0
  fi
  if [[ "$(jq -r '.runtime_available // true' <<<"$entry")" == false ]]; then
    printf 'blocked\truntime_readiness\t%s\n' \
      "$(jq -r '.runtime_unavailable_reason // "runtime_available is false"' <<<"$entry")"
    return 0
  fi
  local readiness
  readiness="$(jq -r '.runtime_readiness_error // empty' <<<"$entry")"
  if [[ -n "$readiness" ]]; then
    printf 'blocked\truntime_readiness\t%s\n' "$readiness"
    return 0
  fi
  printf 'ok\t\t\n'
}

run_render_row() {
  local row="$1" server_log="$2" server_pid="$3" gpu_uuid="$4" models_json="$5"
  local id model kind dir
  id="$(jq -r '.id' <<<"$row")"
  model="$(jq -r '.model' <<<"$row")"
  kind="$(jq -r '.kind' <<<"$row")"
  dir="$rows_dir/$id"
  mkdir -p "$dir"
  local base
  base="$(jq -c '{schema_version: "mold.ltx25.cuda.row.v1", id, model, case, profile, kind}' <<<"$row")"

  local readiness status source reason
  readiness="$(model_readiness "$model" "$models_json")"
  IFS=$'\t' read -r status source reason <<<"$readiness"
  if [[ "$status" == not_run ]]; then
    write_manifest "$dir" "$(jq -c --arg reason "$reason" '. + {status: "not_run", reason: $reason}' <<<"$base")"
    return 0
  fi
  local args
  if ! args="$(resolve_args "$row" 2>"$dir/resolve.err")"; then
    write_manifest "$dir" "$(jq -c --arg reason "$(cat "$dir/resolve.err")" \
      '. + {status: "not_run", reason: $reason}' <<<"$base")"
    return 0
  fi
  local prompt seed format ext output
  prompt="$(jq -r '.prompt' <<<"$row")"
  seed="$(jq -r '.seed' <<<"$row")"
  format="$(jq -r '.format' <<<"$row")"
  ext="$(media_extension "$format")"
  output="$dir/output.$ext"
  rm -f "$output"
  local -a command=("$mold_bin" run "$model" "$prompt")
  while IFS= read -r arg; do
    command+=("$arg")
  done < <(jq -r '.[]' <<<"$args")
  command+=(--seed "$seed" --output "$output")
  jq -n --arg model "$model" --arg prompt "$prompt" --argjson seed "$seed" --argjson args "$args" \
    --arg output "$output" --arg profile "$(jq -r '.profile' <<<"$row")" \
    '{model: $model, prompt: $prompt, seed: $seed, args: $args, output: $output, profile: $profile}' \
    >"$dir/request.json"
  if [[ "$status" == blocked ]]; then
    printf '%s\n' "$reason" >"$dir/stdout.log"
    write_manifest "$dir" "$(jq -c --arg reason "$reason" --arg source "$source" \
      --argjson command "$(printf '%s\n' "${command[@]}" | jq -R . | jq -sc .)" \
      --arg stdout "$dir/stdout.log" --arg stdout_sha "$(file_sha256 "$dir/stdout.log")" \
      --arg request "$dir/request.json" --arg request_sha "$(file_sha256 "$dir/request.json")" \
      '. + {status: "blocked", reason: $reason, reason_source: $source, command: $command,
        request_path: $request, request_sha256: $request_sha,
        stdout_log_path: $stdout, stdout_log_sha256: $stdout_sha}' <<<"$base")"
    return 0
  fi

  local started started_epoch offset
  started="$(utc_now)"
  started_epoch="$(date +%s)"
  offset="$(file_size "$server_log")"
  set +e
  timeout "$row_timeout" env MOLD_HOST="$MOLD_HOST" MOLD_API_KEY="$MOLD_API_KEY" \
    "${command[@]}" >"$dir/stdout.log" 2>&1 &
  local run_pid=$!
  sample_gpu_and_host "$run_pid" "$server_pid" "$gpu_uuid" "$dir" &
  local sampler_pid=$!
  wait "$run_pid"
  local exit_code=$?
  wait "$sampler_pid" 2>/dev/null
  set -e
  local finished seconds
  finished="$(utc_now)"
  seconds=$(($(date +%s) - started_epoch))
  tail -c +"$((offset + 1))" "$server_log" >"$dir/server.log"

  local manifest
  manifest="$(jq -c --arg started "$started" --arg finished "$finished" --argjson seconds "$seconds" \
    --argjson exit_code "$exit_code" \
    --argjson command "$(printf '%s\n' "${command[@]}" | jq -R . | jq -sc .)" \
    --arg dir "$dir" --arg full_log "$server_log" \
    --arg stdout_sha "$(file_sha256 "$dir/stdout.log")" --arg server_sha "$(file_sha256 "$dir/server.log")" \
    --arg request_sha "$(file_sha256 "$dir/request.json")" \
    --arg vram_sha "$(file_sha256 "$dir/vram.csv")" --arg host_sha "$(file_sha256 "$dir/host.csv")" \
    --arg compute_sha "$(file_sha256 "$dir/compute.csv")" \
    --argjson gpu "$(jq -c '.gpus[0]' <<<"$host")" --argjson server_pid "$server_pid" \
    --argjson peak_mib "$(awk -F, 'NR > 1 && $2 > max {max = $2} END {print max + 0}' "$dir/vram.csv")" \
    --argjson peak_hwm "$(awk -F, 'NR > 1 && $2 > max {max = $2} END {print max + 0}' "$dir/host.csv")" \
    --argjson observed "$([[ "$(wc -l <"$dir/compute.csv")" -gt 1 ]] && echo true || echo false)" \
    --argjson expected "$(jq -c '.expect.provenance' <<<"$row")" \
    --argjson metadata "$(jq -c '.expect.metadata // {}' <<<"$row")" '
    . + {started_at: $started, finished_at: $finished, seconds: $seconds, exit_code: $exit_code,
      command: $command,
      request_path: ($dir + "/request.json"), request_sha256: $request_sha,
      stdout_log_path: ($dir + "/stdout.log"), stdout_log_sha256: $stdout_sha,
      server_log_path: ($dir + "/server.log"), server_log_sha256: $server_sha,
      server_log_full_path: $full_log,
      vram_csv_path: ($dir + "/vram.csv"), vram_csv_sha256: $vram_sha,
      host_csv_path: ($dir + "/host.csv"), host_csv_sha256: $host_sha,
      compute_observation_path: ($dir + "/compute.csv"), compute_observation_sha256: $compute_sha,
      gpu: ($gpu + {peak_memory_used_mib: $peak_mib, server_pid: $server_pid, cuda_work_observed: $observed}),
      host: {peak_vmhwm_kib: $peak_hwm},
      provenance_expected: $expected, metadata_expected: $metadata}' <<<"$base")"

  if [[ "$exit_code" -ne 0 ]]; then
    local classified
    classified="$(classify_refusal "$dir/stdout.log")"
    if [[ -n "$classified" ]]; then
      IFS=$'\t' read -r source reason <<<"$classified"
      write_manifest "$dir" "$(jq -c --arg reason "$reason" --arg source "$source" \
        '. + {status: "blocked", reason: $reason, reason_source: $source}' <<<"$manifest")"
    else
      reason="$(grep -v '^\s*$' "$dir/stdout.log" | tail -1 || true)"
      write_manifest "$dir" "$(jq -c --arg reason "mold run exited $exit_code: ${reason:-no output}" \
        '. + {status: "failed", reason: $reason}' <<<"$manifest")"
    fi
    return 0
  fi

  local failure=""
  if [[ ! -s "$output" ]]; then
    failure="mold run exited 0 but wrote no media at $output"
  elif ! failure="$(probe_media_against_expect "$output" "$dir/ffprobe.json" "$row")"; then
    :
  else
    failure=""
  fi
  local generation="" commit="" observed=""
  if [[ -z "$failure" ]]; then
    # The client records its Library row right after saving the file, but the
    # scratch server writes its own row to the same mold.db concurrently, so
    # the client's upsert can land moments after `mold run` exits. Poll
    # briefly before calling the row absent or stale.
    local library_wait=0
    while :; do
      if generation="$(library_row_for "$output" "$row" 2>"$dir/library.err")"; then
        failure=""
        break
      fi
      failure="$(cat "$dir/library.err")"
      [[ "$library_wait" -lt 15 ]] || break
      sleep 3
      library_wait=$((library_wait + 3))
    done
  fi
  if [[ -z "$failure" ]]; then
    commit="$(generator_commit_of "$generation" 2>"$dir/library.err")" || failure="$(cat "$dir/library.err")"
  fi
  if [[ -z "$failure" ]]; then
    observed="$(observe_provenance "$dir/server.log" "$server_log" "$(jq -c '.expect.provenance' <<<"$row")" \
      "$dir" 2>"$dir/provenance.err")" || failure="$(cat "$dir/provenance.err")"
  fi
  if [[ -z "$failure" && "$(jq -r '.gpu.cuda_work_observed' <<<"$manifest")" != true ]]; then
    failure="server PID $server_pid was never observed on $gpu_uuid by nvidia-smi during the render"
  fi
  if [[ "$kind" == perf && -z "$failure" ]]; then
    db_dump_scheduler_estimates >"$dir/scheduler_estimates.json"
    manifest="$(jq -c --arg path "$dir/scheduler_estimates.json" \
      --arg sha "$(file_sha256 "$dir/scheduler_estimates.json")" \
      '. + {scheduler_estimates_path: $path, scheduler_estimates_sha256: $sha}' <<<"$manifest")"
  fi
  if [[ -n "$failure" ]]; then
    write_manifest "$dir" "$(jq -c --arg reason "$failure" '. + {status: "failed", reason: $reason}' <<<"$manifest")"
    return 0
  fi
  write_manifest "$dir" "$(jq -c --arg output "$output" --arg format "$format" \
    --arg sha "$(file_sha256 "$output")" --argjson bytes "$(file_size "$output")" \
    --arg probe "$dir/ffprobe.json" --arg probe_sha "$(file_sha256 "$dir/ffprobe.json")" \
    --argjson generation "$generation" --arg commit "$commit" --argjson observed "$observed" '
    . + {status: "passed",
      media: {path: $output, format: $format, sha256: $sha, bytes: $bytes,
        ffprobe_path: $probe, ffprobe_sha256: $probe_sha},
      generation: $generation, generator_commit: $commit, provenance_observed: $observed}' <<<"$manifest")"
}

run_cancellation_row() {
  local row="$1" server_log="$2" server_pid="$3" host_url="$4" gpu_uuid="$5" models_json="$6"
  local id model dir
  id="$(jq -r '.id' <<<"$row")"
  model="$(jq -r '.model' <<<"$row")"
  dir="$rows_dir/$id"
  mkdir -p "$dir"
  local base
  base="$(jq -c '{schema_version: "mold.ltx25.cuda.row.v1", id, model, case, profile, kind}' <<<"$row")"
  local readiness status source reason
  readiness="$(model_readiness "$model" "$models_json")"
  IFS=$'\t' read -r status source reason <<<"$readiness"
  if [[ "$status" != ok ]]; then
    printf '%s\n' "$reason" >"$dir/stdout.log"
    write_manifest "$dir" "$(jq -c --arg status "$status" --arg reason "$reason" --arg source "$source" \
      --arg stdout "$dir/stdout.log" --arg stdout_sha "$(file_sha256 "$dir/stdout.log")" \
      '. + {status: $status, reason: $reason, command: ["mold", "queue", "cancel"],
        stdout_log_path: $stdout, stdout_log_sha256: $stdout_sha}
        + (if $status == "blocked" then {reason_source: $source} else {} end)' <<<"$base")"
    return 0
  fi
  local prompt seed output
  prompt="$(jq -r '.prompt' <<<"$row")"
  seed="$(jq -r '.seed' <<<"$row")"
  output="$dir/output.mp4"
  rm -f "$output"
  local -a command=("$mold_bin" run "$model" "$prompt")
  while IFS= read -r arg; do
    command+=("$arg")
  done < <(jq -r '.args[]' <<<"$row")
  command+=(--seed "$seed" --output "$output")
  jq -n --arg model "$model" --arg prompt "$prompt" --argjson seed "$seed" --argjson args "$(jq -c '.args' <<<"$row")" \
    --arg output "$output" '{model: $model, prompt: $prompt, seed: $seed, args: $args, output: $output}' \
    >"$dir/request.json"
  # The memory floor is what the server held IMMEDIATELY before this job:
  # earlier rows legitimately leave persistent allocations (cached engines,
  # CUDA context), so a startup-time baseline would fail a correct cancel.
  local baseline_mib
  baseline_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
  local started started_epoch offset
  started="$(utc_now)"
  started_epoch="$(date +%s)"
  offset="$(file_size "$server_log")"
  set +e
  timeout "$row_timeout" env MOLD_HOST="$MOLD_HOST" MOLD_API_KEY="$MOLD_API_KEY" \
    "${command[@]}" >"$dir/stdout.log" 2>&1 &
  local run_pid=$!
  sample_gpu_and_host "$run_pid" "$server_pid" "$gpu_uuid" "$dir" &
  local sampler_pid=$!
  # Wait for the job to be running on the server, then let it denoise for a
  # bounded while before cancelling through the CLI.
  local job_id="" waited=0 cancelled_at=""
  while ((waited < 600)); do
    job_id="$(curl --silent -H "x-api-key: $MOLD_API_KEY" "$host_url/api/queue" \
      | jq -r --arg model "$model" '.entries[]? | select(.model == $model and .state == "running") | .id' | head -1)"
    [[ -z "$job_id" ]] || break
    kill -0 "$run_pid" 2>/dev/null || break
    sleep 2
    waited=$((waited + 2))
  done
  local cancel_exit=-1
  if [[ -n "$job_id" ]]; then
    sleep "${LTX25_CANCEL_AFTER_SECONDS:-20}"
    cancelled_at="$(utc_now)"
    env MOLD_HOST="$MOLD_HOST" MOLD_API_KEY="$MOLD_API_KEY" "$mold_bin" queue cancel "$job_id" \
      >"$dir/cancel.log" 2>&1
    cancel_exit=$?
  fi
  wait "$run_pid"
  local exit_code=$?
  wait "$sampler_pid" 2>/dev/null
  set -e
  local finished seconds
  finished="$(utc_now)"
  seconds=$(($(date +%s) - started_epoch))
  tail -c +"$((offset + 1))" "$server_log" >"$dir/server.log"
  local settle=0 queue_clear=false memory_back=false used
  while ((settle < 30)); do
    if ! curl --silent -H "x-api-key: $MOLD_API_KEY" "$host_url/api/queue" \
      | jq -e --arg id "$job_id" '.entries[]? | select(.id == $id)' >/dev/null; then
      queue_clear=true
    fi
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
    if ((used <= baseline_mib + 1024)); then
      memory_back=true
    fi
    [[ "$queue_clear" == true && "$memory_back" == true ]] && break
    sleep 1
    settle=$((settle + 1))
  done
  local library_row
  library_row="$(db_query_generation "output.mp4" "$dir")"
  local failure=""
  [[ -n "$job_id" ]] || failure="the job never reached the running state within 600 s"
  [[ -n "$failure" || "$cancel_exit" -eq 0 ]] || failure="mold queue cancel exited $cancel_exit"
  [[ -n "$failure" || "$exit_code" -ne 0 ]] || failure="mold run exited 0 after a cancel"
  [[ -n "$failure" || "$queue_clear" == true ]] || failure="job $job_id still listed 30 s after cancel"
  [[ -n "$failure" || -z "$library_row" ]] || failure="a Library row was written for a cancelled job"
  [[ -n "$failure" || ! -s "$output" ]] || failure="media was written for a cancelled job"
  [[ -n "$failure" || "$memory_back" == true ]] || failure="GPU memory did not return to baseline ($baseline_mib MiB) within 30 s"
  local manifest
  manifest="$(jq -c --arg started "$started" --arg finished "$finished" --argjson seconds "$seconds" \
    --argjson exit_code "$exit_code" --argjson cancel_exit "$cancel_exit" --arg job_id "$job_id" \
    --arg cancelled_at "$cancelled_at" --argjson baseline "$baseline_mib" \
    --argjson command "$(printf '%s\n' "${command[@]}" | jq -R . | jq -sc .)" --arg dir "$dir" \
    --arg full_log "$server_log" \
    --arg stdout_sha "$(file_sha256 "$dir/stdout.log")" --arg server_sha "$(file_sha256 "$dir/server.log")" \
    --arg request_sha "$(file_sha256 "$dir/request.json")" \
    --arg vram_sha "$(file_sha256 "$dir/vram.csv")" --arg host_sha "$(file_sha256 "$dir/host.csv")" \
    --arg compute_sha "$(file_sha256 "$dir/compute.csv")" \
    --argjson queue_clear "$queue_clear" --argjson memory_back "$memory_back" '
    . + {started_at: $started, finished_at: $finished, seconds: $seconds, exit_code: $exit_code,
      command: $command, request_path: ($dir + "/request.json"), request_sha256: $request_sha,
      stdout_log_path: ($dir + "/stdout.log"), stdout_log_sha256: $stdout_sha,
      server_log_path: ($dir + "/server.log"), server_log_sha256: $server_sha,
      server_log_full_path: $full_log,
      vram_csv_path: ($dir + "/vram.csv"), vram_csv_sha256: $vram_sha,
      host_csv_path: ($dir + "/host.csv"), host_csv_sha256: $host_sha,
      compute_observation_path: ($dir + "/compute.csv"), compute_observation_sha256: $compute_sha,
      cancellation: {job_id: $job_id, cancelled_at: $cancelled_at, cancel_exit_code: $cancel_exit,
        queue_cleared: $queue_clear, gpu_memory_baseline_mib: $baseline,
        gpu_memory_returned: $memory_back}}' <<<"$base")"
  if [[ -n "$failure" ]]; then
    write_manifest "$dir" "$(jq -c --arg reason "$failure" '. + {status: "failed", reason: $reason}' <<<"$manifest")"
  else
    # A cancellation retains no media; the passed evidence is the cancel
    # log, the queue observation, and the absence of a Library row.
    write_manifest "$dir" "$(jq -c --arg cancel "$dir/cancel.log" --arg cancel_sha "$(file_sha256 "$dir/cancel.log")" \
      '. + {status: "passed", cancel_log_path: $cancel, cancel_log_sha256: $cancel_sha,
        provenance_expected: [], provenance_observed: []}' <<<"$manifest")"
  fi
}

run_fatal_cuda_row() {
  local row="$1" server_log="$2" host_url="$3"
  local id dir
  id="$(jq -r '.id' <<<"$row")"
  dir="$rows_dir/$id"
  mkdir -p "$dir"
  local base
  base="$(jq -c '{schema_version: "mold.ltx25.cuda.row.v1", id, model, case, profile, kind}' <<<"$row")"
  if [[ -z "${LTX25_FATAL_CUDA_REPRO:-}" ]]; then
    write_manifest "$dir" "$(jq -c --arg reason "$(jq -r '.deferred_reason' <<<"$row")" \
      '. + {status: "not_run", reason: $reason}' <<<"$base")"
    return 0
  fi
  local started started_epoch offset
  started="$(utc_now)"
  started_epoch="$(date +%s)"
  offset="$(file_size "$server_log")"
  set +e
  timeout "$row_timeout" env MOLD_HOST="$MOLD_HOST" MOLD_API_KEY="$MOLD_API_KEY" \
    bash -c "$LTX25_FATAL_CUDA_REPRO" >"$dir/stdout.log" 2>&1
  local exit_code=$?
  set -e
  sleep 5
  tail -c +"$((offset + 1))" "$server_log" >"$dir/server.log"
  local status_code
  status_code="$(curl --silent -o "$dir/status-after.json" -w '%{http_code}' \
    -H "x-api-key: $MOLD_API_KEY" "$host_url/api/status" || echo 000)"
  local sentence
  sentence="$(jq -r '.expect.server_log' <<<"$row")"
  local manifest
  manifest="$(jq -c --arg started "$started" --arg finished "$(utc_now)" \
    --argjson seconds "$(($(date +%s) - started_epoch))" --argjson exit_code "$exit_code" \
    --argjson command "$(jq -nc --arg repro "$LTX25_FATAL_CUDA_REPRO" '["bash", "-c", $repro]')" \
    --arg dir "$dir" --arg full_log "$server_log" --arg status_code "$status_code" \
    --arg stdout_sha "$(file_sha256 "$dir/stdout.log")" --arg server_sha "$(file_sha256 "$dir/server.log")" '
    . + {started_at: $started, finished_at: $finished, seconds: $seconds, exit_code: $exit_code,
      command: $command, stdout_log_path: ($dir + "/stdout.log"), stdout_log_sha256: $stdout_sha,
      server_log_path: ($dir + "/server.log"), server_log_sha256: $server_sha,
      server_log_full_path: $full_log, status_after_http: $status_code,
      provenance_expected: [], provenance_observed: []}' <<<"$base")"
  # Both halves of the matrix expectation are required: the quarantine
  # sentence AND the server refusing /api/status afterwards. A reproducer
  # that logs the sentence while the server stays healthy is not a pass.
  local failure=""
  grep -Fq -- "$sentence" "$dir/server.log" \
    || failure="the server log never recorded: $sentence"
  if [[ -z "$failure" && "$status_code" == 200 ]]; then
    failure="/api/status still answered 200 after the fatal context; the server did not refuse"
  fi
  if [[ -z "$failure" ]]; then
    write_manifest "$dir" "$(jq -c '. + {status: "passed"}' <<<"$manifest")"
  else
    write_manifest "$dir" "$(jq -c --arg reason "$failure" \
      '. + {status: "failed", reason: $reason}' <<<"$manifest")"
  fi
}

# --- --seal --------------------------------------------------------------------
seal_mode() {
  # Rows first: which manifests have a passed row decides whether an absent
  # asset is a failure or merely unattempted.
  local rows='[]' passed_models='[]' id row dir manifest status
  while IFS= read -r id; do
    [[ -n "$id" ]] || continue
    row="$(row_json "$id")"
    dir="$rows_dir/$id"
    manifest="$dir/manifest.json"
    if [[ ! -f "$manifest" ]]; then
      local reason="never attempted"
      if [[ "$(jq -r '.kind' <<<"$row")" == deferred || "$(jq -r '.kind' <<<"$row")" == fatal_cuda ]]; then
        reason="$(jq -r '.deferred_reason' <<<"$row")"
      fi
      write_manifest "$dir" "$(jq -c --arg reason "$reason" '{schema_version: "mold.ltx25.cuda.row.v1", id, model,
        case, profile, kind, status: "not_run", reason: $reason}' <<<"$row")"
    fi
    jq -e --argjson row "$row" '.schema_version == "mold.ltx25.cuda.row.v1" and .id == $row.id
      and .model == $row.model and .case == $row.case and .profile == $row.profile' "$manifest" >/dev/null \
      || fail "$id: row manifest does not describe this matrix row"
    status="$(jq -r '.status' "$manifest")"
    case "$status" in
      passed) seal_passed_row "$row" "$manifest" ;;
      blocked)
        jq -e '(.reason | type == "string" and length > 0)
          and (.reason_source | IN("admission", "runtime_readiness", "http_status", "oom_envelope"))
          and (.stdout_log_path | type == "string")' "$manifest" >/dev/null \
          || fail "$id: a blocked row needs a reason, a known reason_source, and its stdout.log"
        verify_manifest_hashes "$manifest"
        ;;
      failed)
        jq -e '(.reason | type == "string" and length > 0) and (.stdout_log_path | type == "string")' \
          "$manifest" >/dev/null || fail "$id: a failed row needs a reason and its stdout.log"
        verify_manifest_hashes "$manifest"
        ;;
      not_run)
        jq -e '.reason | type == "string" and length > 0' "$manifest" >/dev/null \
          || fail "$id: a not_run row needs a reason"
        ;;
      *) fail "$id: unknown row status $status" ;;
    esac
    [[ "$status" != passed ]] || passed_models="$(jq -c --arg model "$(jq -r '.model' <<<"$row")" '. + [$model] | unique' <<<"$passed_models")"
    rows="$(jq -c --arg manifest "$manifest" --arg sha "$(file_sha256 "$manifest")" --slurpfile m "$manifest" \
      '. + [($m[0] | {id, model, case, profile, kind, status, reason, reason_source, started_at, finished_at,
        seconds, exit_code, command, stdout_log_path, stdout_log_sha256, server_log_path,
        server_log_sha256, media, generation, provenance_expected, provenance_observed,
        metadata_expected, gpu, host, generator_commit, metal_ab}
        | with_entries(select(.value != null or .key == "metal_ab"))
        | if has("metal_ab") and ($m[0] | has("metal_ab") | not) then del(.metal_ab) else . end
        | . + {manifest_path: $manifest, manifest_sha256: $sha})]' <<<"$rows")"
  done < <(jq -r '.rows[].id' "$matrix")

  # Assets: every (manifest, file) pair from the generated fixture, resolved
  # under MOLD_MODELS_DIR. A file may be absent only for a manifest with no
  # passed row; a present file must carry a matching verified marker, and
  # outside contract mode its bytes are re-hashed.
  local assets='[]' relative expected_sha expected_bytes manifest_name component path marker_sha actual_sha proof present bytes
  while IFS=$'\t' read -r manifest_name component relative expected_bytes expected_sha; do
    path="$models_dir/$relative"
    marker_sha=null
    actual_sha=null
    bytes=null
    present=false
    if [[ -s "$path" ]]; then
      present=true
      bytes="$(file_size "$path")"
      [[ -f "$path.sha256-verified" ]] || fail "$component of $manifest_name has no verified SHA marker: $path"
      marker_sha="\"$(tr -d '[:space:]' <"$path.sha256-verified")\""
      [[ "$marker_sha" == "\"$expected_sha\"" ]] \
        || fail "$component of $manifest_name marker mismatch: expected $expected_sha, found $marker_sha"
      if [[ "$contract_test" == 1 ]]; then
        proof="contract test: current bytes not qualified"
      else
        [[ "$bytes" == "$expected_bytes" ]] \
          || fail "$component of $manifest_name is $bytes bytes, expected $expected_bytes"
        actual_sha="$(file_sha256 "$path")"
        [[ "$actual_sha" == "$expected_sha" ]] \
          || fail "$component of $manifest_name byte hash mismatch: expected $expected_sha, found $actual_sha"
        actual_sha="\"$actual_sha\""
        proof="current bytes rehashed and matched manifest"
      fi
    else
      if jq -e --arg model "$manifest_name" 'index($model) != null' <<<"$passed_models" >/dev/null; then
        fail "$component of $manifest_name is absent at $path but a row for that model passed"
      fi
      proof="absent: no row for this manifest passed"
    fi
    assets="$(jq -c --arg manifest "$manifest_name" --arg component "$component" --arg relative "$relative" \
      --arg path "$path" --argjson present "$present" --argjson bytes "$bytes" \
      --argjson expected_bytes "$expected_bytes" --arg expected_sha "$expected_sha" \
      --argjson marker_sha "$marker_sha" --argjson actual_sha "$actual_sha" --arg proof "$proof" \
      '. + [{manifest: $manifest, component: $component, storage_relative_path: $relative, path: $path,
        present: $present, bytes: $bytes, expected_bytes: $expected_bytes, expected_sha256: $expected_sha,
        marker_sha256: $marker_sha, actual_sha256: $actual_sha, identity_proof: $proof}]' <<<"$assets")"
  done < <(jq -r '.assets[] | [.manifest, .component, .storage_relative_path, .size_bytes, .sha256] | @tsv' "$assets_fixture")

  local references='[]'
  record_reference() {
    local name="$1" expected="$2" path actual
    path="$references_root/$name"
    [[ -d "$path/.git" ]] || fail "missing pinned reference clone: $path"
    actual="$(git -C "$path" rev-parse HEAD)"
    [[ "$actual" == "$expected" ]] || fail "$name is at $actual, expected $expected"
    [[ -z "$(git -C "$path" status --porcelain)" ]] \
      || fail "$name reference clone has uncommitted or untracked changes"
    references="$(jq -c --arg name "$name" --arg path "$path" --arg commit "$actual" \
      '. + [{name: $name, path: $path, commit: $commit, status: "pinned_clean"}]' <<<"$references")"
  }
  record_reference ltx-2-upstream 400fd31054597515f47125691032c04b1c3ee24e
  record_reference comfyui-ltxvideo-upstream 15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d
  record_reference comfyui-upstream a1079ba16f2674734b065eb036fbfdddaa321a4d
  record_reference diffusers-upstream 95c0d467cc2a4770b71fa25a117320377e6eb08f

  local profiles
  profiles="$(jq -c --argjson rows "$rows" '[.common.profiles | to_entries[] | .key as $name
    | {name: $name, env: .value,
       rows_attempted: ([$rows[] | select(.profile == $name and .status != "not_run")] | length)}]' "$matrix")"

  local comfy_int8 comfy_gguf
  comfy_int8="$(seal_comfy_reference "distilled INT8 ConvRot" "${LTX25_COMFY_INT8_MANIFEST:-}")"
  comfy_gguf="$(seal_comfy_reference "distilled GGUF Q4_K_M" "${LTX25_COMFY_GGUF_MANIFEST:-}")"

  local gates='[]'
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
  run_gate server_ltx25_admission nix develop -c cargo test -p mold-ai-server ltx25

  local summary qualification_status source_tree_state=clean
  summary="$(jq -c '{passed: ([.[] | select(.status == "passed")] | length),
    failed: ([.[] | select(.status == "failed")] | length),
    blocked: ([.[] | select(.status == "blocked")] | length),
    not_run: ([.[] | select(.status == "not_run")] | length)}' <<<"$rows")"
  # A not_run row whose kind is runnable (render/perf/cancellation) is missing
  # coverage, not an answered question: only the explicitly deferred kinds may
  # be absent from a passed campaign. Blocked rows ARE answers (a typed
  # refusal with retained evidence) and do not hold the campaign incomplete.
  local unattempted
  unattempted="$(jq '[.[] | select(.status == "not_run"
    and (.kind | IN("deferred", "fatal_cuda") | not))] | length' <<<"$rows")"
  if [[ "$contract_test" == 1 ]]; then
    qualification_status=not_qualified_contract_test
    source_tree_state=contract_test
  elif [[ "$(jq '.failed' <<<"$summary")" -gt 0 ]]; then
    qualification_status=failed
  elif [[ "$(jq '.passed' <<<"$summary")" -eq 0 || "$unattempted" -gt 0 ]]; then
    qualification_status=incomplete
  else
    qualification_status=passed
  fi

  local tmp_report="$report.tmp.$$"
  jq -n \
    --arg captured_at "$(utc_now)" --arg source_commit "$source_commit" \
    --arg source_tree_state "$source_tree_state" --arg mold_home "$mold_home" --arg models_dir "$models_dir" \
    --arg matrix_path "$matrix" --arg matrix_sha "$(file_sha256 "$matrix")" \
    --argjson matrix_rows "$(jq '.rows | length' "$matrix")" \
    --argjson host "$host" --argjson assets "$assets" --argjson references "$references" \
    --argjson profiles "$profiles" --argjson rows "$rows" --argjson comfy_int8 "$comfy_int8" \
    --argjson comfy_gguf "$comfy_gguf" --argjson gates "$gates" --argjson summary "$summary" \
    --arg qualification_status "$qualification_status" \
    '{schema_version: "mold.ltx25.cuda.verification.v1", captured_at: $captured_at,
      source_commit: $source_commit, source_tree_state: $source_tree_state,
      mold_home: $mold_home, models_dir: $models_dir, backend_scope: "cuda",
      matrix: {path: $matrix_path, sha256: $matrix_sha, schema_version: "mold.ltx25.cuda.matrix.v1",
        rows: $matrix_rows},
      host: $host, assets: $assets, references: $references, server_profiles: $profiles,
      rows: $rows, comfy_reference: {int8: $comfy_int8, gguf_q4: $comfy_gguf}, gates: $gates,
      summary: $summary, qualification_status: $qualification_status,
      preservation: {downloaded_models_deleted: false, rendered_media_deleted: false}}' >"$tmp_report"
  mv "$tmp_report" "$report"
  "$validator" "$report" --schema "$schema" >/dev/null
  echo "$report"
  if [[ "$qualification_status" == failed ]]; then
    echo "LTX-2.5 CUDA qualification did not pass; the report is evidence of failure only" >&2
    exit 1
  fi
}

# Metal <-> CUDA A/B for the int8-conv smoke rows: the reference is whatever
# media the NEWEST sealed Metal report under $metal_reference_root retained
# for the row's label, located by basename and authenticated against the
# sha256 that report recorded. Prints the block as JSON, or `null` when no
# reference root or no matching reference exists (never a failure).
metal_ab_block() {
  local row="$1" media="$2" dir="$3" label
  case "$(jq -r '.model + "/" + .case' <<<"$row")" in
    "ltx-2.5-22b-distilled:int8-conv/smoke_silent_apng") label=silent_video ;;
    "ltx-2.5-22b-distilled:int8-conv/smoke_audio_mp4") label=audio_video ;;
    *) return 0 ;;
  esac
  if [[ ! -d "$metal_reference_root/verification" ]]; then
    echo null
    return 0
  fi
  local newest="" candidate
  shopt -s nullglob
  for candidate in "$metal_reference_root"/verification/ltx25-metal-int8-verification-*.json; do
    if [[ -z "$newest" || "$candidate" -nt "$newest" ]]; then
      newest="$candidate"
    fi
  done
  shopt -u nullglob
  if [[ -z "$newest" ]]; then
    echo null
    return 0
  fi
  local entry reference_name reference_sha reference
  entry="$(jq -c --arg label "$label" '.media[]? | select(.label == $label)' "$newest")"
  if [[ -z "$entry" ]]; then
    echo null
    return 0
  fi
  reference_name="$(basename "$(jq -r '.path' <<<"$entry")")"
  reference_sha="$(jq -r '.sha256' <<<"$entry")"
  reference="$(find "$metal_reference_root" -type f -name "$reference_name" | head -1)"
  if [[ -z "$reference" ]]; then
    echo null
    return 0
  fi
  [[ "$(file_sha256 "$reference")" == "$reference_sha" ]] \
    || fail "Metal reference $reference does not match the sha256 its report $newest recorded"
  "$ab_python" "$ab_script" --reference "$reference" --candidate "$media" --out "$dir/metal-ab.json" >/dev/null \
    || fail "Metal A/B summary failed for $media against $reference"
  jq -c --arg root "$metal_reference_root" --arg report "$newest" --arg label "$label" \
    --arg summary "$dir/metal-ab.json" --arg summary_sha "$(file_sha256 "$dir/metal-ab.json")" \
    --arg generator "$(jq -r '.generator_commit // "unknown"' <<<"$entry")" \
    '. + {reference_root: $root, reference_report: $report, reference_label: $label,
      reference_generator_commit: $generator, summary_path: $summary, summary_sha256: $summary_sha}' \
    "$dir/metal-ab.json"
}

seal_passed_row() {
  local row="$1" manifest="$2" id media dir generation commit observed
  id="$(jq -r '.id' <<<"$row")"
  dir="$(dirname "$manifest")"
  verify_manifest_hashes "$manifest"
  media="$(jq -r '.media.path // empty' "$manifest")"
  local kind
  kind="$(jq -r '.kind' <<<"$row")"
  if [[ "$kind" == cancellation || "$kind" == fatal_cuda ]]; then
    # Those rows retain logs rather than media; their hashes were verified above.
    jq -e '.stdout_log_path and .server_log_path' "$manifest" >/dev/null \
      || fail "$id: a passed $kind row must retain stdout.log and server.log"
    return 0
  fi
  [[ -n "$media" && -s "$media" ]] || fail "$id: passed row has no retained media"
  [[ "$(file_sha256 "$media")" == "$(jq -r '.media.sha256' "$manifest")" ]] \
    || fail "$id: retained media no longer matches its sealed hash"
  local failure
  if ! failure="$(probe_media_against_expect "$media" "$dir/ffprobe.json" "$row")"; then
    fail "$id: $failure"
  fi
  generation="$(library_row_for "$media" "$row")" || fail "$id: Library provenance mismatch"
  commit="$(generator_commit_of "$generation")" || fail "$id: generator commit mismatch"
  observed="$(observe_provenance "$dir/server.log" "$(jq -r '.server_log_full_path // empty' "$manifest")" \
    "$(jq -c '.provenance_expected' "$manifest")" "$dir")" || fail "$id: provenance mismatch"
  # The matrix is the authority: a row manifest may add observations (the
  # dispatcher line, for instance) but never drop a matrix expectation.
  jq -e --argjson expected "$(jq -c '.expect.provenance' <<<"$row")" \
    '($expected - .provenance_expected) == []' "$manifest" >/dev/null \
    || fail "$id: row manifest dropped a matrix provenance expectation"
  local process_log=null process_log_sha=null
  if [[ -s "$dir/server-process.log" ]]; then
    process_log="\"$dir/server-process.log\""
    process_log_sha="\"$(file_sha256 "$dir/server-process.log")\""
  fi
  local metal_ab
  metal_ab="$(metal_ab_block "$row" "$media" "$dir")"
  # Re-seal the derived fields from what was just re-verified so the report
  # never carries a stale ffprobe, Library row, observation, or A/B summary.
  local tmp="$manifest.tmp.$$"
  jq --argjson generation "$generation" --arg commit "$commit" --argjson observed "$observed" \
    --arg probe "$dir/ffprobe.json" --arg probe_sha "$(file_sha256 "$dir/ffprobe.json")" \
    --argjson bytes "$(file_size "$media")" --argjson metal_ab "${metal_ab:-null}" \
    --argjson ab_row "$([[ -n "$metal_ab" ]] && echo true || echo false)" \
    --argjson process_log "$process_log" --argjson process_log_sha "$process_log_sha" \
    '.generation = $generation | .generator_commit = $commit | .provenance_observed = $observed
     | .media.bytes = $bytes | .media.ffprobe_path = $probe | .media.ffprobe_sha256 = $probe_sha
     | (if $process_log == null
        then del(.server_process_log_path, .server_process_log_sha256)
        else .server_process_log_path = $process_log | .server_process_log_sha256 = $process_log_sha end)
     | if $ab_row then .metal_ab = $metal_ab else del(.metal_ab) end' \
    "$manifest" >"$tmp"
  mv "$tmp" "$manifest"
}

seal_comfy_reference() {
  local checkpoint="$1" manifest="$2"
  if [[ -z "$manifest" && "$contract_test" != 1 ]]; then
    shopt -s nullglob
    local candidates=("$verification_root"/comfyui/reference-*/manifest.json)
    shopt -u nullglob
    local candidate
    for candidate in "${candidates[@]}"; do
      [[ "$(jq -r '.checkpoint' "$candidate")" == "$checkpoint" ]] || continue
      if [[ -z "$manifest" || "$candidate" -nt "$manifest" ]]; then
        manifest="$candidate"
      fi
    done
  fi
  if [[ -z "$manifest" || ! -s "$manifest" ]]; then
    jq -nc --arg reason "no retained ComfyUI CUDA manifest for $checkpoint; run capture-ltx25-comfy-cuda-reference.sh" \
      '{status: "not_run", reason: $reason}'
    return 0
  fi
  jq -e --arg checkpoint "$checkpoint" '
    .schema_version == "mold.ltx25.comfy-cuda-reference.v1"
    and (.status == "passed" or .status == "operator_deferred")
    and .implementation == "ComfyUI" and .backend == "CUDA"
    and .checkpoint == $checkpoint
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
      and (.deferred.guard_cause | IN("pressure_unreadable", "host_memory", "server_rss",
        "timeout", "gpu_unreadable", "torch_cuda_unavailable"))
      and .preservation.downloaded_models_deleted == false
      and .preservation.rendered_media_deleted == false
    end)' "$manifest" >/dev/null || fail "ComfyUI CUDA reference manifest contract mismatch: $manifest"
  local graph
  graph="$(jq -r '.graph.path' "$manifest")"
  [[ -s "$graph" ]] || fail "missing retained ComfyUI graph: $graph"
  [[ "$(file_sha256 "$graph")" == "$(jq -r '.graph.sha256' "$manifest")" ]] \
    || fail "ComfyUI retained graph hash does not match its manifest"
  if [[ "$(jq -r '.status' "$manifest")" == passed ]]; then
    local video
    video="$(jq -r '.video.path' "$manifest")"
    [[ -s "$video" ]] || fail "missing retained ComfyUI video: $video"
    [[ "$(file_sha256 "$video")" == "$(jq -r '.video.sha256' "$manifest")" ]] \
      || fail "ComfyUI retained video hash does not match its manifest"
  else
    local marker log
    marker="$(jq -r '.deferred.resource_guard_marker' "$manifest")"
    log="$(jq -r '.server_log_path' "$manifest")"
    [[ -s "$marker" ]] || fail "missing retained ComfyUI resource-guard evidence: $marker"
    [[ -f "$log" ]] || fail "missing retained ComfyUI server log: $log"
    [[ "$(jq -r '.cause' "$marker")" == "$(jq -r '.deferred.guard_cause' "$manifest")" ]] \
      || fail "ComfyUI guard cause does not match its retained marker"
    [[ "$(file_sha256 "$marker")" == "$(jq -r '.deferred.resource_guard_marker_sha256' "$manifest")" ]] \
      || fail "ComfyUI resource-guard marker hash does not match its evidence seal"
    [[ "$(file_sha256 "$log")" == "$(jq -r '.server_log_sha256' "$manifest")" ]] \
      || fail "ComfyUI server log hash does not match its evidence seal"
  fi
  jq -c --arg path "$manifest" --arg sha "$(file_sha256 "$manifest")" \
    '{status, manifest_path: $path, manifest_sha256: $sha, schema_version, backend, checkpoint}
     + (if .status == "passed" then {video} else {deferred} end)' "$manifest"
}

case "$mode" in
  run) run_mode ;;
  seal) seal_mode ;;
esac
