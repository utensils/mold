#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$repo_root/scripts/capture-ltx25-cuda-verification.sh"
validator="$repo_root/scripts/validate-ltx25-cuda-report.py"
schema="$repo_root/docs/qualification/ltx25-cuda-verification.schema.json"
matrix="$repo_root/scripts/fixtures/ltx25-cuda-matrix.json"
assets="$repo_root/scripts/fixtures/ltx25-assets.json"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() {
  echo "LTX-2.5 CUDA verification contract failed: $*" >&2
  exit 1
}

bash -n "$runner"
python3 -m py_compile "$validator"
[[ -x "$validator" ]] || fail "validator is not executable"
jq -e '
  .properties.schema_version.const == "mold.ltx25.cuda.verification.v1"
  and .properties.backend_scope.const == "cuda"
  and (."$defs".row.properties.status.enum == ["passed", "failed", "blocked", "not_run"])
  and (."$defs".row.properties.reason_source.enum
    == ["admission", "runtime_readiness", "http_status", "oom_envelope"])
  and (."$defs".row.allOf | length) >= 3
  and (.properties.summary.required | index("not_run")) != null
  and (.properties.comfy_reference.required | index("gguf_q4")) != null
' "$schema" >/dev/null || fail "schema lost its row status contract"
grep -Fq -- '--query-compute-apps=pid,gpu_uuid' "$runner" \
  || fail "runner does not bind CUDA observations to the server PID"
grep -Fq 'ENGINE_SHAPING_VARIABLES' "$runner" \
  || fail "runner does not check profile variables against the engine-shaping registry"
if grep -Eq 'sqlite3 (-json|"\$database")' "$runner"; then
  fail "runner shells out to the sqlite3 CLI, which this host does not ship"
fi

mold_home="$tmp/home"
models_dir="$tmp/models"
refs="$tmp/refs"
mkdir -p "$mold_home/output" "$models_dir" "$refs" "$tmp/bin"

# Populate the fake models dir from the manifest-driven fixture: no SHA
# literal lives in this test.
while IFS=$'\t' read -r relative sha; do
  mkdir -p "$models_dir/$(dirname "$relative")"
  printf 'fixture for %s\n' "$relative" >"$models_dir/$relative"
  printf '%s\n' "$sha" >"$models_dir/$relative.sha256-verified"
done < <(jq -r '.assets[] | [.storage_relative_path, .sha256] | @tsv' "$assets")
# LTX-2 / LTX-2.3 regression checkpoints are not in the 2.5 asset table and
# are only asset-listed as present:false when their rows never ran.

for name in ltx-2-upstream comfyui-ltxvideo-upstream comfyui-upstream diffusers-upstream; do
  mkdir -p "$refs/$name/.git"
done

head_sha="$(git -C "$repo_root" rev-parse HEAD)"
short_sha="${head_sha:0:7}"
candle_rev="$(grep -Eo 'utensils/candle\.git\?rev=[0-9a-f]{40}' "$repo_root/Cargo.lock" | head -1 | sed 's/.*rev=//')"
[[ -n "$candle_rev" ]] || fail "could not read the candle fork revision from Cargo.lock"

real_git="$(command -v git)"
cat >"$tmp/bin/git" <<FAKE
#!/usr/bin/env bash
set -euo pipefail
if [[ "\$1" == -C && "\$2" == "$refs/"* ]]; then
  if [[ "\$3" == status && "\$4" == --porcelain ]]; then
    exit 0
  fi
  [[ "\$3" == rev-parse && "\$4" == HEAD ]] || exec "$real_git" "\$@"
  case "\$2" in
    */ltx-2-upstream) echo 400fd31054597515f47125691032c04b1c3ee24e ;;
    */comfyui-ltxvideo-upstream) echo 15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d ;;
    */comfyui-upstream) echo a1079ba16f2674734b065eb036fbfdddaa321a4d ;;
    */diffusers-upstream) echo 95c0d467cc2a4770b71fa25a117320377e6eb08f ;;
  esac
else
  exec "$real_git" "\$@"
fi
FAKE
cat >"$tmp/bin/ffprobe" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
path="${@: -1}"
case "$path" in
  *.mp4)
    printf '%s\n' '{"streams":[{"codec_type":"video","codec_name":"h264","width":256,"height":256,"r_frame_rate":"24/1","nb_frames":"9"},{"codec_type":"audio","codec_name":"aac","sample_rate":"48000","channels":2}],"format":{"size":"12","duration":"0.375"}}' ;;
  *.wav)
    printf '%s\n' '{"streams":[{"codec_type":"audio","codec_name":"pcm_s16le","sample_rate":"48000","channels":2}],"format":{"size":"12","duration":"0.375"}}' ;;
  *)
    printf '%s\n' '{"streams":[{"codec_type":"video","codec_name":"apng","width":256,"height":256,"r_frame_rate":"24/1","nb_read_frames":"9"}],"format":{"size":"13"}}' ;;
esac
FAKE
cat >"$tmp/bin/nvidia-smi" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  --query-gpu=*)
    echo 'GPU-11111111-2222-3333-4444-555555555555, NVIDIA GeForce RTX 4090, 8.9, 580.142, 24564' ;;
  --query-compute-apps=*)
    echo '4242, GPU-11111111-2222-3333-4444-555555555555' ;;
  *)
    echo "fake nvidia-smi: unsupported $*" >&2
    exit 1 ;;
esac
FAKE
# rawvideo rgb24 frames: the Metal reference decodes to one constant, the
# CUDA candidate to another, so PSNR is finite and SSIM is below one.
cat >"$tmp/bin/ffmpeg" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
input=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == -i ]]; then input="$2"; shift; fi
  shift
done
byte='\200'
[[ "$input" == *metal* ]] || byte='\160'
head -c $((256 * 256 * 3 * 9)) /dev/zero | tr '\0' "$byte"
FAKE
cat >"$tmp/bin/nvcc" <<'FAKE'
#!/usr/bin/env bash
printf 'nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.8, V12.8.93\nBuild cuda_12.8.r12.8/compiler.35583870_0\n'
FAKE
cat >"$tmp/bin/free" <<'FAKE'
#!/usr/bin/env bash
printf '               total        used        free      shared  buff/cache   available\nMem:     67059044352 20000000000 30000000000   100000000 17059044352 45000000000\nSwap:              0           0           0\n'
FAKE
cat >"$tmp/bin/mold" <<FAKE
#!/usr/bin/env bash
[[ "\${1:-}" == version ]] && { echo "mold 0.0.0-contract ($short_sha 2026-08-28)"; exit 0; }
echo "fake mold: not a generation binary" >&2
exit 1
FAKE
chmod +x "$tmp/bin/"*

build_json="$tmp/build.json"
jq -n --arg candle_rev "$candle_rev" --arg git_sha "$head_sha" '{
  cargo_command: "cargo build --release -p mold-ai --features h3-cuda,preview,mp4,tui",
  features: ["h3-cuda", "preview", "mp4", "tui"],
  candle_rev: $candle_rev, git_sha: $git_sha}' >"$build_json"

# A fake copy of halcyon's Metal evidence: the newest sealed Metal report
# names the retained media by label, and the harness locates each file by
# basename and authenticates it against the sha256 that report recorded.
metal_ref="$tmp/metal-ref"
mkdir -p "$metal_ref/verification"
printf 'metal silent apng fixture\n' >"$metal_ref/verification/ltx25-fixture-int8-metal-silent-seed-25025.apng"
printf 'metal audio mp4 fixture\n' >"$metal_ref/ltx25-fixture-int8-metal-audio-seed-25026.mp4"
jq -n --arg silent_sha "$(sha256sum "$metal_ref/verification/ltx25-fixture-int8-metal-silent-seed-25025.apng" | awk '{print $1}')" \
  --arg audio_sha "$(sha256sum "$metal_ref/ltx25-fixture-int8-metal-audio-seed-25026.mp4" | awk '{print $1}')" '
  {schema_version:"mold.ltx25.metal-int8.verification.v1", media:[
    {label:"silent_video", path:"/Volumes/ExternalStorage/mold2/output/verification/ltx-2.5/ltx25-fixture-int8-metal-silent-seed-25025.apng",
      sha256:$silent_sha, seed:25025, generator_commit:"fixture"},
    {label:"audio_video", path:"/Volumes/ExternalStorage/mold2/output/ltx25-fixture-int8-metal-audio-seed-25026.mp4",
      sha256:$audio_sha, seed:25026, generator_commit:"fixture"}]}' \
  >"$metal_ref/verification/ltx25-metal-int8-verification-fixture.json"

campaign=fixture
verification_root="$mold_home/output/verification/ltx-2.5/cuda"
report="$verification_root/ltx25-cuda-verification-$campaign.json"
evidence="${report%.json}.d"
rows_dir="$evidence/rows"
mkdir -p "$rows_dir"

sha() { sha256sum "$1" | awk '{print $1}'; }

# --- passed row: int8-conv silent APNG smoke ---------------------------------
passed_id="ltx-2.5-22b-distilled-int8-conv--smoke_silent_apng"
passed_dir="$rows_dir/$passed_id"
mkdir -p "$passed_dir"
printf 'apng fixture bytes\n' >"$passed_dir/output.png"
printf 'Generated in 12.0s\n' >"$passed_dir/stdout.log"
# The row's server-log slice carries the per-render lines; the dispatcher's
# once-per-process line lives only in the profile's full server log, so the
# seal must resolve it through the process scope.
jq -r --arg id "$passed_id" '.rows[] | select(.id == $id) | .expect.provenance[]
  | select(startswith("attention backend selected") | not)' "$matrix" \
  | sed 's/^/INFO mold_inference::ltx2: /' >"$passed_dir/server.log"
echo "INFO mold_inference::attention: attention backend selected backend=Math" >"$evidence/server-default.log"
jq -n '{model:"ltx-2.5-22b-distilled:int8-conv", args:["--width","256"]}' >"$passed_dir/request.json"
printf 'polled_at_utc,memory_used_mib,utilization_gpu\n2026-08-28T00:00:00Z,20000,97\n' >"$passed_dir/vram.csv"
printf 'polled_at_utc,vmhwm_kib\n2026-08-28T00:00:00Z,1234567\n' >"$passed_dir/host.csv"
printf 'polled_at_utc,generation_root_pid,observed_pid,gpu_uuid\n2026-08-28T00:00:00Z,4242,4242,GPU-11111111-2222-3333-4444-555555555555\n' \
  >"$passed_dir/compute.csv"
passed_bytes="$(stat -c '%s' "$passed_dir/output.png")"
jq -n \
  --arg id "$passed_id" --arg dir "$passed_dir" --arg full_log "$evidence/server-default.log" \
  --arg media_sha "$(sha "$passed_dir/output.png")" --arg stdout_sha "$(sha "$passed_dir/stdout.log")" \
  --arg server_sha "$(sha "$passed_dir/server.log")" --arg request_sha "$(sha "$passed_dir/request.json")" \
  --arg vram_sha "$(sha "$passed_dir/vram.csv")" --arg host_sha "$(sha "$passed_dir/host.csv")" \
  --arg compute_sha "$(sha "$passed_dir/compute.csv")" \
  --argjson expected "$(jq -c --arg id "$passed_id" '.rows[] | select(.id == $id) | .expect.provenance' "$matrix")" '
  {schema_version:"mold.ltx25.cuda.row.v1", id:$id, model:"ltx-2.5-22b-distilled:int8-conv",
    case:"smoke_silent_apng", profile:"default", kind:"render", status:"passed",
    started_at:"2026-08-28T00:00:00Z", finished_at:"2026-08-28T00:00:12Z", seconds:12, exit_code:0,
    command:["mold","run","ltx-2.5-22b-distilled:int8-conv","A red fox walking through sunlit desert grass, cinematic natural motion","--width","256"],
    request_path:($dir+"/request.json"), request_sha256:$request_sha,
    stdout_log_path:($dir+"/stdout.log"), stdout_log_sha256:$stdout_sha,
    server_log_path:($dir+"/server.log"), server_log_sha256:$server_sha,
    server_log_full_path:$full_log,
    vram_csv_path:($dir+"/vram.csv"), vram_csv_sha256:$vram_sha,
    host_csv_path:($dir+"/host.csv"), host_csv_sha256:$host_sha,
    compute_observation_path:($dir+"/compute.csv"), compute_observation_sha256:$compute_sha,
    media:{path:($dir+"/output.png"), format:"apng", sha256:$media_sha},
    gpu:{uuid:"GPU-11111111-2222-3333-4444-555555555555", name:"NVIDIA GeForce RTX 4090",
      compute_capability:"8.9", driver_version:"580.142", peak_memory_used_mib:20000,
      server_pid:4242, cuda_work_observed:true},
    host:{peak_vmhwm_kib:1234567},
    provenance_expected:$expected}' >"$passed_dir/manifest.json"

# --- blocked row: Q8_0 refused by the OOM envelope ---------------------------
blocked_id="ltx-2.5-22b-distilled-q8--t2v_audio_mp4"
blocked_dir="$rows_dir/$blocked_id"
mkdir -p "$blocked_dir"
printf 'error: predicted VRAM peak 26.1 GB exceeds the 24 GB device\n' >"$blocked_dir/stdout.log"
jq -n --arg id "$blocked_id" --arg dir "$blocked_dir" \
  --arg stdout_sha "$(sha "$blocked_dir/stdout.log")" '
  {schema_version:"mold.ltx25.cuda.row.v1", id:$id, model:"ltx-2.5-22b-distilled:q8",
    case:"t2v_audio_mp4", profile:"default", kind:"render", status:"blocked",
    reason:"predicted VRAM peak 26.1 GB exceeds the 24 GB device", reason_source:"oom_envelope",
    started_at:"2026-08-28T00:01:00Z", finished_at:"2026-08-28T00:01:01Z", seconds:1, exit_code:1,
    command:["mold","run","ltx-2.5-22b-distilled:q8","prompt"],
    stdout_log_path:($dir+"/stdout.log"), stdout_log_sha256:$stdout_sha}' >"$blocked_dir/manifest.json"

# --- explicit not_run row: a model this build does not register --------------
not_run_id="ltx-2.5-22b-distilled-q4--t2v_silent_apng"
mkdir -p "$rows_dir/$not_run_id"
jq -n --arg id "$not_run_id" '
  {schema_version:"mold.ltx25.cuda.row.v1", id:$id, model:"ltx-2.5-22b-distilled:q4",
    case:"t2v_silent_apng", profile:"default", kind:"render", status:"not_run",
    reason:"model ltx-2.5-22b-distilled:q4 is not registered by this server build"}' \
  >"$rows_dir/$not_run_id/manifest.json"

# --- Library row for the passed media ----------------------------------------
database="$mold_home/mold.db"
python3 - "$database" "$passed_dir" "$passed_bytes" "$short_sha" <<'PY'
import json, sqlite3, sys
db, output_dir, size, short_sha = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
conn = sqlite3.connect(db)
conn.execute("""CREATE TABLE generations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL, output_dir TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL, file_mtime_ms INTEGER, file_size_bytes INTEGER,
  format TEXT NOT NULL, model TEXT NOT NULL, prompt TEXT NOT NULL DEFAULT '',
  negative_prompt TEXT, original_prompt TEXT, seed INTEGER NOT NULL DEFAULT 0,
  steps INTEGER NOT NULL DEFAULT 0, guidance REAL NOT NULL DEFAULT 0.0,
  width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0, strength REAL,
  scheduler TEXT, lora TEXT, lora_scale REAL, frames INTEGER, fps INTEGER,
  metadata_version TEXT NOT NULL DEFAULT '', generation_time_ms INTEGER, backend TEXT,
  hostname TEXT, source TEXT NOT NULL DEFAULT 'unknown',
  metadata_synthetic INTEGER NOT NULL DEFAULT 0, title TEXT, metadata_json TEXT,
  UNIQUE(output_dir, filename))""")
conn.execute("""CREATE TABLE scheduler_estimates (
  estimate_key TEXT PRIMARY KEY, device_class TEXT NOT NULL, model_fingerprint TEXT NOT NULL,
  work_kind TEXT NOT NULL, shape_bucket TEXT NOT NULL, execution_fingerprint TEXT NOT NULL,
  sample_count INTEGER NOT NULL, ewma_total_ms REAL NOT NULL, ewma_load_ms REAL,
  vram_high_water_bytes INTEGER, host_high_water_bytes INTEGER, last_observed_at INTEGER NOT NULL,
  model_family TEXT NOT NULL DEFAULT '', ewma_warm_reload_ms REAL, ewma_prompt_encode_ms REAL,
  ewma_denoise_ms REAL, ewma_vae_ms REAL, ewma_upscale_ms REAL)""")
metadata = {
    "enable_audio": False, "output_format": "apng",
    "version": f"0.0.0-contract ({short_sha} 2026-08-28)",
    "attention_path": "ltx2-bf16-math", "video_only": False, "int8_arm": "native-w8a8",
}
conn.execute(
    "INSERT INTO generations (filename, output_dir, created_at_ms, file_size_bytes, format, model,"
    " prompt, seed, steps, guidance, width, height, frames, fps, generation_time_ms, backend,"
    " hostname, source, metadata_synthetic, metadata_json) VALUES"
    " (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    ("output.png", output_dir, 1, size, "apng", "ltx-2.5-22b-distilled:int8-conv",
     "A red fox walking through sunlit desert grass, cinematic natural motion", 25025, 1, 1.0,
     256, 256, 9, 24, 12000, "cuda", "fixture", "cli", 0, json.dumps(metadata)))
conn.commit()
PY

# --- ComfyUI CUDA manifests: INT8 passed, GGUF deferred ----------------------
comfy_dir="$verification_root/comfyui"
mkdir -p "$comfy_dir/reference-int8" "$comfy_dir/reference-gguf"
printf 'comfy mp4 fixture\n' >"$comfy_dir/reference-int8/ltx25-comfy-int8-cuda-seed-25026.mp4"
cp "$repo_root/scripts/fixtures/ltx25-comfy-cuda-int8-api-prompt.json" "$comfy_dir/reference-int8/prompt.json"
cp "$repo_root/scripts/fixtures/ltx25-comfy-cuda-gguf-q4-api-prompt.json" "$comfy_dir/reference-gguf/prompt.json"
comfy_int8="$comfy_dir/reference-int8/manifest.json"
comfy_gguf="$comfy_dir/reference-gguf/manifest.json"
jq -n --arg video "$comfy_dir/reference-int8/ltx25-comfy-int8-cuda-seed-25026.mp4" \
  --arg video_sha "$(sha "$comfy_dir/reference-int8/ltx25-comfy-int8-cuda-seed-25026.mp4")" \
  --arg graph "$comfy_dir/reference-int8/prompt.json" --arg graph_sha "$(sha "$comfy_dir/reference-int8/prompt.json")" '
  {schema_version:"mold.ltx25.comfy-cuda-reference.v1", status:"passed", implementation:"ComfyUI",
    backend:"CUDA", checkpoint:"distilled INT8 ConvRot",
    settings:{width:256,height:256,frames:9,fps:24,stage1_seed:25026,stage2_seed:42,
      video_cfg:1,audio_cfg:1}, graph:{path:$graph,sha256:$graph_sha},
    video:{path:$video,sha256:$video_sha,ffprobe:{streams:[
      {codec_type:"video",width:256,height:256,r_frame_rate:"24/1",nb_frames:"9"},
      {codec_type:"audio",sample_rate:"48000",channels:2}]}}, retained_in_library:true}' \
  >"$comfy_int8"
guard_marker="$comfy_dir/reference-gguf/resource-guard-aborted"
printf '%s\n' '{"cause":"torch_cuda_unavailable","elapsed_seconds":0}' >"$guard_marker"
: >"$comfy_dir/reference-gguf/comfyui-server.log"
jq -n --arg graph "$comfy_dir/reference-gguf/prompt.json" --arg graph_sha "$(sha "$comfy_dir/reference-gguf/prompt.json")" \
  --arg marker "$guard_marker" --arg marker_sha "$(sha "$guard_marker")" \
  --arg log "$comfy_dir/reference-gguf/comfyui-server.log" --arg log_sha "$(sha "$comfy_dir/reference-gguf/comfyui-server.log")" '
  {schema_version:"mold.ltx25.comfy-cuda-reference.v1", status:"operator_deferred",
    implementation:"ComfyUI", backend:"CUDA", checkpoint:"distilled GGUF Q4_K_M",
    settings:{width:256,height:256,frames:9,fps:24,stage1_seed:25026,stage2_seed:42,
      video_cfg:1,audio_cfg:1}, graph:{path:$graph,sha256:$graph_sha}, video:null,
    server_log_path:$log, server_log_sha256:$log_sha, retained_in_library:false,
    deferred:{guard_cause:"torch_cuda_unavailable", resource_guard_marker:$marker,
      resource_guard_marker_sha256:$marker_sha},
    preservation:{downloaded_models_deleted:false,rendered_media_deleted:false}}' \
  >"$comfy_gguf"

# The ambient shell may carry direnv's MOLD_* exports; every invocation pins
# the fixture home, store, and database explicitly.
run_seal() {
  PATH="$tmp/bin:$PATH" \
    MOLD_HOME="$mold_home" MOLD_MODELS_DIR="${MODELS_DIR_OVERRIDE-$models_dir}" \
    MOLD_DB_PATH="$mold_home/mold.db" \
    LTX25_METAL_REFERENCE_ROOT="${METAL_REF_OVERRIDE-$metal_ref}" \
    LTX25_ALLOW_TEST_HOME=1 LTX25_CONTRACT_TEST=1 LTX25_SKIP_GATES=1 \
    LTX25_REFERENCES_ROOT="$refs" LTX25_CAMPAIGN="$campaign" \
    LTX25_MOLD_BIN="${MOLD_BIN_OVERRIDE-$tmp/bin/mold}" LTX25_BUILD_JSON="$build_json" \
    LTX25_COMFY_INT8_MANIFEST="$comfy_int8" LTX25_COMFY_GGUF_MANIFEST="$comfy_gguf" \
    "$runner" "$@"
}

run_seal --seal >/dev/null
[[ -s "$report" ]] || fail "seal did not write $report"
matrix_rows="$(jq '.rows | length' "$matrix")"
jq -e --argjson rows "$matrix_rows" --arg mold_home "$mold_home" --arg models_dir "$models_dir" \
  --arg metal_ref "$metal_ref" '
  .schema_version == "mold.ltx25.cuda.verification.v1"
  and .backend_scope == "cuda"
  and .mold_home == $mold_home and .models_dir == $models_dir
  and .models_dir != ($mold_home + "/models")
  and .source_tree_state == "contract_test"
  and .qualification_status == "not_qualified_contract_test"
  and (.rows | length) == $rows
  and ([.rows[].id] | unique | length) == $rows
  and .summary == {passed:1, failed:0, blocked:1, not_run:($rows - 2)}
  and (.rows[] | select(.status == "blocked") | .reason_source) == "oom_envelope"
  and ([.rows[] | select(.status == "not_run") | .reason] | all(length > 0))
  and (.rows[] | select(.case == "v2v") | .reason | test("fails closed"))
  and (.rows[] | select(.id == "ltx-2.5-22b-distilled-int8-conv--smoke_silent_apng")
    | .status == "passed" and .generation.backend == "cuda"
      and .provenance_observed == (.provenance_expected | map({line: .,
        scope: (if startswith("attention backend selected") then "process" else "slice" end)}))
      and any(.provenance_observed[]; .scope == "process")
      and .media.sha256 != null
      and .metal_ab.reference_file == ($metal_ref + "/verification/ltx25-fixture-int8-metal-silent-seed-25025.apng")
      and (.metal_ab.reference_sha256 | test("^[0-9a-f]{64}$"))
      and .metal_ab.candidate_sha256 == .media.sha256
      and .metal_ab.parity == {frames:{reference:9,candidate:9,equal:true},
        width:{reference:256,candidate:256,equal:true}, height:{reference:256,candidate:256,equal:true}}
      and .metal_ab.frames_compared == 9
      and (.metal_ab.psnr_db.mean | type) == "number" and .metal_ab.psnr_db.mean > 0
      and (.metal_ab.ssim.mean | type) == "number" and .metal_ab.ssim.mean < 1
      and .metal_ab.identical == false
      and (.metal_ab.summary_sha256 | test("^[0-9a-f]{64}$")))
  and ([.rows[] | select(.status != "passed") | has("metal_ab")] | all(. == false))
  and (.assets | length) == 45
  and ([.assets[].present] | all(. == true))
  and ([.assets[].actual_sha256] | all(. == null))
  and ([.assets[].path] | all(startswith($models_dir + "/")))
  and (.references | length) == 4
  and ([.references[].status] | all(. == "pinned_clean"))
  and ([.gates[].status] | all(. == "skipped_contract_test"))
  and (.server_profiles | map(.name)) == ["default", "attn_f32", "flash", "int8_dequant", "qmatmul"]
  and .host.gpus[0].compute_capability == "8.9"
  and .host.nvcc == "Build cuda_12.8.r12.8/compiler.35583870_0"
  and .host.host_ram_bytes == 67059044352
  and .host.build.candle_rev == (.host.build.cargo_lock_candle_rev)
  and .comfy_reference.int8.status == "passed"
  and .comfy_reference.gguf_q4.status == "operator_deferred"
  and .comfy_reference.gguf_q4.deferred.guard_cause == "torch_cuda_unavailable"
  and .preservation == {downloaded_models_deleted:false, rendered_media_deleted:false}
' "$report" >/dev/null || fail "sealed report shape mismatch"
"$validator" "$report" >/dev/null || fail "validator rejected the sealed fixture report"

# A per-render line may never be satisfied from the process scope.
cp "$evidence/server-default.log" "$tmp/full.bak"
grep 'int8 arm' "$passed_dir/server.log" >>"$evidence/server-default.log"
cp "$passed_dir/server.log" "$tmp/server.bak"
grep -v 'int8 arm' "$tmp/server.bak" >"$passed_dir/server.log"
if run_seal --seal >/dev/null 2>&1; then
  fail "runner satisfied a per-render provenance line from the whole server log"
fi
cp "$tmp/server.bak" "$passed_dir/server.log"
cp "$tmp/full.bak" "$evidence/server-default.log"
run_seal --seal >/dev/null

# The process-scoped dispatcher evidence is retained per row and hash-bound.
grep -Fq 'attention backend selected backend=Math' "$passed_dir/server-process.log" \
  || fail "seal did not retain the process-scoped line in server-process.log"
jq -e '.server_process_log_sha256 | test("^[0-9a-f]{64}$")' "$passed_dir/manifest.json" >/dev/null \
  || fail "row manifest does not hash-bind server-process.log"
cp "$passed_dir/server-process.log" "$tmp/process.bak"
printf 'tampered\n' >>"$passed_dir/server-process.log"
if "$validator" "$report" >/dev/null 2>&1; then
  fail "validator accepted a mutated server-process.log"
fi
if run_seal --seal >/dev/null 2>&1; then
  fail "seal accepted a mutated server-process.log"
fi
cp "$tmp/process.bak" "$passed_dir/server-process.log"
run_seal --seal >/dev/null
"$validator" "$report" >/dev/null || fail "restored server-process.log did not seal"

# Without a Metal reference root the block is null, never a failure.
METAL_REF_OVERRIDE="$tmp/no-such-reference" run_seal --seal >/dev/null
jq -e '(.rows[] | select(.status == "passed") | has("metal_ab") and .metal_ab == null)' "$report" >/dev/null \
  || fail "missing Metal reference root must record metal_ab as null"
"$validator" "$report" >/dev/null || fail "validator rejected a null metal_ab block"
# A reference copy whose bytes disagree with the Metal report is refused.
printf 'tampered\n' >>"$metal_ref/verification/ltx25-fixture-int8-metal-silent-seed-25025.apng"
if run_seal --seal >/dev/null 2>&1; then
  fail "runner accepted a Metal reference that disagrees with its sealed report"
fi
printf 'metal silent apng fixture\n' >"$metal_ref/verification/ltx25-fixture-int8-metal-silent-seed-25025.apng"
run_seal --seal >/dev/null

# --- the validator rejects mutated evidence ----------------------------------
printf 'tampered\n' >>"$passed_dir/stdout.log"
if "$validator" "$report" >/dev/null 2>&1; then
  fail "validator accepted a mutated stdout.log"
fi
if run_seal --seal >/dev/null 2>&1; then
  fail "seal accepted a mutated stdout.log"
fi
printf 'Generated in 12.0s\n' >"$passed_dir/stdout.log"
run_seal --seal >/dev/null
"$validator" "$report" >/dev/null

# --- negatives ---------------------------------------------------------------
marker="$models_dir/shared/ltx2/model_patches/ltx-2.5-duration-head-bf16.safetensors.sha256-verified"
cp "$marker" "$tmp/marker.bak"
printf '%064d\n' 0 >"$marker"
if run_seal --seal >/dev/null 2>&1; then
  fail "runner accepted a mismatched verified SHA marker"
fi
cp "$tmp/marker.bak" "$marker"

cp "$blocked_dir/manifest.json" "$tmp/blocked.bak"
jq 'del(.reason)' "$tmp/blocked.bak" >"$blocked_dir/manifest.json"
if run_seal --seal >/dev/null 2>&1; then
  fail "runner accepted a blocked row without a reason"
fi
jq '.reason_source = "gut_feeling"' "$tmp/blocked.bak" >"$blocked_dir/manifest.json"
if run_seal --seal >/dev/null 2>&1; then
  fail "runner accepted a blocked row with an unknown reason source"
fi
cp "$tmp/blocked.bak" "$blocked_dir/manifest.json"

python3 - "$database" <<'PY'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("UPDATE generations SET backend = 'metal'")
conn.commit()
PY
if run_seal --seal >/dev/null 2>&1; then
  fail "runner accepted a passed row whose Library backend is metal"
fi
python3 - "$database" <<'PY'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("UPDATE generations SET backend = 'cuda'")
conn.commit()
PY

cp "$passed_dir/server.log" "$tmp/server.bak"
grep -v 'audio branch' "$tmp/server.bak" >"$passed_dir/server.log"
if run_seal --seal >/dev/null 2>&1; then
  fail "runner accepted a passed row missing an expected provenance line"
fi
cp "$tmp/server.bak" "$passed_dir/server.log"

if PATH="$tmp/bin:$PATH" MOLD_HOME="$mold_home" MOLD_MODELS_DIR="" MOLD_DB_PATH="$mold_home/mold.db" \
  LTX25_ALLOW_TEST_HOME=1 LTX25_CONTRACT_TEST=1 \
  LTX25_SKIP_GATES=1 LTX25_REFERENCES_ROOT="$refs" LTX25_CAMPAIGN="$campaign" \
  LTX25_MOLD_BIN="$tmp/bin/mold" LTX25_BUILD_JSON="$build_json" \
  LTX25_COMFY_INT8_MANIFEST="$comfy_int8" LTX25_COMFY_GGUF_MANIFEST="$comfy_gguf" \
  "$runner" --seal >/dev/null 2>&1; then
  fail "runner ran without MOLD_MODELS_DIR"
fi
mkdir -p "$mold_home/models"
if MODELS_DIR_OVERRIDE="$mold_home/models" run_seal --seal >/dev/null 2>&1; then
  fail "runner accepted MOLD_MODELS_DIR colliding with MOLD_HOME/models"
fi

cat >"$tmp/bin/mold-wrong" <<'FAKE'
#!/usr/bin/env bash
echo "mold 0.0.0-contract (0000000 2026-08-28)"
FAKE
chmod +x "$tmp/bin/mold-wrong"
if MOLD_BIN_OVERRIDE="$tmp/bin/mold-wrong" run_seal --seal >/dev/null 2>&1; then
  fail "runner accepted a binary built from another commit"
fi

# --- --run preflight: profile variables must be registered ------------------
registry="$tmp/runtime_env.rs"
cat >"$registry" <<'RS'
pub const ENGINE_SHAPING_VARIABLES: &[&str] = &[
    "MOLD_ATTN",
    "MOLD_LTX2_ATTN_F32",
    "MOLD_LTX2_INT8",
    "MOLD_LTX2_QMATMUL",
];
RS
run_preflight() {
  LTX25_RUNTIME_ENV_SOURCE="$registry" LTX25_RUN_PREFLIGHT_ONLY=1 \
    LTX25_SERVER_PROFILE="$1" MOLD_HOST=http://127.0.0.1:1 MOLD_API_KEY=contract \
    LTX25_SERVER_PID=4242 LTX25_SERVER_LOG="$evidence/server-default.log" \
    run_seal --run
}
run_preflight qmatmul >/dev/null || fail "registered profile variable was refused"
sed -i '/MOLD_LTX2_QMATMUL/d' "$registry"
if run_preflight qmatmul >/dev/null 2>&1; then
  fail "runner accepted a profile whose variable is not in ENGINE_SHAPING_VARIABLES"
fi
run_preflight default >/dev/null || fail "the default profile needs no registration"
if run_preflight nonesuch >/dev/null 2>&1; then
  fail "runner accepted an undefined server profile"
fi

echo "LTX-2.5 CUDA verification contract OK"
