#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$repo_root/scripts/capture-ltx25-metal-verification.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

bash -n "$runner"
mkdir -p "$tmp/home/output/verification/ltx-2.5" "$tmp/home/output" "$tmp/refs"

make_asset() {
  local relative="$1" sha="$2" path
  path="$tmp/home/$relative"
  mkdir -p "$(dirname "$path")"
  printf 'fixture for %s\n' "$relative" >"$path"
  printf '%s\n' "$sha" >"$path.sha256-verified"
}

make_asset \
  models/ltx-2.5-22b-distilled-int8-conv/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  c4279eeff115cbeaca494bd2183e7d768c38fe85a184dc6afbb7159157c44334
make_asset models/shared/ltx2/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f
make_asset models/shared/ltx2/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d
make_asset models/shared/ltx2/vae/ltx-2.5-audio-vae-bf16.safetensors \
  c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5
make_asset models/shared/ltx2/model_patches/ltx-2.5-duration-head-bf16.safetensors \
  2ec71e4206ed365d015f00c05a48caccfb0ee862986809d06ae376c09f5d9190
make_asset models/shared/ltx2/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8
make_asset models/shared/ltx2/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors \
  2bc3300f2b3c3c1834d72164fbf13a3b9fd73e5a741e8a2c3f4035f89a75c3fe

for name in ltx-2-upstream comfyui-ltxvideo-upstream comfyui-upstream diffusers-upstream; do
  mkdir -p "$tmp/refs/$name/.git"
done

mkdir -p "$tmp/bin"
real_git="$(command -v git)"
cat >"$tmp/bin/git" <<FAKE
#!/usr/bin/env bash
set -euo pipefail
if [[ "\$1" == -C && "\$2" == "$tmp/refs/"* ]]; then
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
chmod +x "$tmp/bin/git"
cat >"$tmp/bin/ffprobe" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
path="${@: -1}"
if [[ "$path" == *.mp4 ]]; then
  printf '%s\n' '{"streams":[{"codec_type":"video","codec_name":"h264","width":256,"height":256,"r_frame_rate":"24/1","nb_frames":"9"},{"codec_type":"audio","codec_name":"aac","sample_rate":"48000","channels":2}],"format":{"size":"12","duration":"0.375"}}'
else
  printf '%s\n' '{"streams":[{"codec_type":"video","codec_name":"apng","width":256,"height":256,"r_frame_rate":"24/1","nb_read_frames":"9"}],"format":{"size":"13"}}'
fi
FAKE
chmod +x "$tmp/bin/ffprobe"
printf 'mp4 fixture\n' >"$tmp/home/output/ltx25-final-int8-metal-audio-seed-25026.mp4"
printf 'apng fixture\n' >"$tmp/home/output/verification/ltx-2.5/phase2-int8-metal-smoke-seed-25025.apng"
database="$tmp/home/mold.db"
sqlite3 "$database" 'CREATE TABLE generations (
  id INTEGER PRIMARY KEY, filename TEXT, output_dir TEXT, format TEXT, title TEXT,
  prompt TEXT, model TEXT, seed INTEGER, steps INTEGER, guidance REAL, width INTEGER,
  height INTEGER, frames INTEGER, fps INTEGER, generation_time_ms INTEGER,
  backend TEXT, hostname TEXT, source TEXT, metadata_synthetic INTEGER,
  file_size_bytes INTEGER, metadata_json TEXT);'
sqlite3 "$database" "INSERT INTO generations VALUES
  (1, 'ltx25-final-int8-metal-audio-seed-25026.mp4', '$tmp/home/output', 'mp4',
   'fixture', 'A small brass automaton drummer performing in gentle rain, locked camera, cinematic reflections',
   'ltx-2.5-22b-distilled:int8-conv', 25026, 1, 1, 256, 256, 9, 24, 1,
   'metal', 'fixture', 'cli', 0, 12, '{\"enable_audio\":true,\"output_format\":\"mp4\"}'),
  (2, 'phase2-int8-metal-smoke-seed-25025.apng', '$tmp/home/output/verification/ltx-2.5',
   'apng', NULL, 'A red fox walking through sunlit desert grass, cinematic natural motion',
   'ltx-2.5-22b-distilled:int8-conv', 25025, 1, 1, 256, 256, 9, 24, 1,
   'metal', 'fixture', 'cli', 0, 13, '{\"enable_audio\":false,\"output_format\":\"apng\"}');"

report="$tmp/report.json"
PATH="$tmp/bin:$PATH" \
MOLD_HOME="$tmp/home" \
LTX25_ALLOW_TEST_HOME=1 \
LTX25_CONTRACT_TEST=1 \
LTX25_REFERENCES_ROOT="$tmp/refs" \
LTX25_REPORT="$report" \
LTX25_CAPTURE_TIMESTAMP=fixture \
LTX25_SKIP_GATES=1 \
LTX25_HOST_JSON='{"os":"fixture","arch":"arm64","metal_devices":[{"name":"fixture"}]}' \
  "$runner" >/dev/null

jq -e '
  .schema_version == "mold.ltx25.metal-int8.verification.v1"
  and .backend_scope == "metal"
  and .default_model == "ltx-2.5-22b-distilled:int8-conv"
  and (.assets | length) == 7
  and ([.assets[].expected_sha256] | all(test("^[0-9a-f]{64}$")))
  and ([.assets[].actual_sha256] | all(. == null))
  and (.references | length) == 4
  and ([.references[].status] | all(. == "pinned_clean"))
  and (.media | length) == 2
  and ([.media[].retained_in_library] | all(. == true))
  and ([.gates[].status] | all(. == "skipped_contract_test"))
  and (.comparison_matrix[] | select(.implementation == "Mold"
    and .backend == "Metal" and .checkpoint == "distilled INT8 ConvRot").status)
    == "not_qualified_contract_test"
  and (.comparison_matrix[] | select(.implementation == "ComfyUI").status) == "pending"
  and .preservation.downloaded_models_deleted == false
  and .preservation.rendered_media_deleted == false
' "$report" >/dev/null

bad_marker="$tmp/home/models/shared/ltx2/model_patches/ltx-2.5-duration-head-bf16.safetensors.sha256-verified"
printf '%064d\n' 0 >"$bad_marker"
if PATH="$tmp/bin:$PATH" MOLD_HOME="$tmp/home" LTX25_ALLOW_TEST_HOME=1 \
  LTX25_CONTRACT_TEST=1 LTX25_REFERENCES_ROOT="$tmp/refs" \
  LTX25_REPORT="$tmp/bad.json" LTX25_SKIP_GATES=1 \
  LTX25_HOST_JSON='{}' "$runner" >/dev/null 2>&1; then
  echo "runner accepted a mismatched verified SHA marker" >&2
  exit 1
fi

echo "LTX-2.5 Metal verification contract OK"
