#!/usr/bin/env bash
set -euo pipefail

# Contract test for the Hunyuan3D Apple Metal qualification harness.
#
# Exercises the real runners with a fake `mold`, a fake timing command and a
# fake comparison interpreter, so the evidence layout, the parsing of the CLI
# save summary, the peak-RSS parse and the report shape are all covered
# without a GPU, a checkpoint or a ComfyUI server.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
uat_runner="$repo_root/scripts/capture-hunyuan3d-metal-uat.sh"
comfy_runner="$repo_root/scripts/capture-hunyuan3d-comfy-metal-reference.sh"
compare_script="$repo_root/scripts/hunyuan3d-mesh-compare.py"
frame_script="$repo_root/scripts/hunyuan3d-frame-source.py"
fixture="$repo_root/scripts/fixtures/hunyuan3d-comfy-metal-api-prompt.json"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() {
  echo "hunyuan3d UAT contract failed: $*" >&2
  exit 1
}

bash -n "$uat_runner"
bash -n "$comfy_runner"
[[ -f "$compare_script" ]] || fail "missing $compare_script"
[[ -f "$frame_script" ]] || fail "missing $frame_script"

# ---------------------------------------------------------------------------
# The API graph must keep the node ids and input names the ComfyUI sources use.
# ---------------------------------------------------------------------------
jq -e '
  length == 10
  and .["1"].class_type == "ImageOnlyCheckpointLoader"
  and .["2"].class_type == "LoadImage"
  and .["3"].class_type == "CLIPVisionEncode"
  and .["3"].inputs.crop == "center"
  and .["4"].class_type == "Hunyuan3Dv2Conditioning"
  and .["5"].class_type == "ModelSamplingAuraFlow"
  and .["5"].inputs.shift == 1.0
  and .["6"].class_type == "EmptyLatentHunyuan3Dv2"
  and .["6"].inputs.resolution == 3072
  and .["7"].class_type == "KSampler"
  and .["7"].inputs.cfg == 1.0
  and .["7"].inputs.steps == 5
  and .["7"].inputs.sampler_name == "euler"
  and .["7"].inputs.scheduler == "normal"
  and .["8"].class_type == "VAEDecodeHunyuan3D"
  and .["8"].inputs.num_chunks == 8000
  and .["9"].class_type == "VoxelToMesh"
  and .["9"].inputs.algorithm == "surface net"
  and .["9"].inputs.threshold == 0.6
  and .["10"].class_type == "SaveGLB"
' "$fixture" >/dev/null || fail "API graph fixture does not match the ComfyUI node contract"

HUNYUAN3D_COMFY_VALIDATE_ONLY=1 "$comfy_runner" >/dev/null

rendered="$(HUNYUAN3D_COMFY_RENDER_GRAPH_ONLY=1 HUNYUAN3D_SEED=4242 \
  HUNYUAN3D_RENDER_OCTREE=320 HUNYUAN3D_RENDER_CKPT=model.fp16.safetensors \
  HUNYUAN3D_RENDER_IMAGE=cutout.png HUNYUAN3D_RENDER_PREFIX=3d/rung \
  "$comfy_runner")"
jq -e '
  .["1"].inputs.ckpt_name == "model.fp16.safetensors"
  and .["2"].inputs.image == "cutout.png"
  and .["7"].inputs.seed == 4242
  and .["8"].inputs.octree_resolution == 320
  and .["10"].inputs.filename_prefix == "3d/rung"
' <<<"$rendered" >/dev/null || fail "graph substitution did not take"

# ---------------------------------------------------------------------------
# The oracle is pinned: a ComfyUI that has moved is a different reference.
# ---------------------------------------------------------------------------
pinned_commit=7fe8a6138504f90ff7be82f3babf416da32876b1
# The runner prepends `$HOME/.nix-profile/bin` to PATH so a long capture cannot
# cache a collectable devshell shim. That is also the only entry ahead of the
# system tools, so the fake git goes there rather than fighting the reset.
fake_home="$tmp/fakehome"
mkdir -p "$fake_home/.nix-profile/bin" "$tmp/comfy"
cat > "$fake_home/.nix-profile/bin/git" <<FAKE
#!/usr/bin/env bash
set -euo pipefail
case "\$1" in
  rev-parse) printf '%s\n' "\${HUNYUAN3D_FAKE_COMMIT:-$pinned_commit}" ;;
  status) printf '%s' "\${HUNYUAN3D_FAKE_STATUS:-}" ;;
  *) echo "fake git: unexpected \$*" >&2; exit 2 ;;
esac
FAKE
chmod +x "$fake_home/.nix-profile/bin/git"

pin_probe() {
  HOME="$fake_home" HUNYUAN3D_COMFY_TEST_PIN=1 \
    HUNYUAN3D_COMFY_ROOT="$tmp/comfy" "$@" "$comfy_runner"
}

[[ "$(pin_probe env)" == "$pinned_commit false" ]] \
  || fail "the default pin must pass against a clone at that commit"

if pin_probe env HUNYUAN3D_FAKE_COMMIT=0000000000000000000000000000000000000000 \
  >/dev/null 2>&1; then
  fail "runner accepted a ComfyUI clone that is not at the pinned commit"
fi
pin_probe env HUNYUAN3D_FAKE_COMMIT=0000000000000000000000000000000000000000 \
  > /dev/null 2> "$tmp/pin.stderr" || true
grep -q 'HUNYUAN3D_COMFY_COMMIT' "$tmp/pin.stderr" \
  || fail "the pin mismatch message must name HUNYUAN3D_COMFY_COMMIT"

if pin_probe env HUNYUAN3D_FAKE_STATUS=' M comfy/sd.py' >/dev/null 2>&1; then
  fail "runner accepted a dirty ComfyUI clone without an explicit override"
fi
[[ "$(pin_probe env HUNYUAN3D_FAKE_STATUS=' M comfy/sd.py' \
  HUNYUAN3D_COMFY_ALLOW_DIRTY=1)" == "$pinned_commit true" ]] \
  || fail "HUNYUAN3D_COMFY_ALLOW_DIRTY=1 must admit a dirty clone and record it"

# Re-pinning deliberately is what the message tells the operator to do.
[[ "$(pin_probe env HUNYUAN3D_FAKE_COMMIT=abc HUNYUAN3D_COMFY_COMMIT=abc)" == "abc false" ]] \
  || fail "HUNYUAN3D_COMFY_COMMIT must re-pin the oracle"

# ---------------------------------------------------------------------------
# Fakes.
# ---------------------------------------------------------------------------
mkdir -p "$tmp/bin" "$tmp/home" "$tmp/evidence"

glb_b64='Z2xURgIAAACgAwAAlAIAAEpTT057ImFzc2V0IjogeyJ2ZXJzaW9uIjogIjIuMCIsICJnZW5lcmF0
b3IiOiAiaHVueXVhbjNkLW1lc2gtY29tcGFyZSBzZWxmLXRlc3QifSwgImJ1ZmZlcnMiOiBbeyJi
eXRlTGVuZ3RoIjogMjQwfV0sICJidWZmZXJWaWV3cyI6IFt7ImJ1ZmZlciI6IDAsICJieXRlT2Zm
c2V0IjogMCwgImJ5dGVMZW5ndGgiOiA5NiwgInRhcmdldCI6IDM0OTYyfSwgeyJidWZmZXIiOiAw
LCAiYnl0ZU9mZnNldCI6IDk2LCAiYnl0ZUxlbmd0aCI6IDE0NCwgInRhcmdldCI6IDM0OTYzfV0s
ICJhY2Nlc3NvcnMiOiBbeyJidWZmZXJWaWV3IjogMCwgImJ5dGVPZmZzZXQiOiAwLCAiY29tcG9u
ZW50VHlwZSI6IDUxMjYsICJjb3VudCI6IDgsICJ0eXBlIjogIlZFQzMiLCAibWluIjogWy0xLjAs
IC0xLjAsIC0xLjBdLCAibWF4IjogWzEuMCwgMS4wLCAxLjBdfSwgeyJidWZmZXJWaWV3IjogMSwg
ImJ5dGVPZmZzZXQiOiAwLCAiY29tcG9uZW50VHlwZSI6IDUxMjUsICJjb3VudCI6IDM2LCAidHlw
ZSI6ICJTQ0FMQVIifV0sICJtZXNoZXMiOiBbeyJwcmltaXRpdmVzIjogW3siYXR0cmlidXRlcyI6
IHsiUE9TSVRJT04iOiAwfSwgImluZGljZXMiOiAxLCAibW9kZSI6IDR9XX1dLCAibm9kZXMiOiBb
eyJtZXNoIjogMH1dLCAic2NlbmVzIjogW3sibm9kZXMiOiBbMF19XSwgInNjZW5lIjogMH3wAAAA
QklOAAAAgL8AAIC/AACAvwAAgD8AAIC/AACAvwAAgD8AAIA/AACAvwAAgL8AAIA/AACAvwAAgL8A
AIC/AACAPwAAgD8AAIC/AACAPwAAgD8AAIA/AACAPwAAgL8AAIA/AACAPwAAAAABAAAAAgAAAAAA
AAACAAAAAwAAAAQAAAAGAAAABQAAAAQAAAAHAAAABgAAAAAAAAAEAAAABQAAAAAAAAAFAAAAAQAA
AAEAAAAFAAAABgAAAAEAAAAGAAAAAgAAAAIAAAAGAAAABwAAAAIAAAAHAAAAAwAAAAMAAAAHAAAA
BAAAAAMAAAAEAAAAAAAAAA=='
printf '%s' "$glb_b64" | tr -d '\n' | base64 -d > "$tmp/cube.glb"
[[ -s "$tmp/cube.glb" ]] || fail "could not materialise the fixture GLB"

printf 'fake cutout\n' > "$tmp/source.png"

cat > "$tmp/bin/fake-mold" <<FAKE
#!/usr/bin/env bash
set -euo pipefail
output=""
octree=""
seed=""
image=""
while (( \$# > 0 )); do
  case "\$1" in
    -o|--output) shift; output="\$1" ;;
    --octree) shift; octree="\$1" ;;
    --seed) shift; seed="\$1" ;;
    --image) shift; image="\$1" ;;
  esac
  shift || true
done
[[ -n "\$output" ]] || { echo "fake mold: no --output" >&2; exit 3; }
# Recorded so the test can prove mold was handed the FRAMED picture.
printf '%s\n' "\$image" >> "\${HUNYUAN3D_FAKE_IMAGE_LOG:-/dev/null}"
cp "$tmp/cube.glb" "\$output"
if [[ "\${HUNYUAN3D_FAKE_NO_SUMMARY:-0}" == 1 ]]; then
  printf 'done\n' >&2
  exit 0
fi
printf '%s\n' "  Saved: \$output (\$(( octree * 10 )) verts, \$(( octree * 20 )) tris, 1.98 units wide)" >&2
printf 'seed \$seed\n' >&2
FAKE
chmod +x "$tmp/bin/fake-mold"

# /usr/bin/time cannot be shadowed on PATH, so the runner takes the whole
# timing command as a parameter. This stands in for `/usr/bin/time -l`.
cat > "$tmp/bin/fake-time" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
"$@"
status=$?
{
  printf '        1.42 real         0.90 user         0.20 sys\n'
  printf '          1234567890  maximum resident set size\n'
  printf '                 512  peak memory footprint\n'
} >&2
exit "$status"
FAKE
chmod +x "$tmp/bin/fake-time"

# One fake interpreter for both helper scripts, dispatching on the script it
# was handed. Getting the wiring backwards is then a hard failure rather than a
# report that quietly says nothing was framed.
cat > "$tmp/bin/fake-python" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
script="${1:-}"
shift || true
out=""
png=""
mold=""
comfy=""
floor_a=""
input=""
frame_out=""
while (( $# > 0 )); do
  case "$1" in
    --out) shift; out="$1" ;;
    --png) shift; png="$1" ;;
    --mold) shift; mold="$1" ;;
    --comfy) shift; comfy="$1" ;;
    --floor-a) shift; floor_a="$1" ;;
    --input) shift; input="$1" ;;
    --output) shift; frame_out="$1" ;;
  esac
  shift || true
done

case "$(basename "$script")" in
  hunyuan3d-frame-source.py)
    [[ -s "$input" ]] || { echo "fake frame: missing --input $input" >&2; exit 7; }
    [[ -n "$frame_out" ]] || { echo "fake frame: no --output" >&2; exit 8; }
    printf 'fake framed png\n' > "$frame_out"
    printf 'framed %s -> %s\n' "$input" "$frame_out"
    exit 0
    ;;
  hunyuan3d-mesh-compare.py) ;;
  *)
    echo "fake python: unexpected script $script" >&2
    exit 9
    ;;
esac

[[ -n "$out" ]] || { echo "fake compare: no --out" >&2; exit 3; }
[[ -s "$mold" ]] || { echo "fake compare: missing mold mesh $mold" >&2; exit 4; }
[[ -s "$comfy" ]] || { echo "fake compare: missing comfy mesh $comfy" >&2; exit 5; }
[[ -n "$floor_a" ]] || { echo "fake compare: no noise floor supplied" >&2; exit 6; }
cat > "$out" <<JSON
{"schema_version":"mold.hunyuan3d.mesh-compare.v1","pass":true,"reasons":[],
 "chamfer_normalized":0.01,"face_count_relative_difference":0.0,
 "extent_relative_differences":[0.0,0.0,0.0]}
JSON
[[ -z "$png" ]] || printf 'fake png\n' > "$png"
printf '{"pass": true, "reasons": []}\n'
FAKE
chmod +x "$tmp/bin/fake-python"

# ---------------------------------------------------------------------------
# A ComfyUI reference manifest with two of the three rungs, so both the
# compared and the missing-rung paths are exercised in one run.
# ---------------------------------------------------------------------------
evidence="$tmp/evidence/run"
mkdir -p "$evidence"
for octree in 192 256; do
  cp "$tmp/cube.glb" "$evidence/comfy-$octree.glb"
done
comfy_manifest="$evidence/comfy-manifest.json"
jq -n --arg dir "$evidence" '
  {schema_version:"mold.hunyuan3d.comfy-metal-reference.v1", status:"passed",
   implementation:"ComfyUI", backend:"MPS", device:"mps",
   settings:{seed:25026,steps:5,cfg:1.0},
   runs:[{octree_resolution:192, glb_path:($dir + "/comfy-192.glb"), wall_seconds:11},
         {octree_resolution:256, glb_path:($dir + "/comfy-256.glb"), wall_seconds:12}]}' \
  > "$comfy_manifest"

before_marker="$tmp/before-marker"
: > "$before_marker"
# The runner is invoked from the repository root because it reads HEAD there.
# Snapshot the tree so a stray file dropped into the checkout is caught too.
repo_listing_before="$tmp/repo-listing-before.txt"
find "$repo_root" -maxdepth 2 -not -path '*/.git/*' -not -path '*/target/*' \
  -not -path '*/node_modules/*' | sort > "$repo_listing_before"
sleep 1

report="$evidence/report.json"
image_log="$evidence/fake-mold-images.txt"
(
  cd "$repo_root"
  HUNYUAN3D_CONTRACT_TEST=1 \
  MOLD_HOME="$tmp/home" \
  MOLD_BIN="$tmp/bin/fake-mold" \
  HUNYUAN3D_SOURCE_IMAGE="$tmp/source.png" \
  HUNYUAN3D_EVIDENCE_DIR="$evidence" \
  HUNYUAN3D_OCTREES="192 256 320" \
  HUNYUAN3D_TIME_CMD="$tmp/bin/fake-time" \
  HUNYUAN3D_COMFY_MANIFEST="$comfy_manifest" \
  HUNYUAN3D_COMPARE_PYTHON="$tmp/bin/fake-python" \
  HUNYUAN3D_FAKE_IMAGE_LOG="$image_log" \
  HUNYUAN3D_HOST_JSON='{"os":"fixture","arch":"arm64"}' \
    "$uat_runner" > "$tmp/uat.stdout" 2> "$tmp/uat.stderr"
) || {
  cat "$tmp/uat.stderr" >&2
  fail "runner exited non-zero"
}

[[ "$(tail -n 1 "$tmp/uat.stdout")" == "$report" ]] \
  || fail "runner did not print its report path"
[[ -s "$report" ]] || fail "missing report: $report"

# ---------------------------------------------------------------------------
# Evidence layout.
# ---------------------------------------------------------------------------
for octree in 192 256 320; do
  for suffix in glb stdout stderr summary.txt; do
    [[ -s "$evidence/mold-$octree.$suffix" || -f "$evidence/mold-$octree.$suffix" ]] \
      || fail "missing evidence file: mold-$octree.$suffix"
  done
done
[[ -s "$evidence/mold-256-seed25027.glb" ]] || fail "missing noise-floor mesh"
[[ -s "$evidence/source-framed.png" ]] || fail "missing framed source image"
[[ -f "$evidence/source-framing.log" ]] || fail "missing framing log"

# Every mold invocation must have been handed the FRAMED picture, not the raw
# cutout. That is the whole point of the framing step.
while IFS= read -r used; do
  [[ "$used" == "$evidence/source-framed.png" ]] \
    || fail "mold was handed $used instead of the framed source"
done < "$image_log"
[[ "$(wc -l < "$image_log" | tr -d ' ')" == 4 ]] \
  || fail "expected four mold invocations, got $(wc -l < "$image_log")"
for octree in 192 256; do
  [[ -s "$evidence/compare-$octree.json" ]] || fail "missing compare report at $octree"
  [[ -s "$evidence/compare-$octree.png" ]] || fail "missing compare PNG at $octree"
done
[[ ! -e "$evidence/compare-320.json" ]] \
  || fail "compared a rung the reference manifest does not carry"

# Nothing may be written outside the evidence directory. The fake home and the
# fake bin directory are the harness's own, not the runner's.
strays="$(find "$tmp" -newer "$before_marker" -type f \
  -not -path "$evidence/*" -not -path "$tmp/bin/*" -not -name 'uat.std*' \
  -not -name 'before-marker' -not -name 'repo-listing-*' | sort)"
[[ -z "$strays" ]] || fail "runner wrote outside the evidence directory:"$'\n'"$strays"

repo_listing_after="$tmp/repo-listing-after.txt"
find "$repo_root" -maxdepth 2 -not -path '*/.git/*' -not -path '*/target/*' \
  -not -path '*/node_modules/*' | sort > "$repo_listing_after"
repo_strays="$(comm -13 "$repo_listing_before" "$repo_listing_after")"
[[ -z "$repo_strays" ]] || fail "runner wrote into the checkout:"$'\n'"$repo_strays"

# ---------------------------------------------------------------------------
# Report shape.
# ---------------------------------------------------------------------------
jq -e '
  .schema_version == "mold.hunyuan3d.metal-uat.v1"
  and .backend_scope == "metal"
  and .model == "hunyuan3d-mini-turbo"
  and (.source_commit | test("^[0-9a-f]{40}$"))
  and (.source_tree_state | IN("clean", "dirty"))
  and .settings.format == "glb"
  and .settings.seed == 25026
  and .settings.noise_floor_seed == 25027
  and .settings.noise_floor_octree == 256
  and (.source_image.sha256 | test("^[0-9a-f]{64}$"))
  and .source_image.framed == true
  and (.source_image.framed_sha256 | test("^[0-9a-f]{64}$"))
  and (.source_image.framed_sha256 != .source_image.sha256)
  and (.source_image.framed_path | endswith("/source-framed.png"))
  and (.source_image.framing_script | endswith("/hunyuan3d-frame-source.py"))
  and (.runs | length) == 4
  and ([.runs[].label] == ["mold-192","mold-256","mold-320","mold-256-seed25027"])
  and ([.runs[].max_rss_bytes] | all(. == 1234567890))
  and ([.runs[].glb_sha256] | all(test("^[0-9a-f]{64}$")))
  and ([.runs[].wall_seconds] | all(type == "number"))
  and ([.runs[].units_wide] | all(. == 1.98))
  and ([.runs[].textured] | all(. == false))
  and ((.runs[] | select(.label == "mold-192")).vertex_count) == 1920
  and ((.runs[] | select(.label == "mold-192")).triangle_count) == 3840
  and ((.runs[] | select(.label == "mold-320")).vertex_count) == 3200
  and ((.runs[] | select(.label == "mold-256-seed25027")).seed) == 25027
  and .comfy_reference.status == "compared"
  and (.comparisons | length) == 3
  and ([.comparisons[] | select(.status == "compared") | .octree_resolution] == [192, 256])
  and ((.comparisons[] | select(.octree_resolution == 320)).status) == "no_reference_at_rung"
  and ([.comparisons[] | select(.status == "compared") | .report.pass] | all(. == true))
  and .comparison_coverage == {requested: 3, compared: 2}
  and .pass == false
  and .preservation.downloaded_models_deleted == false
  and .preservation.rendered_media_deleted == false
' "$report" >/dev/null || fail "report contract mismatch: $report"

# ---------------------------------------------------------------------------
# Full coverage is what passes. The run above compared two of three rungs and
# every comparison it DID make passed, so a coverage-blind verdict would call
# the family qualified on two rungs nothing ever measured.
# ---------------------------------------------------------------------------
covered_evidence="$tmp/evidence/covered"
mkdir -p "$covered_evidence"
(
  cd "$repo_root"
  HUNYUAN3D_CONTRACT_TEST=1 \
  MOLD_HOME="$tmp/home" MOLD_BIN="$tmp/bin/fake-mold" \
  HUNYUAN3D_SOURCE_IMAGE="$tmp/source.png" \
  HUNYUAN3D_EVIDENCE_DIR="$covered_evidence" \
  HUNYUAN3D_OCTREES="192 256" \
  HUNYUAN3D_TIME_CMD="$tmp/bin/fake-time" \
  HUNYUAN3D_COMFY_MANIFEST="$comfy_manifest" \
  HUNYUAN3D_COMPARE_PYTHON="$tmp/bin/fake-python" \
  HUNYUAN3D_HOST_JSON='{}' \
    "$uat_runner"
) >/dev/null 2>&1 || fail "runner exited non-zero on the fully covered run"
jq -e '.comparison_coverage == {requested: 2, compared: 2} and .pass == true' \
  "$covered_evidence/report.json" >/dev/null \
  || fail "a fully compared run must pass"

# ---------------------------------------------------------------------------
# A run whose CLI printed no mesh summary is a failure, not a silent null.
# ---------------------------------------------------------------------------
no_summary_evidence="$tmp/evidence/no-summary"
mkdir -p "$no_summary_evidence"
if (
  cd "$repo_root"
  HUNYUAN3D_CONTRACT_TEST=1 HUNYUAN3D_FAKE_NO_SUMMARY=1 \
  MOLD_HOME="$tmp/home" MOLD_BIN="$tmp/bin/fake-mold" \
  HUNYUAN3D_SOURCE_IMAGE="$tmp/source.png" \
  HUNYUAN3D_EVIDENCE_DIR="$no_summary_evidence" \
  HUNYUAN3D_OCTREES="256" HUNYUAN3D_TIME_CMD="$tmp/bin/fake-time" \
  HUNYUAN3D_SKIP_COMPARE=1 HUNYUAN3D_HOST_JSON='{}' \
    "$uat_runner"
) >/dev/null 2>&1; then
  fail "runner accepted a run with no mesh save summary"
fi

# ---------------------------------------------------------------------------
# HUNYUAN3D_FRAME_SOURCE=0 bypasses framing: no framed file, and mold is handed
# the raw cutout. A deliberate raw-cutout capture has to stay possible.
# ---------------------------------------------------------------------------
raw_evidence="$tmp/evidence/raw"
raw_image_log="$raw_evidence/fake-mold-images.txt"
mkdir -p "$raw_evidence"
(
  cd "$repo_root"
  HUNYUAN3D_CONTRACT_TEST=1 HUNYUAN3D_FRAME_SOURCE=0 \
  MOLD_HOME="$tmp/home" MOLD_BIN="$tmp/bin/fake-mold" \
  HUNYUAN3D_SOURCE_IMAGE="$tmp/source.png" \
  HUNYUAN3D_EVIDENCE_DIR="$raw_evidence" \
  HUNYUAN3D_OCTREES="256" HUNYUAN3D_TIME_CMD="$tmp/bin/fake-time" \
  HUNYUAN3D_SKIP_COMPARE=1 HUNYUAN3D_HOST_JSON='{}' \
  HUNYUAN3D_FAKE_IMAGE_LOG="$raw_image_log" \
    "$uat_runner"
) >/dev/null 2>&1 || fail "runner rejected a deliberate raw-cutout capture"
[[ ! -e "$raw_evidence/source-framed.png" ]] \
  || fail "framing ran despite HUNYUAN3D_FRAME_SOURCE=0"
jq -e '.source_image.framed == false
  and .source_image.framed_path == null
  and .source_image.framed_sha256 == null
  and .source_image.framing_script == null' "$raw_evidence/report.json" >/dev/null \
  || fail "bypassed run still reports a framed source"
while IFS= read -r used; do
  [[ "$used" == "$tmp/source.png" ]] \
    || fail "bypassed run handed mold $used instead of the raw cutout"
done < "$raw_image_log"

# ---------------------------------------------------------------------------
# The real helper scripts, when an interpreter carrying their dependencies is
# available. CI hosts without them skip rather than fail. The two probes differ:
# framing needs numpy and Pillow, comparison also needs scipy.
# ---------------------------------------------------------------------------
# No personal paths: an explicit override first, then the retained ComfyUI
# virtualenv only when MOLD_HOME actually names a home, then whatever python3
# is on PATH. A CI host with none of them skips rather than fails.
find_python() {
  local imports="$1" candidate
  local -a candidates=()
  [[ -n "${HUNYUAN3D_COMPARE_PYTHON:-}" ]] && candidates+=("$HUNYUAN3D_COMPARE_PYTHON")
  [[ -n "${MOLD_HOME:-}" ]] && candidates+=("$MOLD_HOME/comfyui-venv/bin/python")
  candidates+=("$(command -v python3 || true)")
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if "$candidate" -c "import $imports" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 0
}

frame_python="$(find_python 'numpy, PIL')"
if [[ -n "$frame_python" ]]; then
  "$frame_python" "$frame_script" --self-test >/dev/null \
    || fail "hunyuan3d-frame-source self-test failed under $frame_python"
  echo "source framing self-test OK under $frame_python"
else
  echo "skipping source framing self-test: no interpreter with numpy and Pillow;" \
    "set HUNYUAN3D_COMPARE_PYTHON or MOLD_HOME to run it"
fi

compare_python="$(find_python 'numpy, scipy.spatial, PIL')"
if [[ -n "$compare_python" ]]; then
  "$compare_python" "$compare_script" --self-test >/dev/null \
    || fail "hunyuan3d-mesh-compare self-test failed under $compare_python"
  echo "mesh compare self-test OK under $compare_python"
else
  echo "skipping mesh compare self-test: no interpreter with numpy, scipy and Pillow;" \
    "set HUNYUAN3D_COMPARE_PYTHON or MOLD_HOME to run it"
fi

echo "Hunyuan3D Metal UAT contract OK"
