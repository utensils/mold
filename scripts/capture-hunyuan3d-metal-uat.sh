#!/usr/bin/env bash
set -euo pipefail

# Apple Metal qualification run for the Hunyuan3D image-to-3D family.
#
# Renders one mesh per octree rung with mold's local Metal engine, measures
# wall time and peak RSS for each, parses the CLI's own save summary for the
# vertex/triangle counts, adds a second-seed run at the middle rung as a noise
# floor, and — when a ComfyUI Metal reference capture is present — compares
# every rung against it with `scripts/hunyuan3d-mesh-compare.py`.
#
# Everything lands under the evidence directory and nothing is ever deleted.
# Pair this with `scripts/capture-hunyuan3d-comfy-metal-reference.sh`, which
# renders the SAME checkpoint, image and seed through ComfyUI.
#
# The source image is PRE-FRAMED first, by `scripts/hunyuan3d-frame-source.py`,
# and the framed copy is what mold is given. mold applies Tencent's `recenter`
# letterbox to a raw cutout — the alpha bounding box rescaled to fill 85 % of a
# square — while ComfyUI's `CLIPVisionEncode` with `crop: center` just
# centre-crops the picture as handed over. Comparing the two without framing
# therefore measures the conditioning policy, not the port: the 2026-09-01
# Metal captures of the same armchair scored a normalised Chamfer of 0.030 with
# the raw cutout and 0.0103 once both sides were fed the same pre-framed
# picture. After framing, mold's own letterbox is (very nearly) the identity
# and ComfyUI's centre crop of a square is a no-op, so what is left to measure
# is the networks. `HUNYUAN3D_FRAME_SOURCE=0` bypasses it for a deliberate
# raw-cutout capture.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mold_home="${MOLD_HOME:-/Volumes/ExternalStorage/mold2}"
mold_bin="${MOLD_BIN:-./target/release/mold}"
model="${HUNYUAN3D_MODEL:-hunyuan3d-mini-turbo}"
# The family has no text encoder; the prompt is recorded as provenance only.
# The CLI still requires one until the contract makes it optional for mesh
# families, so the harness labels the run with it.
prompt="${HUNYUAN3D_PROMPT:-hunyuan3d metal uat}"
source_image="${HUNYUAN3D_SOURCE_IMAGE:-}"
frame_source="${HUNYUAN3D_FRAME_SOURCE:-1}"
frame_script="${HUNYUAN3D_FRAME_SCRIPT:-$repo_root/scripts/hunyuan3d-frame-source.py}"
octrees="${HUNYUAN3D_OCTREES:-192 256 320}"
seed="${HUNYUAN3D_SEED:-25026}"
floor_seed="${HUNYUAN3D_FLOOR_SEED:-25027}"
floor_octree="${HUNYUAN3D_FLOOR_OCTREE:-256}"
verification_root="${HUNYUAN3D_VERIFICATION_ROOT:-$mold_home/output/verification/hunyuan3d}"
timestamp="${HUNYUAN3D_CAPTURE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
evidence_dir="${HUNYUAN3D_EVIDENCE_DIR:-$verification_root/$timestamp}"
comfy_manifest="${HUNYUAN3D_COMFY_MANIFEST:-}"
compare_script="${HUNYUAN3D_COMPARE_SCRIPT:-$repo_root/scripts/hunyuan3d-mesh-compare.py}"
compare_python="${HUNYUAN3D_COMPARE_PYTHON:-$mold_home/comfyui-venv/bin/python}"
frame_python="${HUNYUAN3D_FRAME_PYTHON:-$compare_python}"
skip_compare="${HUNYUAN3D_SKIP_COMPARE:-0}"
contract_test="${HUNYUAN3D_CONTRACT_TEST:-0}"
report="${HUNYUAN3D_REPORT:-$evidence_dir/report.json}"

# `/usr/bin/time` cannot be shadowed by a PATH entry, so the whole timing
# command is a parameter: the contract test points it at a fake that prints a
# `maximum resident set size` line the parser has to accept.
IFS=' ' read -r -a time_cmd <<<"${HUNYUAN3D_TIME_CMD:-/usr/bin/time -l}"

fail() {
  echo "Hunyuan3D Metal UAT failed: $*" >&2
  exit 1
}

# Associative arrays hold the per-rung mesh paths the comparison step reuses,
# so macOS's ancient /bin/bash cannot run this even if it is invoked directly.
(( BASH_VERSINFO[0] >= 4 )) \
  || fail "bash 4 or newer is required, found ${BASH_VERSION:-unknown}"

file_sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

file_size() {
  stat -c '%s' "$1" 2>/dev/null || stat -f '%z' "$1"
}

for command in git jq shasum; do
  command -v "$command" >/dev/null 2>&1 || fail "missing command: $command"
done
[[ -n "$source_image" ]] || fail "HUNYUAN3D_SOURCE_IMAGE is required"
[[ -s "$source_image" ]] || fail "missing source image: $source_image"
command -v "$mold_bin" >/dev/null 2>&1 || [[ -x "$mold_bin" ]] \
  || fail "mold binary is not executable: $mold_bin"
if [[ "$contract_test" != 1 ]]; then
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] \
    || fail "runtime capture is restricted to Apple Silicon Metal"
fi
for octree in $octrees; do
  [[ "$octree" =~ ^[0-9]+$ ]] || fail "octree rungs must be integers, found: $octree"
done

# Run from the current directory on purpose: the source commit that produced
# these meshes is this worktree's HEAD, and the report has to name it.
source_commit="$(git rev-parse HEAD)"
source_tree_state=clean
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || source_tree_state=dirty

mkdir -p "$evidence_dir" "$(dirname "$report")"
image_sha256="$(file_sha256 "$source_image")"

# Pre-framing (see the header): mold and the ComfyUI reference must be handed
# the SAME picture. The framed copy is retained beside the original; neither is
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

# ANSI escapes only appear when mold thinks it is talking to a terminal, but
# stripping unconditionally keeps the parser honest either way.
strip_ansi() {
  sed -E $'s/\x1b\\[[0-9;]*[A-Za-z]//g' "$1"
}

runs='[]'
declare -A run_glb=()

record_run() {
  local label="$1" octree="$2" run_seed="$3"
  local glb="$evidence_dir/$label.glb"
  local stdout_path="$evidence_dir/$label.stdout"
  local stderr_path="$evidence_dir/$label.stderr"
  local clean_path="$evidence_dir/$label.summary.txt"
  local started ended wall_seconds max_rss summary verts tris units textured exit_code

  echo "RUN $label (octree $octree, seed $run_seed) -> $glb"
  started="$(date +%s)"
  set +e
  MOLD_HOME="$mold_home" "${time_cmd[@]}" \
    "$mold_bin" run "$model" "$prompt" \
      --image "$effective_source" --local --seed "$run_seed" --octree "$octree" \
      --format glb -o "$glb" \
    >"$stdout_path" 2>"$stderr_path"
  exit_code=$?
  set -e
  ended="$(date +%s)"
  wall_seconds=$((ended - started))
  [[ "$exit_code" -eq 0 ]] || fail "$label exited $exit_code; see $stderr_path"
  [[ -s "$glb" ]] || fail "$label produced no mesh: $glb"

  strip_ansi "$stderr_path" >"$clean_path"

  # macOS `/usr/bin/time -l` reports peak RSS in bytes on its own line.
  max_rss="$(awk '/maximum resident set size/ {print $1; exit}' "$clean_path")"
  [[ "$max_rss" =~ ^[0-9]+$ ]] || max_rss=""

  # `Saved: <file> (N verts, M tris, [textured, ]W units wide)` —
  # crates/mold-cli/src/commands/generate.rs, save_and_preview_mesh.
  summary="$(grep -E '\([0-9]+ verts, [0-9]+ tris,' "$clean_path" | grep -F 'Saved' | tail -n 1 || true)"
  [[ -n "$summary" ]] || fail "$label printed no mesh save summary; see $clean_path"
  verts="$(sed -E 's/.*\(([0-9]+) verts,.*/\1/' <<<"$summary")"
  tris="$(sed -E 's/.*, ([0-9]+) tris,.*/\1/' <<<"$summary")"
  units="$(sed -E 's/.*tris, (textured, )?(-?[0-9]+(\.[0-9]+)?) units wide.*/\2/' <<<"$summary")"
  # sed echoes the line back when nothing matched; only a bare number is a
  # measurement, anything else is recorded as absent rather than as garbage.
  [[ "$units" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] || units=""
  textured=false
  [[ "$summary" == *"tris, textured, "* ]] && textured=true
  [[ "$verts" =~ ^[0-9]+$ && "$tris" =~ ^[0-9]+$ ]] \
    || fail "$label save summary did not parse: $summary"

  run_glb["$label"]="$glb"
  runs="$(jq -c \
    --arg label "$label" --arg glb "$glb" --arg glb_sha "$(file_sha256 "$glb")" \
    --arg stdout_path "$stdout_path" --arg stderr_path "$stderr_path" \
    --arg summary_path "$clean_path" --arg summary "$summary" \
    --arg max_rss "$max_rss" --arg units "$units" \
    --argjson octree "$octree" --argjson run_seed "$run_seed" \
    --argjson wall_seconds "$wall_seconds" --argjson bytes "$(file_size "$glb")" \
    --argjson verts "$verts" --argjson tris "$tris" --argjson textured "$textured" \
    '. + [{label: $label, octree_resolution: $octree, seed: $run_seed,
      wall_seconds: $wall_seconds,
      max_rss_bytes: (if $max_rss == "" then null else ($max_rss | tonumber) end),
      vertex_count: $verts, triangle_count: $tris,
      units_wide: (if $units == "" then null else ($units | tonumber) end),
      textured: $textured,
      glb_path: $glb, glb_sha256: $glb_sha, glb_bytes: $bytes,
      summary_line: $summary, summary_path: $summary_path,
      stdout_path: $stdout_path, stderr_path: $stderr_path}]' <<<"$runs")"
}

for octree in $octrees; do
  record_run "mold-$octree" "$octree" "$seed"
done
# Noise floor: the same rung, one seed apart. Two mold meshes that differ only
# by seed bound how much difference is inherent to the sampler rather than to
# the implementation being compared.
record_run "mold-$floor_octree-seed$floor_seed" "$floor_octree" "$floor_seed"

if [[ -z "$comfy_manifest" && "$skip_compare" != 1 ]]; then
  shopt -s nullglob
  candidates=("$evidence_dir/comfy-manifest.json" "$verification_root"/*/comfy-manifest.json)
  shopt -u nullglob
  for candidate in "${candidates[@]}"; do
    if [[ -s "$candidate" && ( -z "$comfy_manifest" || "$candidate" -nt "$comfy_manifest" ) ]]; then
      comfy_manifest="$candidate"
    fi
  done
fi

comparisons='[]'
comparison_status=skipped_no_reference
if [[ "$skip_compare" == 1 ]]; then
  comparison_status=skipped_by_request
elif [[ -n "$comfy_manifest" && -s "$comfy_manifest" ]]; then
  jq -e '.schema_version == "mold.hunyuan3d.comfy-metal-reference.v1"' "$comfy_manifest" >/dev/null \
    || fail "unexpected ComfyUI reference manifest schema: $comfy_manifest"
  [[ -f "$compare_script" ]] || fail "missing mesh compare script: $compare_script"
  [[ -x "$compare_python" ]] || command -v "$compare_python" >/dev/null 2>&1 \
    || fail "missing compare interpreter: $compare_python"
  floor_a="${run_glb["mold-$floor_octree"]:-}"
  floor_b="${run_glb["mold-$floor_octree-seed$floor_seed"]:-}"
  comparison_status=compared
  for octree in $octrees; do
    comfy_glb="$(jq -r --argjson octree "$octree" \
      '[.runs[]? | select(.octree_resolution == $octree) | .glb_path] | first // ""' \
      "$comfy_manifest")"
    if [[ -z "$comfy_glb" || ! -s "$comfy_glb" ]]; then
      comparisons="$(jq -c --argjson octree "$octree" \
        '. + [{octree_resolution: $octree, status: "no_reference_at_rung"}]' <<<"$comparisons")"
      continue
    fi
    compare_out="$evidence_dir/compare-$octree.json"
    compare_png="$evidence_dir/compare-$octree.png"
    compare_log="$evidence_dir/compare-$octree.log"
    compare_args=(
      "$compare_script"
      --mold "${run_glb["mold-$octree"]}"
      --comfy "$comfy_glb"
      --out "$compare_out"
      --png "$compare_png"
    )
    if [[ -s "$floor_a" && -s "$floor_b" ]]; then
      compare_args+=(--floor-a "$floor_a" --floor-b "$floor_b")
    fi
    set +e
    "$compare_python" "${compare_args[@]}" >"$compare_log" 2>&1
    compare_code=$?
    set -e
    if [[ ! -s "$compare_out" ]]; then
      fail "mesh comparison at octree $octree produced no report; see $compare_log"
    fi
    comparisons="$(jq -c --argjson octree "$octree" --arg comfy_glb "$comfy_glb" \
      --arg report_path "$compare_out" --arg png "$compare_png" --arg log "$compare_log" \
      --argjson exit_code "$compare_code" --slurpfile compare "$compare_out" \
      '. + [{octree_resolution: $octree, status: "compared", comfy_glb: $comfy_glb,
        report_path: $report_path, png_path: $png, log_path: $log,
        exit_code: $exit_code, report: $compare[0]}]' <<<"$comparisons")"
  done
fi

host='{}'
if [[ -n "${HUNYUAN3D_HOST_JSON:-}" ]]; then
  host="$HUNYUAN3D_HOST_JSON"
elif command -v sw_vers >/dev/null 2>&1; then
  host="$(jq -nc --arg os "$(sw_vers -productVersion)" --arg arch "$(uname -m)" \
    '{os: "macOS", os_version: $os, arch: $arch}')"
fi

tmp_report="$report.tmp.$$"
jq -n \
  --arg captured_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg source_commit "$source_commit" --arg source_tree_state "$source_tree_state" \
  --arg mold_home "$mold_home" --arg mold_bin "$mold_bin" --arg model "$model" \
  --arg evidence_dir "$evidence_dir" --arg source_image "$source_image" \
  --arg image_sha256 "$image_sha256" --arg comfy_manifest "$comfy_manifest" \
  --arg framed_source "$framed_source" --arg framed_sha256 "$framed_sha256" \
  --arg frame_script "$frame_script" \
  --argjson framed "$([[ "$frame_source" == 1 ]] && echo true || echo false)" \
  --arg comparison_status "$comparison_status" --arg time_cmd "${time_cmd[*]}" \
  --argjson seed "$seed" --argjson floor_seed "$floor_seed" \
  --argjson floor_octree "$floor_octree" --argjson host "$host" \
  --argjson runs "$runs" --argjson comparisons "$comparisons" \
  '{schema_version: "mold.hunyuan3d.metal-uat.v1", captured_at: $captured_at,
    source_commit: $source_commit, source_tree_state: $source_tree_state,
    backend_scope: "metal", mold_home: $mold_home, mold_bin: $mold_bin,
    model: $model, evidence_dir: $evidence_dir, timing_command: $time_cmd,
    host: $host,
    source_image: {path: $source_image, sha256: $image_sha256, framed: $framed,
      framing_script: (if $framed then $frame_script else null end),
      framed_path: (if $framed_source == "" then null else $framed_source end),
      framed_sha256: (if $framed_sha256 == "" then null else $framed_sha256 end)},
    settings: {seed: $seed, noise_floor_seed: $floor_seed,
      noise_floor_octree: $floor_octree, format: "glb"},
    runs: $runs,
    comfy_reference: {manifest_path: (if $comfy_manifest == "" then null else $comfy_manifest end),
      status: $comparison_status},
    comparisons: $comparisons,
    comparison_coverage: {requested: ($comparisons | length),
      compared: ([$comparisons[] | select(.status == "compared")] | length)},
    # Coverage is part of the verdict. A reference manifest carrying one of the
    # three rungs used to report pass on that rung alone, which reads as "the
    # family is qualified on Metal" for two rungs nothing ever compared.
    pass: ((($comparisons | length) > 0)
      and ([$comparisons[] | select(.status == "compared")] | length)
        == ($comparisons | length)
      and ([$comparisons[] | select(.status == "compared")] | all(.report.pass == true))),
    preservation: {downloaded_models_deleted: false, rendered_media_deleted: false}}' \
  >"$tmp_report"
mv "$tmp_report" "$report"

echo "$report"
