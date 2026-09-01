#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin" "$tmp/run" "$tmp/state"

cat > "$tmp/bin/magick" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" > "${MOLD_FAKE_STATE:?}/magick-args-$(basename "${@: -1}")"
printf '%s\n' "$*" > "${MOLD_FAKE_STATE:?}/magick-args"
out="${@: -1}"
printf 'fake image\n' > "$out"
FAKE
chmod +x "$tmp/bin/magick"

cat > "$tmp/bin/curl" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
url="${@: -1}"
case "$url" in
  */api/models)
    cat <<'JSON'
[
  {
    "name": "sdxl-source:q8",
    "family": "sdxl",
    "downloaded": true,
    "default_steps": 1,
    "default_width": 1024,
    "default_height": 1024
  },
  {
    "name": "hunyuan3d-mini-turbo:fp16",
    "family": "hunyuan3d",
    "downloaded": true,
    "default_steps": 5,
    "default_width": 1022,
    "default_height": 1022,
    "source_image": "required"
  }
]
JSON
    ;;
  */api/catalog/installed?kind=lora)
    cat <<'JSON'
{"entries":[]}
JSON
    ;;
  *)
    echo "unexpected curl URL: $url" >&2
    exit 2
    ;;
esac
FAKE
chmod +x "$tmp/bin/curl"

cat > "$tmp/bin/fake-mold" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
output=""
while (($# > 0)); do
  if [[ "$1" == "--output" ]]; then
    shift
    output="$1"
  fi
  shift || true
done
printf 'fake output\n' > "$output"
FAKE
chmod +x "$tmp/bin/fake-mold"

PATH="$tmp/bin:$PATH" \
MOLD_BIN="$tmp/bin/fake-mold" \
MOLD_FAKE_STATE="$tmp/state" \
MOLD_REGRESSION_OUT="$tmp/run" \
MOLD_REGRESSION_RUN_ID=source-image \
MOLD_REGRESSION_TIMEOUT_IMAGE=10 \
"$repo_root/scripts/regression-matrix.sh" > "$tmp/stdout" 2> "$tmp/stderr"

# The harness now synthesizes a second still for wan first/last-frame
# cases, so assert against the source.png invocation specifically rather
# than whichever magick call happened to run last.
args="$(<"$tmp/state/magick-args-source.png")"
if [[ "$args" != *"ellipse 512,560"* || "$args" != *"arc 300,430 455,665"* || "$args" != *"path 'M 690,520"* ]]; then
  echo "expected regression source image to draw a teapot body, handle, and spout" >&2
  echo "$args" >&2
  exit 1
fi

echo "regression matrix source image is teapot-shaped"

# A mesh family has no output canvas: the manifest's width/height are the
# conditioning size the source image is letterboxed to, not a raster the render
# produces. The arm must therefore queue exactly one image-to-mesh case and
# must not hand the CLI a canvas it would have to ignore.
log="$tmp/run/source-image/results.jsonl"
mapfile -t mesh_cmds < <(jq -r 'select(.status == "start" and .family == "hunyuan3d") | .cmd' "$log")
if (( ${#mesh_cmds[@]} != 1 )); then
  echo "expected exactly one queued hunyuan3d case, got ${#mesh_cmds[@]}" >&2
  printf '%s\n' "${mesh_cmds[@]}" >&2
  exit 1
fi
mesh_cmd="${mesh_cmds[0]}"
for required in "--format glb" "--image" ".source.glb" "--steps 5"; do
  if [[ "$mesh_cmd" != *"$required"* ]]; then
    echo "hunyuan3d case is missing $required" >&2
    echo "$mesh_cmd" >&2
    exit 1
  fi
done
for forbidden in "--width" "--height" "--frames" "--fps"; do
  if [[ "$mesh_cmd" == *"$forbidden"* ]]; then
    echo "hunyuan3d case must not pass $forbidden: a mesh has no canvas or timeline" >&2
    echo "$mesh_cmd" >&2
    exit 1
  fi
done
mesh_output="$(jq -r 'select(.status == "ok" and .family == "hunyuan3d") | .output' "$log")"
if [[ "$mesh_output" != *"hunyuan3d.hunyuan3d-mini-turbo"*".source.glb" ]]; then
  echo "unexpected hunyuan3d output path: $mesh_output" >&2
  exit 1
fi

echo "regression matrix queues one canvasless hunyuan3d mesh case"
