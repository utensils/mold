#!/usr/bin/env bash
# #790: the wan arm of scripts/regression-matrix.sh, gated with no GPU and no
# weights. Asserts that the generated case set follows what each checkpoint
# ADVERTISES — its conditioning contract, its dimension alignment, and its
# tier's own step count — rather than a family constant.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin" "$tmp/run" "$tmp/state"

cat > "$tmp/bin/magick" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
out="${@: -1}"
printf 'fake image\n' > "$out"
FAKE
chmod +x "$tmp/bin/magick"

# Three wan tiers covering all three conditioning contracts and both grids:
# a T2V-only 4-step Lightning tier, a 20-step quality I2V-required tier, and
# TI2V-5B, which is `optional` and needs /32 rather than /16.
cat > "$tmp/bin/curl" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
url="${@: -1}"
case "$url" in
  */api/models)
    cat <<'JSON'
[
  {
    "name": "wan22-t2v-a14b:q5",
    "family": "wan",
    "downloaded": true,
    "default_steps": 4,
    "default_width": 848,
    "default_height": 480,
    "dimension_alignment": 16,
    "source_image": "unsupported"
  },
  {
    "name": "wan22-i2v-a14b:q8",
    "family": "wan",
    "downloaded": true,
    "default_steps": 20,
    "default_width": 832,
    "default_height": 480,
    "dimension_alignment": 16,
    "source_image": "required"
  },
  {
    "name": "wan22-ti2v-5b:fp16",
    "family": "wan",
    "downloaded": true,
    "default_steps": 20,
    "default_width": 1296,
    "default_height": 720,
    "dimension_alignment": 32,
    "source_image": "optional"
  }
]
JSON
    ;;
  */api/catalog/installed?kind=lora)
    printf '{"entries": []}\n'
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
state_dir="${MOLD_FAKE_STATE:?}"
output=""; width=""; height=""; steps=""; frames=""; last_image=""; image=""
while (($# > 0)); do
  case "$1" in
    --output) shift; output="$1" ;;
    --width) shift; width="$1" ;;
    --height) shift; height="$1" ;;
    --steps) shift; steps="$1" ;;
    --frames) shift; frames="$1" ;;
    --image) shift; image="$1" ;;
    --last-image) shift; last_image="$1" ;;
  esac
  shift || true
done
[[ -n "$output" ]] || { echo "fake mold got no --output" >&2; exit 2; }
stem="$(basename "$output")"
printf '%s|%s|%s|%s|%s|%s\n' "$width" "$height" "$steps" "$frames" "$image" "$last_image" \
  > "$state_dir/$stem"
printf 'fake output\n' > "$output"
FAKE
chmod +x "$tmp/bin/fake-mold"

PATH="$tmp/bin:$PATH" \
MOLD_BIN="$tmp/bin/fake-mold" \
MOLD_FAKE_STATE="$tmp/state" \
MOLD_REGRESSION_OUT="$tmp/run" \
MOLD_REGRESSION_RUN_ID=wan \
MOLD_REGRESSION_TIMEOUT_IMAGE=10 \
MOLD_REGRESSION_TIMEOUT_VIDEO=10 \
MOLD_REGRESSION_STEPS_VIDEO=99 \
"$repo_root/scripts/regression-matrix.sh" > "$tmp/stdout" 2> "$tmp/stderr" || {
  echo "regression-matrix.sh failed" >&2
  tail -n 40 "$tmp/stderr" >&2
  exit 1
}

field() { cut -d'|' -f"$2" < "$tmp/state/$1"; }
exists() { [[ -e "$tmp/state/$1" ]]; }

fail() {
  echo "$1" >&2
  echo "--- cases generated ---" >&2
  ls "$tmp/state" >&2
  exit 1
}

# A T2V-only checkpoint gets base + still, and NO source or first/last case:
# it would be refused at admission.
exists wan.wan22-t2v-a14b-q5.base.mp4 || fail "expected a base case for the T2V tier"
exists wan.wan22-t2v-a14b-q5.still.png || fail "expected a single-frame still case (#798)"
! exists wan.wan22-t2v-a14b-q5.source.mp4 || fail "T2V-only must not get a source case"
! exists wan.wan22-t2v-a14b-q5.flf.mp4 || fail "T2V-only must not get a first/last case"

# An I2V-required checkpoint gets source + first/last, and NO bare base or
# still: both would be rejected for lacking the image the model needs.
exists wan.wan22-i2v-a14b-q8.source.mp4 || fail "expected a source case for the I2V tier"
exists wan.wan22-i2v-a14b-q8.flf.mp4 || fail "expected a first/last case for the I2V tier (#779)"
! exists wan.wan22-i2v-a14b-q8.base.mp4 || fail "I2V-required must not get a bare base case"
! exists wan.wan22-i2v-a14b-q8.still.png || fail "I2V-required must not get a still case"

# `optional` gets the full spread.
for c in base.mp4 source.mp4 flf.mp4 still.png; do
  exists "wan.wan22-ti2v-5b-fp16.$c" || fail "expected TI2V case $c"
done

# The first/last case must anchor two DIFFERENT stills, or a broken
# last-frame pin renders correctly anyway and the case proves nothing.
flf_first="$(field wan.wan22-i2v-a14b-q8.flf.mp4 5)"
flf_last="$(field wan.wan22-i2v-a14b-q8.flf.mp4 6)"
[[ -n "$flf_first" && -n "$flf_last" ]] || fail "first/last case must pass both endpoints"
[[ "$flf_first" != "$flf_last" ]] || fail "first/last endpoints must be different images"

# Geometry comes from the checkpoint's ADVERTISED alignment. The fixture
# sizes are chosen so each grid produces a different answer, and the
# assertions pin the exact forwarded value — a modulo check would pass even
# if the script ignored dimension_alignment entirely.
#   848x480  is /16-clean but NOT /32: a wrong /32 floors width to 832.
#   1296x720 is /16-clean but NOT /32: a wrong /16 leaves it unfloored.
ti2v_geom="$(field wan.wan22-ti2v-5b-fp16.base.mp4 1)x$(field wan.wan22-ti2v-5b-fp16.base.mp4 2)"
[[ "$ti2v_geom" == "1280x704" ]] || fail "TI2V must floor 1296x720 to its /32 grid, saw $ti2v_geom"
a14b_geom="$(field wan.wan22-t2v-a14b-q5.base.mp4 1)x$(field wan.wan22-t2v-a14b-q5.base.mp4 2)"
[[ "$a14b_geom" == "848x480" ]] || fail "A14B must keep 848x480 on its /16 grid, saw $a14b_geom"

# Steps come from the tier, NOT from MOLD_REGRESSION_STEPS_VIDEO (set to 99
# above): flattening would collapse the 4-step Lightning and 20-step quality
# recipes into one case that exercises neither distill path.
fast_steps="$(field wan.wan22-t2v-a14b-q5.base.mp4 3)"
quality_steps="$(field wan.wan22-i2v-a14b-q8.source.mp4 3)"
[[ "$fast_steps" == "4" ]] || fail "Lightning tier must keep its 4 steps, saw $fast_steps"
[[ "$quality_steps" == "20" ]] || fail "quality tier must keep its 20 steps, saw $quality_steps"

# The still case is one frame; the video cases are not.
still_frames="$(field wan.wan22-t2v-a14b-q5.still.png 4)"
[[ "$still_frames" == "1" ]] || fail "still case must request exactly 1 frame, saw $still_frames"

echo "wan regression-matrix contract OK"
