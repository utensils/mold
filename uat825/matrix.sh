#!/usr/bin/env bash
# The complete #825 / #827 Ref2VA acceptance matrix, run unattended.
# Each case waits for the production server to go idle before it submits.
set -uo pipefail
root=/home/jamesbrink/Projects/utensils/mold/.claude/worktrees/agent-a81576867db752e34
fix=/storage-fast/mold/uat-825/fixtures
run="$root/uat825/run-case.sh"
summary=/storage-fast/mold/uat-825/logs/matrix-summary.txt
: >"$summary"

note() { echo "$@" | tee -a "$summary"; }

note "== a: image only =="
bash "$run" a-image-only "a slow cinematic dolly through a misty pine forest at dawn" \
  --reference "image=$fix/ref-image-sq.png" 2>&1 | tee -a "$summary"

# Case a is the MEASURED shape and keeps the reviewed 21-step count so it is
# directly comparable with #827/#1033. Cases b-e exist to exercise the
# conditioning paths and the order sensitivity, neither of which is a function
# of the step count, so they run the shortest schedule the base tier admits
# that still produces a real print.
export MOLD_UAT_STEPS=8

note "== b: video with soundtrack =="
bash "$run" b-video-audio "the same scene continues, camera drifting to the right" \
  --reference "video=$fix/ref-video-audio-short.mp4" 2>&1 | tee -a "$summary"

note "== c: image and standalone audio =="
bash "$run" c-image-audio "a quiet forest clearing, matched to the soundtrack" \
  --reference "image=$fix/ref-image-sq.png" \
  --reference "audio=$fix/ref-audio-short.wav" 2>&1 | tee -a "$summary"

note "== d: mixed ordered image, video, audio =="
bash "$run" d-mixed "the referenced subject moves through the referenced scene" \
  --reference "image=$fix/ref-image-sq.png" \
  --reference "video=$fix/ref-video-silent-short.mp4" \
  --reference "audio=$fix/ref-audio-short.wav" 2>&1 | tee -a "$summary"

note "== e: same set, first two references swapped =="
bash "$run" e-swapped "the referenced subject moves through the referenced scene" \
  --reference "video=$fix/ref-video-silent-short.mp4" \
  --reference "image=$fix/ref-image-sq.png" \
  --reference "audio=$fix/ref-audio-short.wav" 2>&1 | tee -a "$summary"

note "== f: count-limit refusal (10 images, cap is 9) =="
export MOLD_HOST=http://localhost:7681
export MOLD_API_KEY=uat825
set +e
"$root/target/release/mold" run minimax-h3-ref2va:comfy-pruned-int8 "ten references" \
  --width 1344 --height 768 --frames 124 --fps 24 \
  --steps 21 --guidance 0 --strength 1.0 --format mp4 \
  --output /storage-fast/mold/uat-825/output/f-count-limit.mp4 \
  --reference "image=$fix/many-1.png" --reference "image=$fix/many-2.png" \
  --reference "image=$fix/many-3.png" --reference "image=$fix/many-4.png" \
  --reference "image=$fix/many-5.png" --reference "image=$fix/many-6.png" \
  --reference "image=$fix/many-7.png" --reference "image=$fix/many-8.png" \
  --reference "image=$fix/many-9.png" --reference "image=$fix/many-10.png" \
  >/storage-fast/mold/uat-825/logs/f-count-limit.stdout \
  2>/storage-fast/mold/uat-825/logs/f-count-limit.stderr
note "case=f-count-limit rc=$?"
set -e
tail -3 /storage-fast/mold/uat-825/logs/f-count-limit.stderr | tee -a "$summary"
note "== matrix complete =="
