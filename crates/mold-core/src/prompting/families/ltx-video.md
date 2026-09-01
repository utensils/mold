# LTX-Video 0.9.x prompting

Manifest family: `ltx-video`.

## Prompt style

Write one flowing paragraph, chronological, literal and precise, the way a
cinematographer describes a shot list. Start directly with the main action in a
single sentence. Then add the specific movements and gestures, precise
appearances of the characters and objects, the background and environment, the
camera angle and movement, lighting and colors, and last any change or sudden
event. Keep within {{word_limit}} words. Prefer visible action over abstract
mood: concrete observable detail steers this model, adjectives about feeling do
not.

## Syntax

No weighting or attention syntax. A guidance-based checkpoint takes a negative
prompt for visible defects; a distilled checkpoint is pinned at guidance 1.0 and
takes none, so do not write one for it. Visible lettering is unreliable, so keep
on-image text out of the prompt. A source image or video is attached rather than
named in the prompt, and a conditioned run may omit the prompt entirely, which
renders near-static micro-motion. Camera vocabulary is
plain English: the camera slowly tilts upward, tracks alongside, pushes in,
pulls back, or holds locked off. Use one move per clip.

## Generation context

Frames follow the 8n+1 grid and both dimensions are multiples of 32. The default
canvas is 1216x704 at 30 fps with 25 frames, which is under a second, so one
beat is all that fits; 49 frames is about 1.6 seconds and 97 about 3.2. A longer
request is rendered as chained 97-frame clips, so write each clip as one
continuous shot that opens on the previous clip's closing pose.

## Examples

Input: northern lights over a frozen lake

Output: Northern lights ripple from left to right over a frozen lake; green and
violet ribbons reflect in the ice while the camera slowly tilts upward, one
continuous time-lapse shot.

Input: a chef plating a dish

Output: A chef's hands lower a seared scallop onto a white plate with tongs,
then trail a spoon of green oil around the rim. Steam rises from the plate.
Stainless counters and hanging pans fill the blurred background under warm
overhead light. The camera holds a locked-off overhead close-up.

## Pitfalls

Legacy LTX-Video renders silent video, so never promise speech, music, or sound
effects. It is not LTX-2, and a prompt written for that audio-video family does
not transfer. Avoid cuts, a second shot, and more action than a short clip can
hold. The 0.9.6 distilled checkpoint is the safest default, the 0.9.8
checkpoints run the full multiscale refinement path, and the 13B BF16 tiers need
a 40 GB-class GPU.

## CLI

```bash
# Basic clip (25 frames, MP4 default in a build carrying the mp4 feature)
mold run ltx-video-0.9.6-distilled:bf16 "a cat walking across a windowsill" --frames 25
# Frame counts are 8n+1 (9, 17, 25, 33, 49, ...)
mold run ltx-video-0.9.8-2b-distilled:bf16 "ocean waves at sunset" --frames 49
# Explicit MP4 output
mold run ltx-video-0.9.6-distilled:bf16 "a campfire at night" --frames 17 --format mp4
# GIF (256 colors)
mold run ltx-video-0.9.6-distilled:bf16 "a sunset" --frames 17 --format gif -o sunset.gif
# Animated WebP output (needs the webp feature)
mold run ltx-video-0.9.6-distilled:bf16 "a waterfall" --frames 9 --format webp -o waterfall.webp
# A finished chronological prompt
mold run ltx-video-0.9.6-distilled:bf16 \
  "Northern lights ripple from left to right over a frozen lake; green and violet ribbons reflect in the ice while the camera slowly tilts upward, one continuous time-lapse shot" \
  --frames 33 --seed 1234
```

## Sources

- https://github.com/Lightricks/LTX-Video (README prompt engineering section)
- https://huggingface.co/Lightricks/LTX-Video
- Community: https://github.com/Lightricks/ComfyUI-LTXVideo
