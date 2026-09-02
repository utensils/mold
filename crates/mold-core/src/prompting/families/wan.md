# Wan prompting

Manifest family: `wan`.

## Prompt style

Write one English shot description of about {{word_limit}} words. Infer missing
detail for a short input without changing its intent. Enhance what the user
named: appearance, expression, quantity, posture, visual style, spatial
relationships, shot scale. Never add a subject the input lacks. Detail how the
action unfolds and give background elements their own motion. Use simple direct
verbs. Emphasize motion and camera movement. Skip literary mood writing. Name a
style only when the user did, and put it first; a 2D style takes no cinematic
terms.

## Syntax

Wan has no weighting syntax. The server prefills the model's tuned default
negative prompt, a standard Chinese quality list; never restate its terms in the
positive. Keep quoted text verbatim.

Add at most four cinematic settings, drawn from time, light, tone, composition,
shot size, and camera angle. Shot size: Extreme close-up shot, Close-up shot, Medium close-up shot, Medium
shot, Medium wide shot, Wide shot, Extreme wide shot; default Medium shot or
Wide shot. Camera angle: Over-the-shoulder shot, Low angle shot, High angle
shot, Dutch angle shot, Aerial shot, Overhead shot; skip it when the request
already gives a camera move.

## Generation context

Wan runs at 16 fps on a 4k+1 frame grid, so 49 frames is about three seconds and
81 about five. Fit the action to that: two or three beats, one shot, one camera
move. A longer request renders as chained clips, and the seam carries motion
only on an image-conditioned checkpoint.

## Examples

Input: a fox in the snow

Output: Medium wide shot, day time, side lighting. A red fox trots through fresh
snow in a pine forest, breath drifting back, powder lifting from each paw. The
camera tracks alongside.

Input: a paper boat in a rain gutter

Output: Close-up shot, overcast lighting. A folded paper boat drifts down a
gutter, spins once against a leaf, then straightens and speeds up as the camera
pushes forward.

## Pitfalls

Wan generates silent video. Never request dialogue or sound, or claim
synchronized audio. Wan S2V is a separate speech-to-video model. Wan 2.2 A14B
drives two experts from this one prompt, so keep it internally consistent. Read
one task leaf: text-to-video for T2V identities, image-conditioned for I2V and
TI2V identities or any source frame.

## CLI

```bash
# Wan 2.1 text-to-video (frames are 4k+1: 49, 81, 121, ...; MP4 default)
mold run wan21-t2v-1.3b "a red fox trotting through snow" --frames 81 --fps 16
# 3-step DMD distill of the same 1.3B (no CFG; steps/solver/shift are pinned)
mold run wan21-t2v-1.3b:turbo "a red fox trotting through snow" --frames 81 --fps 16
# Wan 2.1 14B, the dense 2.1 quality tier (a bare name resolves :q8)
mold run wan21-t2v-14b "a red fox trotting through snow"
# Wan 2.2 A14B, 4-step Lightning tier (two experts, one resident at a time)
mold run wan22-t2v-a14b:q5 "a paper boat drifting down a rain gutter"
mold run wan22-i2v-a14b:q5 "the balloon lifts off" --image balloon.png
# Low-VRAM tier: Q4_K_M A14B keeps the same Lightning recipe
mold run wan22-t2v-a14b:q4 "a paper boat drifting down a rain gutter"
# fp8-scaled A14B quality tier (20-step recipe, lower peak VRAM than :q8)
mold run wan22-t2v-a14b:fp8 "storm waves crash over the lighthouse"
# Wan 2.2 5B at 720p24
mold run wan22-ti2v-5b "waves on a black sand beach" --width 1280 --height 704 --frames 121 --fps 24
# Q8_0 5B reaches smaller cards
mold run wan22-ti2v-5b:q8 "waves on a black sand beach" --width 1280 --height 704
# Sequences: past the per-clip envelope this auto-chains and stitches one MP4
# delivering exactly the requested total (keep --frames on the 4k+1 grid).
# The seam continues only on an image-conditioned checkpoint; clips are 4k+1.
mold run wan22-ti2v-5b:q8 "a paper boat drifting down a rain gutter" --frames 97 --clip-frames 49
# Single-frame text-to-image: --frames 1 renders a still (png default, jpeg allowed)
mold run wan22-t2v-a14b:q5 "a lighthouse at dusk, volumetric fog" --frames 1 -o still.png
# Recipe controls: flow shift, sample solver, per-expert distill strength
mold run wan22-t2v-a14b:q8 "storm waves" --sample-shift 12
mold run wan22-t2v-a14b:q5 "storm waves" --sample-solver euler
mold run wan22-t2v-a14b:q5 "storm waves" --distill-strength high=1.8,low=1.0 --steps 6
# First/last-frame interpolation (A14B I2V or TI2V-5B; endpoints only)
mold run wan22-i2v-a14b:q5 "the sapling grows into an oak" --image sapling.png --last-image oak.png
# Send an explicit empty negative prompt, disabling the tuned model default
mold run wan22-t2v-a14b:q5 "a cat" --no-negative
# Animated WebP output
mold run wan22-ti2v-5b:q8 "waves on a black sand beach" --frames 49 --format webp -o waves.webp
```

## Sources

- https://github.com/Wan-Video/Wan2.2
- https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/system_prompt.py
- https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/prompt_extend.py
- https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/shared_config.py
