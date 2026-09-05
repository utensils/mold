# LTX-2 prompting

Manifest family: `ltx2`.

## Prompt style

Write one flowing paragraph under {{word_limit}} words, four to eight
sentences. Start directly with the action, never with "The scene opens" or "We
see". Add movement, appearance, environment, camera, lighting, and colour,
then any sudden change. Use present-progressive verbs and strict chronological
order joined by "as", "then", and "while". Describe only what is visible or
audible: physical cues, not emotion labels. Keep wording restrained and
colours plain, so "red dress" rather than "vibrant red dress".

## Syntax

No weighting, no negative prompt. When the style is known, prefix the
paragraph with `Style: <style>,`. LTX-2 generates audio, so integrate the
soundscape beside the actions rather than appending it. Quote speech exactly,
keep the user's words verbatim, and name voice, delivery, accent, and language
when it is not English. Camera vocabulary is plain film language: static
frame, pans, tilts, pushes in, pulls back, tracks, arcs, handheld. Never
invent camera motion or dialogue the user did not request, and never write
timestamps or cuts.

## Generation context

Frames sit on the `8k+1` grid at 24 fps, so 121 frames is about five seconds.
Both axes must be multiples of 32, and of 64 for lip dub. Size the action to
that duration, keep one continuous take per clip, and keep lip sync inside a
single clip. With a source image attached, describe only what changes from it;
restating the frame inaccurately makes the model cut. Distilled LTX-2.5 fixes
guidance at 1.0. Audio renders by default on MP4 output, one-shots and
sequences alike, so write the soundscape unless the request says silent.

## Examples

Input: a woman on a rainy street says the tagline

Output: Style: cinematic-realistic, a woman in a red raincoat stands beneath a
glass awning at night as rain taps the panels above her. She turns toward the
lens and says in a clear, delighted voice, "This was made locally with Mold,
including the sound." The camera remains static. Traffic hisses on the wet
street beneath her close voice.

Input: make this photo of a chef move (source image attached)

Output: The chef lifts the pan off the flame and tilts it as the vegetables
slide forward, then sets it down. Oil crackles and an extractor fan hums under
the sizzle. The camera remains static.

## Pitfalls

Readable on-screen text is unreliable, so avoid signage and logos. Avoid fast
twisting motion, crowded layered scenes, and conflicting light sources.

## CLI

```bash
# Fast joint audio-video on the distilled default
mold run ltx-2-19b-distilled:fp8 "rain on a neon taxi window" --frames 97 --format mp4

# LTX-2.5 distilled, duration head picks the clip length
mold run ltx-2.5-22b-distilled "a complete product reveal" --predict-duration --fps 24 --audio --format mp4

# A fully written LTX-2.5 prompt in one continuous clip
mold run ltx-2.5-22b-distilled:q6 "A woman in a red raincoat stands beneath a glass awning at night as rain taps the panels above her. She turns toward the lens and says in a clear, delighted voice, 'This was made locally with Mold, including the sound.' The camera remains static. Traffic hisses on the wet street beneath her close, clean voice." --width 768 --height 512 --frames 121 --clip-frames 121 --fps 24 --audio --seed 83007

# Audio-to-video: motion driven by a supplied track
mold run ltx-2-19b-distilled:fp8 "paper sculpture reacting to music" --audio-file cello.wav

# Keyframe interpolation between two stills
mold run ltx-2-19b-distilled:fp8 "a canyon flyover" --pipeline keyframe --frames 97 --keyframe 0:start.png --keyframe 96:end.png

# Camera-control preset
mold run ltx-2-19b-distilled:fp8 "lantern-lit cave entrance" --camera-control dolly-in

# Lip dub: re-voice a clip. Frames and fps come from the reference video; both axes must be multiples of 64
mold run ltx-2.3-22b-distilled:fp8 "she says: the harbour freezes every winter" --ic-lora-control lipdub --video speaker.mp4 --width 704 --height 448

# Guidance overrides on the two-stage pipeline
mold run ltx-2-19b-distilled:fp8 "handheld shot through a night market" --pipeline two-stage --stg-scale 0.6 --stg-blocks 20,29 --rescale-scale 0.9

# Text-to-audio: no video at all, WAV output
mold run ltx-2.3-22b-dev:fp8 "heavy rain on a tin roof, distant thunder" --pipeline t2a --frames 121 --fps 24 --output rain.wav
```

## Sources

- https://github.com/Lightricks/LTX-2 (README, "Prompting for LTX-2")
- LTX-2 official prompt-enhancer system prompts, `packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/` in the same repository
- https://ltx.io/blog/prompting-guide-for-ltx-2
- https://docs.ltx.io/open-source-model/usage-guides/prompting-guide
