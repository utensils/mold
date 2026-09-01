# MiniMax H3 Ref2VA

## Prompt style

Ref2VA replaces the three core fields with six, in order:
`subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`. Establish
the style in one sentence before `[Shot 1]`. State which traits each reference
owns, and never let one reference silently replace another's identity, motion,
or sound role.

## Syntax

Label references `<Picture n>`, `<Video n>`, and `<Audio n>` in the order
supplied, counting each category independently. A reference video carrying a
soundtrack also takes the next `<Audio n>` label first, so strip unwanted
audio before upload. Retention markers are `fully_preserved`,
`partially_preserved`, `attribute_transfer`, and `weak_reference`; audio uses
`fully_copy`, `partially_copy`, `reference`, and `weak_reference`.

## Generation context

A five-second clip normally stays one continuous shot. Keep speech short
enough to finish before the final pose.

## CLI

References upload through authenticated endpoints, so `MOLD_API_KEY` must be
set. The supplied order is part of the render.

```bash
# Ordered reference set: images, then video, then audio
mold run minimax-h3-ref2va:comfy-pruned-int8 "$(cat h3-ref-prompt.txt)" --reference image=a.png --reference video=b.mp4 --reference audio=c.wav
```

## Sources

- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt
