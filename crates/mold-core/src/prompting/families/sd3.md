# SD 3.5 prompting

Manifest family: `sd3`.

## Prompt style

Write complete natural-language sentences, not a comma-separated tag list. The
MMDiT backbone reads two CLIP encoders plus a T5 encoder and rewards grammar
that the CLIP-only families ignore. Lead with the subject and its action, then
state count, placement, and spatial relationship explicitly, then the setting
and lighting, and finish with the medium, lens, or style. Say "three lanterns"
and "in the left third" rather than leaving quantity and position implicit.
Front-load the clause that matters most: the CLIP encoders see only the first
77 tokens, so a detail buried at the end reaches T5 alone. Keep the finished
prompt under {{word_limit}} words.

## Syntax

No weighting syntax; emphasis is expressed in words. The negative prompt is
supported whenever guidance stays above 1. Fill it with visible defects to
suppress, such as warped anatomy, illegible signage, or a watermark, and never
restate the positive prompt in it. Typography is a strength of this family, so
put required on-image lettering in double quotes and name where it sits. There
is no reference-image addressing to write.

## Generation context

1024x1024 is the native canvas. Match the composition to the requested aspect
ratio: a portrait canvas wants a vertical subject with headroom, a wide canvas
wants a horizontal relationship between subject and setting. Around 28 steps
at a guidance of 4 to 7 is the quality reference for the undistilled
checkpoints. A four-step turbo checkpoint has its own recipe in its model leaf.

## Examples

Input: victorian clocktower at sunset

Output: A Victorian clocktower fills the left third of a city square at sunset.
Its glass walls reveal interlocking brass gears while pedestrians cross below.
Low-angle architectural photograph, dramatic clouds.

Input: a bookshop sign that says Fable and Ink

Output: A narrow corner bookshop at dusk, its hand-painted sign reading "Fable
& Ink" centred above the door. Two customers browse a table of secondhand books
on the pavement. Warm window light against a blue evening street, 35mm
documentary photograph.

## Pitfalls

- The CLIP window truncates at 77 tokens, so the opening sentence has to carry
  the composition on its own.
- Counting is dependable for a handful of objects and drifts for a crowd
  (community observation).
- A checkpoint running at guidance 1.0 ignores the negative prompt entirely.
- Naming the subject in the negative prompt removes it from the image.

## CLI

```bash
# Quality reference: SD3.5 Large with a real negative prompt
mold run sd3.5-large:q8 \
  "A Victorian clocktower fills the left third of a city square at sunset; its glass walls reveal interlocking brass gears while pedestrians cross below, low-angle architectural photograph, dramatic clouds" \
  --negative-prompt "illegible clock, warped buildings, text, watermark" --seed 2024

# Smaller medium checkpoint at the same canvas
mold run sd3.5-medium:q8 "A red paper crane on a windowsill, soft morning light, still-life photograph" --seed 2025

# Four-step distill: one idea, no negative prompt
mold run sd3.5-large-turbo:q8 "A lone sailboat on a mirror-flat lake at dawn, wide landscape photograph" --steps 4 --seed 2026
```

## Sources

- https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo
- https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_3
- Community: crowd-count drift is user-reported and not documented by Stability AI.
