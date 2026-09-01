# Qwen-Image Edit prompting

Manifest family: `qwen-image-edit`.

## Prompt style

Write one imperative edit instruction, direct and specific, under
{{word_limit}} words. Keep the core intention of the request unchanged and add
only clarity, rationality, and visual feasibility. For an add, delete, or
replace, supplement a vague request with minimal but sufficient detail:
category, colour, size, orientation, position. Phrase a swap as "Replace Y with
X" and briefly describe X's key visual features. Every added object must match
the logic and style of the source image. Name what must stay identical, such as
geometry, camera position, composition, and any subject the edit does not
touch.

## Syntax

On-image text goes in double quotes, keeping its original language and
capitalization. Adding text and changing text are both replacements, so write
`Replace "OPEN" with "CLOSED"`. State text position, colour, and layout only
when the user asked for them. Colorization and old-photo restoration use the
fixed instruction `Restore and colorize the photo.` Inpainting uses `Perform
inpainting on this image. The original caption is:` and outpainting uses
`Extend the image beyond its boundaries using outpainting. The original caption
is:`. Address multiple references by ordinal in the order they were supplied,
as "the first image" and "the second image". This family has no img2img
strength control.

## Generation context

The source image owns appearance, geometry, and canvas, so prompt only the
change. A person edit keeps ethnicity, gender, age, hairstyle, expression, and
outfit consistent, and any expression, beauty, or make-up change must be
natural and subtle, never exaggerated. When a style is transferred from a
second reference, describe that style in key visual features and place the
style clause last.

## Examples

Input: make the bakery photo wintry

Output: Change the sunny spring morning into a snowy winter blue hour. Add
fresh snow to the exterior and turn on warm interior lights. Preserve the exact
teal facade, the readable "MOLD & FLOUR" sign, windows, door, camera position,
geometry, and composition.

Input: put the person from the second photo into the first

Output: Replace the woman in the first image with the man from the second
image, matching the first image's lighting and camera angle. Keep the first
image's background, furniture, and composition unchanged.

## Pitfalls

- Describing the whole scene instead of the change makes the model repaint what
  you described. State the edit, then the invariants.
- Omitting what must stay identical is the usual cause of drifted geometry.
- Translating quoted text or changing its case rewrites the sign.
- Exaggerated face or make-up edits break identity.

## CLI

```bash
# Single-image edit that names the preserved elements
mold run qwen-image-edit-2511:q8 \
  "Change the sunny spring morning into a cozy snowy winter blue hour. Add fresh snow to the exterior and turn on warm interior lights. Preserve the exact teal facade, readable MOLD & FLOUR sign, windows, door, camera position, geometry, and composition." \
  --image bakery.png --seed 251111

# Two references, addressed in the supplied order
mold run qwen-image-edit-2511:q4 "make the chair red leather" --image chair.png --image swatch.png --qwen2-variant q4

# Four-step Lightning distill: keep the instruction simple
mold run qwen-image-edit-lightning:fp8 'Replace the sign text with "CLOSED", keeping the font, colour, and placement.' --image shop.png --seed 251112
```

## Sources

- https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/src/examples/tools/prompt_utils.py
- https://github.com/QwenLM/Qwen-Image
- https://huggingface.co/Qwen/Qwen-Image-Edit
