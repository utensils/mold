# Qwen-Image prompting

Manifest family: `qwen-image`.

## Prompt style

Rewrite the request into a complete, expressive prompt that preserves its
original meaning. For an overly brief input, reasonably infer and add detail so
the frame is visually complete, without altering the core content. Refine four
things in order: subject characteristics, visual style, spatial relationships,
and shot composition. Match the prompt to a precise, niche style aligned with
the stated intent; when no style is given, choose the most appropriate one,
usually realistic photography. Keep the rewrite under {{word_limit}} words,
then append `Ultra HD, 4K, cinematic composition` (a Chinese prompt appends
`超清，4K，电影级构图`).

## Syntax

Enclose on-image text in quotation marks and name its position, such as the
top-left corner, along with its style. Quoted text is never translated and
never altered. Do not add lettering the user did not ask for. Avoid negation in
the positive prompt: describe what should be present rather than what should be
absent. The negative prompt is honoured whenever guidance stays above 1. No
weighting syntax, and no reference-image addressing on this family.

## Generation context

1328x1328 is the default canvas; match the composition to the requested aspect
ratio. The base and 2512 checkpoints at fifty steps are the quality reference
and carry a dense art-direction brief. The few-step distills need a simpler
prompt and run their own fixed step-and-guidance recipe, described in their
model leaf.

## Examples

Input: a poster for a small bakery called MOLD & FLOUR

Output: Straight-on editorial architectural photograph of a tiny artisan bakery
on a quiet European corner, deep teal facade with three arched windows and a
striped awning. The hand-painted sign above the door reads "MOLD & FLOUR" in
cream serif capitals. A vintage delivery bicycle leans at the right edge. Sunny
spring morning, crisp realistic detail, balanced symmetrical composition. Ultra
HD, 4K, cinematic composition.

Input: a lion statue

Output: Dynamic lion stone sculpture mid-pounce, front legs airborne and hind
legs pushing off, smooth lines and defined muscles showing power. Faded ancient
courtyard background with trees and stone steps. Weathered surface gives an
antique look. Documentary photography style with fine details. Ultra HD, 4K,
cinematic composition.

## Pitfalls

- A vague text request such as "a sign with the date" renders as garbled
  lettering. Write the exact string the image should carry.
- Negation words leak into the frame; rephrase them as what should be there.
- Dense small text, hair-fine texture, and very complex scenes degrade on the
  few-step distills.

## CLI

```bash
# Quality reference: the 2512 recipe with quoted signage
mold run qwen-image-2512:q8 \
  'Straight-on editorial architectural photograph of a tiny artisan bakery named "MOLD & FLOUR" on a quiet European corner, deep teal facade, three arched windows, striped awning, sunny spring morning, crisp realistic detail, balanced symmetrical composition' \
  --seed 251201

# Quantize the Qwen2.5 text encoder when VRAM is tight
mold run qwen-image:q2 "a travel poster of a mountain lake at dawn" --qwen2-variant q6 --seed 251202

# Four-step distill: one simple idea at its fixed recipe
mold run qwen-image-flash:q8 "A red enamel teapot on a sunlit windowsill, still-life photograph" --steps 4 --seed 251203
```

## Sources

- https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/src/examples/tools/prompt_utils.py
- https://github.com/QwenLM/Qwen-Image
- https://huggingface.co/Qwen/Qwen-Image
