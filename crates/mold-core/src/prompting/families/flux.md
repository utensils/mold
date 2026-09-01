# FLUX.1 prompting

Manifest family: `flux`.

## Prompt style

Applies to Schnell, Dev, Krea, and FLUX.1 fine-tunes. Write the prompt as a
clear description of the finished image. FLUX reads natural sentences, so
prefer prose to tag piles. BFL publishes one starting structure, "a useful
starting structure, not a strict formula": subject, location, style, camera
settings, lighting, colors, effect, additional elements. Put the subject and
its action first, then composition, lighting, lens or material cues, and style
last. English gives the most precise results. Keep the expanded prompt under
{{word_limit}} words. Schnell is a four-step draft model and wants a single
idea. Dev, Krea, and quality fine-tunes reward exact materials, spatial
relationships, and light.

## Syntax

mold parses no weighting syntax. `(word:1.3)` and `[word]` reach the encoder as
literal characters, so lead with the important term or repeat it instead.
FLUX.1 Dev is guidance distilled and has no negative branch, so a negative
prompt does nothing in ordinary use. Describe what you want, not what you want
removed. The one exception is true CFG, which requires an identity reference:
`--true-cfg` above 1.0 paired with `--guidance 1.0` runs a real negative
branch, and it is FLUX only. Put visible lettering in quotation marks and name
its placement, typeface feel, and colour. With a source image, prompt the
change and name what must stay unchanged. With an identity reference the
reference owns the face: describe role, clothing, setting, pose, composition,
and light, and never re-describe facial features. Start near `--id-weight 0.8`.

## Generation context

The default canvas is 1024x1024. Match composition to the aspect: a wide
establishing frame for landscape, a tight head-and-shoulders crop for portrait.
Dev follows guidance 3.5 over 50 steps upstream and wants a full brief. Schnell
is fixed at four steps and guidance 0, so cut back to one subject and one
setting. In img2img a higher strength moves further from the source, so carry
more of the target description as strength rises. When inpainting, describe
only what belongs inside the mask and how it meets the surrounding pixels.

## Examples

Input: my face as an astronaut botanist, identity reference attached

Output: Cinematic medium close-up of an orbital botanist inside a glass greenhouse above Earth, cream flight jacket, tending luminous blue orchids, sunrise through curved windows, natural skin texture, 50mm documentary photograph.

Input: bakery sign at night

Output: A rain-slick corner bakery at dusk, the window sign reading "OPEN" in warm pink neon, reflections pooling on wet pavement, shallow depth of field, 35mm night photograph.

## Pitfalls

Tag piles and two competing styles weaken prompt following. A negative prompt
on Dev is silently inert without true CFG. A very high identity weight makes
skin look waxy. Raise it only when the face drifts. Identity cannot be combined
with a LoRA or with img2img. Schnell loaded with detail returns its average.

## CLI

```bash
# Text to image; Dev rewards a full brief
mold run flux-dev:q4 \
  "A cozy Japanese tea house interior, two ceramic cups steaming on a low cedar table, warm paper lanterns, rain beyond the open shoji, intimate eye-level composition, delicate watercolor texture" \
  --seed 1337

# Four-step draft
mold run flux-schnell:q8 "a red fox asleep on a mossy log, soft morning light" --steps 4

# img2img: prompt the change, raise --strength for a bigger move
mold run flux-dev:q4 "oil painting style, visible brushwork" --image photo.png --strength 0.6

# Inpaint: white in the mask is repainted, black is preserved
mold run flux-dev:q4 "a golden retriever sitting on the grass" --image park.png --mask mask.png

# LoRA adapters, stacked; one --lora-scale applies to the stack
mold run flux-dev:bf16 "epic mountain shot at golden hour" \
  --lora cinematic.safetensors --lora dramatic-lighting.safetensors --lora-scale 0.8

# PuLID identity: one-time licence-gated pull, then reference the face
mold pull pulid-flux --accept-license insightface-antelopev2
mold run flux-dev:q4 "an astronaut in a roadside diner" --id-image face.jpg --id-weight 0.85 --id-start-step 2

# Several views of the same person are averaged, up to four
mold run flux-dev:q4 "a chef plating in a copper kitchen" \
  --id-image front.jpg --id-image side.jpg --id-image smiling.jpg

# The real negative branch: FLUX only, and only with an active identity
mold run flux-dev:q4 "a hiker on a ridge at sunrise" --id-image face.jpg \
  --true-cfg 2.0 --guidance 1.0 --negative-prompt "blurry, cartoon, waxy skin"
```

## Sources

- https://docs.bfl.ai/guides/prompting_summary
- https://docs.bfl.ai/guides/prompting_unified_basics
- https://huggingface.co/black-forest-labs/FLUX.1-dev
- https://huggingface.co/black-forest-labs/FLUX.1-schnell
- https://github.com/ToTheBeginning/PuLID
