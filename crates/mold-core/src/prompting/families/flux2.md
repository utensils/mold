# FLUX.2 prompting

Manifest family: `flux2`.

## Prompt style

Applies to Klein 4B and 9B, Klein Base, Dev, and compatible fine-tunes. BFL's
formula is Subject + Action + Style + Context: the main focus, what it is doing
or its pose, the artistic approach or medium, then the setting, lighting, time,
and mood. "Word order matters - FLUX.2 pays more attention to what comes
first", so order the prompt main subject, key action, critical style, essential
context, secondary details. Length tiers: 10 to 30 words for quick concepts, 30
to 80 words "usually ideal for most projects", 80 or more for complex scenes
needing detailed specifications. Keep the expanded prompt under {{word_limit}}
words. Distilled Klein renders in four steps and holds one visual idea best.

## Syntax

mold parses no weighting syntax, so `(word:1.2)` reaches the encoder as literal
characters. "FLUX.2 does not support negative prompts. Focus on describing what
you want, not what you don't want." Turn "no blur" into "sharp focus
throughout" and "no people" into "an empty scene". mold's undistilled
`flux2-klein-base` tiers are the one exception, because above guidance 1.0 they
run a real unconditional branch and honour a negative prompt. Put visible
lettering in quotation marks: the text "OPEN" appears in red neon letters. Name
its placement, typography, and colour. Tie a hex colour to a named object: a
cobalt jacket, color #1B4FA0. "Hex codes work best when clearly associated with
specific objects. Vague references like 'use #FF0000 somewhere' may produce
inconsistent results." With several references, "clearly describe the role of
each: subject from image 1, style from image 2, background from image 3." BFL
also accepts a JSON prompt for a complex scene, with the keys scene, subjects
(description, position, action), style, color_palette, lighting, mood,
background, composition, and camera (angle, lens, depth_of_field).

## Generation context

The default canvas is 1024x1024, so match the composition clause to the
requested aspect. Distilled Klein runs four steps at guidance 1.0, so spend the
words on one subject and one setting. Klein Base and Dev run 50 steps at
guidance 4.0 and reward placed elements and named materials. With a source
image, prompt the change and name what must stay.

## Examples

Input: ship in a bottle

Output: Macro photograph of a glass bottle ship inside a curling ocean wave, the bottle centered and fully visible, lightning behind translucent blue water, dramatic dark background.

Input: combine these two references into a product shot

Output: The ceramic kettle from image 1 in the warm studio lighting and muted palette of image 2, three-quarter view on pale linen, sharp focus throughout.

## Pitfalls

A negative prompt is inert on every distilled Klein and Dev tier, so rewrite an
exclusion as a positive description. Burying the subject behind style words
costs prompt following, because leading words weigh more. A hex code naming no
object drifts. A four-step Klein given an 80-word brief averages it.

## CLI

```bash
# Distilled Klein: one idea at four steps
mold run flux2-klein-9b:q4 \
  "Macro photograph of a glass bottle ship caught inside a curling ocean wave, the bottle centered and fully visible, lightning behind translucent blue water, crisp reflections, dramatic dark background" \
  --steps 4 --seed 999

# Klein Base is undistilled: 50 steps at guidance 4.0 and a real negative prompt
mold run flux2-klein-base:q8 "a brass orrery on a walnut desk, low winter sun" \
  --steps 50 --guidance 4.0 --negative-prompt "blurry, warped rings, text, watermark"

# Dev carries the densest brief
mold run flux2-dev:q6 "An empty art-deco cinema lobby at dawn, brass handrails, the marquee reading \"CLOSED\", color #C8A24B accents, wide symmetrical composition" \
  --steps 50 --guidance 4.0

# Image editing: name the change and what stays
mold run flux2-klein:q8 "repaint the front door in deep teal, leave the brickwork and planters unchanged" --image house.png
```

## Sources

- https://docs.bfl.ai/guides/prompting_guide_flux2
- https://docs.bfl.ai/guides/prompting_unified_basics
- https://huggingface.co/black-forest-labs/FLUX.2-dev
