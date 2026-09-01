# Wuerstchen v2 prompting

Manifest family: `wuerstchen`.

## Prompt style

Name a broad subject, its dominant colours, the lighting, and the medium, in
one or two sentences under {{word_limit}} words. Both cascade stages read a
CLIP text encoder with a 77-token window, so extra clauses are truncated rather
than honoured. Ask for mood and palette. Do not ask for small, countable, or
finely structured detail.

## Syntax

No weighting syntax. The negative prompt is supported and applied by the prior
stage, which runs guidance above 1; mold ships a defect-suppression default.
Use it for blur, muddy colour, and watermarks. There is no on-image text worth
quoting and no reference image to address.

## Generation context

1024x1024 is the default canvas and resolution moves in 128-pixel steps, so the
next size up is 1152x1152. Choose an aspect ratio that suits one large subject
filling the frame.

## Examples

Input: a lighthouse at sunset

Output: A lighthouse on a rocky coast during a dramatic sunset, bold oil
painting, vibrant orange and purple sky, crashing surf.

Input: a forest in autumn

Output: A dense autumn forest in low golden light, deep amber and rust foliage,
painterly landscape illustration with soft atmospheric haze.

## Pitfalls

- The 42x latent compression costs fine detail, and it shows first in faces and
  hands.
- The model cannot render correct text in an image.
- Output is often not photorealistic, so painterly and illustrative requests
  suit it best.
- Difficult compositional prompts, meaning several named objects in stated
  positions, are unreliable.

## CLI

```bash
# Painterly subject with a real negative prompt
mold run wuerstchen-v2:fp16 \
  "A lighthouse on a rocky coast during a dramatic sunset, bold oil painting, vibrant orange and purple sky, crashing surf" \
  --negative-prompt "fine text, watermark, muddy colors" --seed 42

# Next canvas up the 128-pixel ladder
mold run wuerstchen-v2:fp16 "A dense autumn forest in low golden light, painterly landscape illustration" --width 1152 --height 1152 --seed 43
```

## Sources

- https://arxiv.org/abs/2306.00637
- https://huggingface.co/docs/diffusers/main/en/api/pipelines/wuerstchen
- https://huggingface.co/warp-ai/wuerstchen
