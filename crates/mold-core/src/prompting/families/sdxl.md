# SDXL prompting

Manifest family: `sdxl`.

## Prompt style

Applies to SDXL Base, Turbo, Playground v2.5, Juggernaut XL, DreamShaper XL,
RealVis XL, Pony Diffusion V6, and CyberRealistic Pony. SDXL runs two text
encoders, CLIP ViT-L and OpenCLIP ViT-bigG, and mold sends the same prompt to
both, so it reads fuller sentences than SD1.5. Open with a concise subject and
scene sentence, then add photographic or illustrative treatment as short
clauses: lens, lighting, palette, medium. Keep the expanded prompt under
{{word_limit}} words. Each encoder still truncates at 75 content tokens, so a
very long prompt loses its tail. Turbo and other few-step tiers need one simple
idea. Base and quality fine-tunes carry richer composition. Fine-tunes publish
trigger words and tag prefixes: preserve them exactly and place them first.

## Syntax

mold parses no attention weighting, so `(word:1.3)` and `[word]` reach the
encoders as literal characters. Move a term earlier or repeat it instead. The
negative prompt works normally on every SDXL checkpoint because SDXL uses
ordinary classifier-free guidance, and there is no default, so supply visible
defects such as blur, jpeg artifacts, text, watermark, extra fingers. Never
pass true CFG on SDXL: it is a FLUX-only path and mold refuses it by name.
Quoted lettering renders more reliably than on SD1.5 but should stay short.
With an identity reference the reference owns the face, so describe role,
clothing, setting, pose, composition, and light without re-describing features.
Start near an identity weight of 0.8. The negative pass is conditioned on the
unconditional identity automatically.

## Generation context

The native canvas is 1024x1024 at 25 steps and guidance 7.5. mold ships no
SDXL refiner stage, so never write a prompt that assumes a refinement pass.
Render at native scale and upscale afterwards. Turbo is native at 512 and
guidance 0. Name the framing that fits the aspect. In img2img a higher strength
moves further from the source. When inpainting, describe only the masked
content and how it meets the surrounding pixels.

## Examples

Input: night market photo

Output: Vibrant Bangkok street-food market at night, steam rising from woks, neon reflected on wet pavement, bustling documentary photograph, 35mm, shallow depth of field.

Input: noir detective with my face, identity reference attached

Output: Film-noir detective in a rain-soaked 1940s train station, charcoal overcoat, single platform lamp, drifting steam, wet pavement reflections, black-and-white 35mm photograph.

## Pitfalls

Dropping a fine-tune's trigger words or tag prefix loses the look the
checkpoint was trained for. Piling quality words onto a modern fine-tune adds
nothing. A few-step tier given base-model steps and guidance degrades. Turbo is
the one SDXL checkpoint that identity conditioning does not accept.

## CLI

```bash
# Turbo: one simple idea at four steps
mold run sdxl-turbo:fp16 \
  "Vibrant Bangkok street-food market at night, steam rising from woks, neon reflected on wet pavement, bustling documentary photograph" \
  --steps 4 --seed 88

# Base and fine-tunes take a real negative prompt
mold run sdxl-base:fp16 "a landscape at golden hour, layered ridgelines, drifting mist" \
  --negative-prompt "low quality, jpeg artifacts, text, watermark"
mold run juggernaut-xl:fp16 "a studio portrait on grey seamless, softbox key light, 85mm" \
  --negative-prompt "blurry, distorted face, text, watermark" --seed 4242

# PuLID identity: one-time licence-gated pull, then reference the face
mold pull pulid-sdxl --accept-license insightface-antelopev2
mold run juggernaut-xl:fp16 \
  "film-noir detective in a rain-soaked 1940s train station, charcoal overcoat, single platform lamp, wet pavement reflections, black-and-white 35mm photograph" \
  --id-image portrait.png --id-weight 0.8 \
  --negative-prompt "cartoon, waxy skin, distorted face, text, watermark" --seed 83121

# LoRA, img2img, and inpainting
mold run sdxl-base:fp16 "a lighthouse in a storm" --lora style.safetensors --lora-scale 0.8
mold run sdxl-base:fp16 "anime style" --image photo.png --strength 0.7
mold run sdxl-base:fp16 "a red bicycle leaning on the wall" --image street.png --mask mask.png

# No refiner stage exists; upscale the native render instead
mold upscale portrait.png -o portrait_4x.png
```

## Sources

- https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- https://huggingface.co/stabilityai/sdxl-turbo
- https://huggingface.co/docs/diffusers/using-diffusers/sdxl
- https://github.com/ToTheBeginning/PuLID
- https://education.civitai.com/civitais-prompt-crafting-guide-part-1-basics/ (community consensus on trigger words and negative prompts)
