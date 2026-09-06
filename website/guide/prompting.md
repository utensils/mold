---
layout: doc
---

<!-- GENERATED FILE. Do not edit by hand.
     Source: crates/mold-core/src/prompting/ (the prompting corpus).
     Regenerate: cargo run -p mold-ai-core --bin generate_prompting_guides
     Verified in CI with --check. -->

# Prompting Guides

Every guide on this page comes from one corpus in the Mold source tree. The same files are installed with the agent skill (`mold skill install`), published to MCP hosts as `mold://prompting/` resources, and injected into the prompt expander that powers `mold expand`, `mold remix`, `--expand`, and the Expand and Remix actions in the desktop, web, and iPhone apps. The expander receives every section except `CLI` and `Sources`, plus a generation-context block naming the exact model, canvas, frame count, fps, duration, and ordered references.

The expander budget is 700 words per route. Word limits below are the corpus defaults; `[expand.families.<family>] word_limit` in `config.toml` overrides them, and `style_notes` replaces the guide text entirely.

## Routes by model

| Model | Family | Guides read in order | Word limit | Excerpt words |
| --- | --- | --- | --- | --- |
| `flux-schnell` | `flux` | `shared.md`, `families/flux.md`, `models/flux-schnell.md` | 150 | 627 |
| `flux-dev` | `flux` | `shared.md`, `families/flux.md` | 150 | 544 |
| `flux-krea` | `flux` | `shared.md`, `families/flux.md` | 150 | 544 |
| `jibmix-flux` | `flux` | `shared.md`, `families/flux.md` | 150 | 544 |
| `ultrareal-v2` | `flux` | `shared.md`, `families/flux.md` | 150 | 544 |
| `ultrareal-v3` | `flux` | `shared.md`, `families/flux.md` | 150 | 544 |
| `ultrareal-v4` | `flux` | `shared.md`, `families/flux.md` | 150 | 544 |
| `iniverse-mix` | `flux` | `shared.md`, `families/flux.md` | 150 | 544 |
| `sd15` | `sd15` | `shared.md`, `families/sd15.md` | 50 | 471 |
| `dreamshaper-v8` | `sd15` | `shared.md`, `families/sd15.md` | 50 | 471 |
| `realistic-vision-v5` | `sd15` | `shared.md`, `families/sd15.md` | 50 | 471 |
| `sd3.5-large` | `sd3` | `shared.md`, `families/sd3.md` | 150 | 470 |
| `sd3.5-large-turbo` | `sd3` | `shared.md`, `families/sd3.md`, `models/sd3.5-large-turbo.md` | 150 | 555 |
| `sd3.5-medium` | `sd3` | `shared.md`, `families/sd3.md` | 150 | 470 |
| `sdxl-base` | `sdxl` | `shared.md`, `families/sdxl.md` | 60 | 509 |
| `dreamshaper-xl` | `sdxl` | `shared.md`, `families/sdxl.md` | 60 | 509 |
| `juggernaut-xl` | `sdxl` | `shared.md`, `families/sdxl.md` | 60 | 509 |
| `realvis-xl` | `sdxl` | `shared.md`, `families/sdxl.md` | 60 | 509 |
| `playground-v2.5` | `sdxl` | `shared.md`, `families/sdxl.md`, `models/playground-v2.5.md` | 60 | 587 |
| `pony-v6` | `sdxl` | `shared.md`, `families/sdxl.md`, `models/pony-v6.md` | 60 | 596 |
| `cyberrealistic-pony` | `sdxl` | `shared.md`, `families/sdxl.md`, `models/pony-v6.md` | 60 | 596 |
| `sdxl-turbo` | `sdxl` | `shared.md`, `families/sdxl.md`, `models/sdxl-turbo.md` | 60 | 596 |
| `z-image-turbo` | `z-image` | `shared.md`, `families/z-image.md` | 150 | 425 |
| `flux2-klein` | `flux2` | `shared.md`, `families/flux2.md` | 120 | 614 |
| `flux2-klein-9b` | `flux2` | `shared.md`, `families/flux2.md` | 120 | 614 |
| `flux2-dev` | `flux2` | `shared.md`, `families/flux2.md` | 120 | 614 |
| `flux2-klein-base` | `flux2` | `shared.md`, `families/flux2.md`, `models/flux2-klein-base.md` | 120 | 699 |
| `flux2-klein-base-9b` | `flux2` | `shared.md`, `families/flux2.md`, `models/flux2-klein-base.md` | 120 | 699 |
| `qwen-image` | `qwen-image` | `shared.md`, `families/qwen-image.md` | 180 | 484 |
| `qwen-image-2512` | `qwen-image` | `shared.md`, `families/qwen-image.md` | 180 | 484 |
| `qwen-image-lightning` | `qwen-image` | `shared.md`, `families/qwen-image.md`, `models/qwen-image-flash.md` | 180 | 567 |
| `qwen-image-flash` | `qwen-image` | `shared.md`, `families/qwen-image.md`, `models/qwen-image-flash.md` | 180 | 567 |
| `qwen-image-distill` | `qwen-image` | `shared.md`, `families/qwen-image.md`, `models/qwen-image-flash.md` | 180 | 567 |
| `qwen-image-edit-2511` | `qwen-image-edit` | `shared.md`, `families/qwen-image-edit.md` | 100 | 513 |
| `qwen-image-edit-lightning` | `qwen-image-edit` | `shared.md`, `families/qwen-image-edit.md`, `models/qwen-image-edit-lightning.md` | 100 | 594 |
| `wuerstchen-v2` | `wuerstchen` | `shared.md`, `families/wuerstchen.md` | 50 | 325 |
| `hunyuan3d-mini-turbo` | `hunyuan3d` | `shared.md`, `families/hunyuan3d.md` | 40 | 603 |
| `hunyuan3d-turbo` | `hunyuan3d` | `shared.md`, `families/hunyuan3d.md` | 40 | 603 |
| `hunyuan3d` | `hunyuan3d` | `shared.md`, `families/hunyuan3d.md` | 40 | 603 |
| `hunyuan3d-2.1` | `hunyuan3d` | `shared.md`, `families/hunyuan3d.md` | 40 | 603 |
| `ltx-video-0.9.6` | `ltx-video` | `shared.md`, `families/ltx-video.md` | 150 | 523 |
| `ltx-video-0.9.6-distilled` | `ltx-video` | `shared.md`, `families/ltx-video.md` | 150 | 523 |
| `ltx-video-0.9.8-2b-distilled` | `ltx-video` | `shared.md`, `families/ltx-video.md` | 150 | 523 |
| `ltx-video-0.9.8-13b-dev` | `ltx-video` | `shared.md`, `families/ltx-video.md` | 150 | 523 |
| `ltx-video-0.9.8-13b-distilled` | `ltx-video` | `shared.md`, `families/ltx-video.md` | 150 | 523 |
| `ltx-2-19b-dev` | `ltx2` | `shared.md`, `families/ltx2.md` | 200 | 478 |
| `ltx-2-19b-distilled` | `ltx2` | `shared.md`, `families/ltx2.md` | 200 | 478 |
| `ltx-2.3-22b-dev` | `ltx2` | `shared.md`, `families/ltx2.md` | 200 | 478 |
| `ltx-2.3-22b-distilled` | `ltx2` | `shared.md`, `families/ltx2.md` | 200 | 478 |
| `ltx-2.5-22b-dev` | `ltx2` | `shared.md`, `families/ltx2.md`, `models/ltx-2.5.md` | 200 | 559 |
| `ltx-2.5-22b-distilled` | `ltx2` | `shared.md`, `families/ltx2.md`, `models/ltx-2.5.md` | 200 | 559 |
| `wan21-t2v-1.3b` | `wan` | `shared.md`, `families/wan.md`, `wan/text-to-video.md` | 100 | 592 |
| `wan21-t2v-14b` | `wan` | `shared.md`, `families/wan.md`, `wan/text-to-video.md` | 100 | 592 |
| `wan22-ti2v-5b` | `wan` | `shared.md`, `families/wan.md`, `wan/image-conditioned.md`, `models/wan22-ti2v-5b.md` | 80 | 684 |
| `wan22-t2v-a14b` | `wan` | `shared.md`, `families/wan.md`, `wan/text-to-video.md` | 100 | 592 |
| `wan22-i2v-a14b` | `wan` | `shared.md`, `families/wan.md`, `wan/image-conditioned.md` | 80 | 587 |
| `minimax-h3-fl2va` | `minimax-h3` | `shared.md`, `families/minimax-h3.md`, `minimax-h3/base-modes.md` | 250 | 583 |
| `minimax-h3-ref2va` | `minimax-h3` | `shared.md`, `families/minimax-h3.md`, `minimax-h3/ref2va.md` | 300 | 567 |
| `real-esrgan-x4plus` | `upscaler` | `shared.md`, `families/upscaler.md` | 20 | 229 |
| `real-esrgan-x4plus-anime` | `upscaler` | `shared.md`, `families/upscaler.md` | 20 | 229 |
| `real-esrgan-anime-v3` | `upscaler` | `shared.md`, `families/upscaler.md` | 20 | 229 |
| `real-esrgan-x2plus` | `upscaler` | `shared.md`, `families/upscaler.md` | 20 | 229 |

## Shared practice

### Shared prompting practice

#### Prompt style

State the subject and action before style, lighting, lens, or material cues.
Keep one coherent visual or temporal idea. Preserve exact user wording unless
the user asks for expansion. When source media owns appearance or composition,
prompt only the requested change and name what must stay unchanged.

#### Pitfalls

Hold one seed while refining a prompt, then vary the seed for final art. Never
promise sound, speech, or motion the selected family does not generate.

#### CLI

Read this file for every generation or upscale request, then read exactly one
family base guide linked from `SKILL.md`. Read a task leaf only when that task
needs its own prompt grammar.

Confirm the selected identity with `mold info <model>` or the remote
`/api/models` row before choosing dimensions, frames, steps, guidance, or
conditioning. Installed catalog checkpoints can differ from manifest defaults.

```bash
# Confirm an identity and its advertised defaults before writing the prompt
mold info flux2-klein:q8
```

## Families

<!-- families/flux.md -->

Default word limit: 150. Also accepted on the wire as `flux.1`, `flux-1`.

### FLUX.1 prompting

Manifest family: `flux`.

#### Prompt style

Applies to Schnell, Dev, Krea, and FLUX.1 fine-tunes. Write the prompt as a
clear description of the finished image. FLUX reads natural sentences, so
prefer prose to tag piles. BFL publishes one starting structure, "a useful
starting structure, not a strict formula": subject, location, style, camera
settings, lighting, colors, effect, additional elements. Put the subject and
its action first, then composition, lighting, lens or material cues, and style
last. English gives the most precise results. Keep the expanded prompt under
150 words. Schnell is a four-step draft model and wants a single
idea. Dev, Krea, and quality fine-tunes reward exact materials, spatial
relationships, and light.

#### Syntax

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

#### Generation context

The default canvas is 1024x1024. Match composition to the aspect: a wide
establishing frame for landscape, a tight head-and-shoulders crop for portrait.
Dev follows guidance 3.5 over 50 steps upstream and wants a full brief. Schnell
is fixed at four steps and guidance 0, so cut back to one subject and one
setting. In img2img a higher strength moves further from the source, so carry
more of the target description as strength rises. When inpainting, describe
only what belongs inside the mask and how it meets the surrounding pixels.

#### Examples

Input: my face as an astronaut botanist, identity reference attached

Output: Cinematic medium close-up of an orbital botanist inside a glass greenhouse above Earth, cream flight jacket, tending luminous blue orchids, sunrise through curved windows, natural skin texture, 50mm documentary photograph.

Input: bakery sign at night

Output: A rain-slick corner bakery at dusk, the window sign reading "OPEN" in warm pink neon, reflections pooling on wet pavement, shallow depth of field, 35mm night photograph.

#### Pitfalls

Tag piles and two competing styles weaken prompt following. A negative prompt
on Dev is silently inert without true CFG. A very high identity weight makes
skin look waxy. Raise it only when the face drifts. Identity cannot be combined
with a LoRA or with img2img. Schnell loaded with detail returns its average.

#### CLI

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

#### Sources

- https://docs.bfl.ai/guides/prompting_summary
- https://docs.bfl.ai/guides/prompting_unified_basics
- https://huggingface.co/black-forest-labs/FLUX.1-dev
- https://huggingface.co/black-forest-labs/FLUX.1-schnell
- https://github.com/ToTheBeginning/PuLID

<!-- families/flux2.md -->

Default word limit: 120.

### FLUX.2 prompting

Manifest family: `flux2`.

#### Prompt style

Applies to Klein 4B and 9B, Klein Base, Dev, and compatible fine-tunes. BFL's
formula is Subject + Action + Style + Context: the main focus, what it is doing
or its pose, the artistic approach or medium, then the setting, lighting, time,
and mood. "Word order matters - FLUX.2 pays more attention to what comes
first", so order the prompt main subject, key action, critical style, essential
context, secondary details. Length tiers: 10 to 30 words for quick concepts, 30
to 80 words "usually ideal for most projects", 80 or more for complex scenes
needing detailed specifications. Keep the expanded prompt under 120
words. Distilled Klein renders in four steps and holds one visual idea best.

#### Syntax

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

#### Generation context

The default canvas is 1024x1024, so match the composition clause to the
requested aspect. Distilled Klein runs four steps at guidance 1.0, so spend the
words on one subject and one setting. Klein Base and Dev run 50 steps at
guidance 4.0 and reward placed elements and named materials. With a source
image, prompt the change and name what must stay. Every tier also edits from
references — a separate ordered input, at most four, where "image 1" is the
first `--reference`. Klein renders from a source image OR from references,
never both, so a reference edit has no strength or mask; Dev takes no source
image.

#### Examples

Input: ship in a bottle

Output: Macro photograph of a glass bottle ship inside a curling ocean wave, the bottle centered and fully visible, lightning behind translucent blue water, dramatic dark background.

Input: combine these two references into a product shot

Output: The ceramic kettle from image 1 in the warm studio lighting and muted palette of image 2, three-quarter view on pale linen, sharp focus throughout.

#### Pitfalls

A negative prompt is inert on every distilled Klein and Dev tier, so rewrite an
exclusion as a positive description. Burying the subject behind style words
costs prompt following, because leading words weigh more. A hex code naming no
object drifts. A four-step Klein given an 80-word brief averages it.
An edit prompt that names the change but never what stays invites a redraw.

#### CLI

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

# Image editing from a source image: name the change and what stays
mold run flux2-klein:q8 "repaint the front door in deep teal, leave the brickwork and planters unchanged" --image house.png

# Single-reference editing: --reference sends the ordered group
mold run flux2-klein:bf16 "put sunglasses on the person, keep the pose and background" \
  --reference person.jpg --steps 4 --seed 42

# Multi-reference: name the role of each image in prompt order
mold run flux2-klein-9b:q8 "the woman from image 1 wearing the eyeglasses from image 2, same pose and lighting" \
  --reference person.jpg --reference glasses.jpg

# Klein Base takes references with a real CFG branch
mold run flux2-klein-base:q8 "place the kettle from image 1 on the linen table from image 2, soft window light" \
  --reference kettle.png --reference table.png --steps 50 --guidance 4.0

# Dev reads its ordered references from repeated --image
mold run flux2-dev:q6 "the jacket from image 1 on the model from image 2, studio backdrop" \
  --image jacket.png --image model.png
```

#### Sources

- https://docs.bfl.ai/guides/prompting_guide_flux2
- https://docs.bfl.ai/guides/prompting_unified_basics
- https://huggingface.co/black-forest-labs/FLUX.2-dev
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
- https://github.com/black-forest-labs/flux2 (README: every tier does single- and multi-reference editing)

<!-- families/sd15.md -->

Default word limit: 50. Also accepted on the wire as `sd1.5`, `stable-diffusion-1.5`.

### SD 1.5 prompting

Manifest family: `sd15`.

#### Prompt style

Applies to the base model and SD1.5 fine-tunes such as DreamShaper and
Realistic Vision. Community practice on AUTOMATIC1111 and Civitai is compact
comma-separated tags rather than sentences, and SD1.5 follows tags far better
than prose. Front-load the subject: the earliest tags carry the most weight.
Then medium and style, then setting, then lighting, then quality words. Keep
the expanded prompt under 50 words. The CLIP text encoder holds 75
content tokens and mold truncates past that with no chunking, so anything after
roughly 60 words never reaches the model. Fine-tunes publish trigger words:
preserve them exactly and place them first.

#### Syntax

mold parses no attention weighting. `(word:1.3)`, `(word)`, and `[word]` are an
AUTOMATIC1111 and ComfyUI interface convention. mold sends those characters to
CLIP as text. To emphasise a term, move it earlier or repeat it. The negative
prompt is load-bearing on SD1.5 and there is no default, so always supply one.
Fill it with visible defects such as blurry, low detail, extra fingers, bad
anatomy, text, watermark, and never restate the positive prompt inside it.
On-image lettering is unreliable at this scale, so avoid quoted text. With
ControlNet the control image owns pose, edges, or depth, so the prompt
describes only the rendered subject, its materials, and the light.

#### Generation context

The native canvas is 512x512 at 25 steps and guidance 7.5. Above roughly 768
pixels SD1.5 duplicates subjects and limbs, so keep one subject and a simple
composition when the canvas grows, and upscale afterwards instead of rendering
large. Name the framing that fits the aspect, such as a wide establishing shot
or a head-and-shoulders portrait. In img2img a higher strength moves further
from the source. When inpainting, tag only what belongs inside the mask.

#### Examples

Input: castle on floating islands

Output: fantasy castle on floating islands, waterfalls, sunrise, wide establishing shot, detailed concept art, luminous atmosphere

Input: portrait of a woman, photo

Output: portrait photograph of a woman, freckles, soft window light, shallow depth of field, 85mm lens, film grain, natural skin texture

#### Pitfalls

A long prose paragraph is truncated before its ending is read. Weighting
parentheses are inert and waste tokens. An empty negative prompt is the most
common cause of muddy SD1.5 output. Hands and small faces degrade, so frame
them large or leave them out. Rendering far above 512 duplicates subjects.

#### CLI

```bash
# Tags plus a real negative prompt
mold run dreamshaper-v8:fp16 \
  "fantasy castle on floating islands, waterfalls, sunrise, wide establishing shot, detailed concept art" \
  --negative-prompt "blurry, low detail, text, watermark, malformed architecture" --seed 555

# Base model at its native scale
mold run sd15:fp16 "a portrait of an old fisherman, weathered face, harbour light" \
  -n "blurry, watermark, ugly, bad anatomy" --width 512 --height 512

# ControlNet: the control image owns the pose or geometry
mold run sd15:fp16 "a person in a red raincoat" --control edges.png --control-model controlnet-canny-sd15:fp16
mold run sd15:fp16 "a sunlit loft interior" --control depth.png --control-model controlnet-depth-sd15:fp16 --control-scale 0.8

# img2img and inpainting
mold run sd15:fp16 "watercolor painting" --image photo.png --strength 0.6
mold run sd15:fp16 "a golden retriever on the grass" --image park.png --mask mask.png

# Render at native scale, then upscale
mold upscale portrait.png
```

#### Sources

- https://huggingface.co/runwayml/stable-diffusion-v1-5
- https://huggingface.co/docs/diffusers/using-diffusers/weighted_prompts
- https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features (community convention for weighting syntax)
- https://education.civitai.com/civitais-prompt-crafting-guide-part-1-basics/ (community consensus on tag order and negative prompts)
- https://github.com/lllyasviel/ControlNet

<!-- families/sdxl.md -->

Default word limit: 60.

### SDXL prompting

Manifest family: `sdxl`.

#### Prompt style

Applies to SDXL Base, Turbo, Playground v2.5, Juggernaut XL, DreamShaper XL,
RealVis XL, Pony Diffusion V6, and CyberRealistic Pony. SDXL runs two text
encoders, CLIP ViT-L and OpenCLIP ViT-bigG, and mold sends the same prompt to
both, so it reads fuller sentences than SD1.5. Open with a concise subject and
scene sentence, then add photographic or illustrative treatment as short
clauses: lens, lighting, palette, medium. Keep the expanded prompt under
60 words. Each encoder still truncates at 75 content tokens, so a
very long prompt loses its tail. Turbo and other few-step tiers need one simple
idea. Base and quality fine-tunes carry richer composition. Fine-tunes publish
trigger words and tag prefixes: preserve them exactly and place them first.

#### Syntax

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

#### Generation context

The native canvas is 1024x1024 at 25 steps and guidance 7.5. mold ships no
SDXL refiner stage, so never write a prompt that assumes a refinement pass.
Render at native scale and upscale afterwards. Turbo is native at 512 and
guidance 0. Name the framing that fits the aspect. In img2img a higher strength
moves further from the source. When inpainting, describe only the masked
content and how it meets the surrounding pixels.

#### Examples

Input: night market photo

Output: Vibrant Bangkok street-food market at night, steam rising from woks, neon reflected on wet pavement, bustling documentary photograph, 35mm, shallow depth of field.

Input: noir detective with my face, identity reference attached

Output: Film-noir detective in a rain-soaked 1940s train station, charcoal overcoat, single platform lamp, drifting steam, wet pavement reflections, black-and-white 35mm photograph.

#### Pitfalls

Dropping a fine-tune's trigger words or tag prefix loses the look the
checkpoint was trained for. Piling quality words onto a modern fine-tune adds
nothing. A few-step tier given base-model steps and guidance degrades. Turbo is
the one SDXL checkpoint that identity conditioning does not accept.

#### CLI

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

#### Sources

- https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- https://huggingface.co/stabilityai/sdxl-turbo
- https://huggingface.co/docs/diffusers/using-diffusers/sdxl
- https://github.com/ToTheBeginning/PuLID
- https://education.civitai.com/civitais-prompt-crafting-guide-part-1-basics/ (community consensus on trigger words and negative prompts)

<!-- families/sd3.md -->

Default word limit: 150. Also accepted on the wire as `sd3.5`, `sd35`.

### SD 3.5 prompting

Manifest family: `sd3`.

#### Prompt style

Write complete natural-language sentences, not a comma-separated tag list. The
MMDiT backbone reads two CLIP encoders plus a T5 encoder and rewards grammar
that the CLIP-only families ignore. Lead with the subject and its action, then
state count, placement, and spatial relationship explicitly, then the setting
and lighting, and finish with the medium, lens, or style. Say "three lanterns"
and "in the left third" rather than leaving quantity and position implicit.
Front-load the clause that matters most: the CLIP encoders see only the first
77 tokens, so a detail buried at the end reaches T5 alone. Keep the finished
prompt under 150 words.

#### Syntax

No weighting syntax; emphasis is expressed in words. The negative prompt is
supported whenever guidance stays above 1. Fill it with visible defects to
suppress, such as warped anatomy, illegible signage, or a watermark, and never
restate the positive prompt in it. Typography is a strength of this family, so
put required on-image lettering in double quotes and name where it sits. There
is no reference-image addressing to write.

#### Generation context

1024x1024 is the native canvas. Match the composition to the requested aspect
ratio: a portrait canvas wants a vertical subject with headroom, a wide canvas
wants a horizontal relationship between subject and setting. Around 28 steps
at a guidance of 4 to 7 is the quality reference for the undistilled
checkpoints. A four-step turbo checkpoint has its own recipe in its model leaf.

#### Examples

Input: victorian clocktower at sunset

Output: A Victorian clocktower fills the left third of a city square at sunset.
Its glass walls reveal interlocking brass gears while pedestrians cross below.
Low-angle architectural photograph, dramatic clouds.

Input: a bookshop sign that says Fable and Ink

Output: A narrow corner bookshop at dusk, its hand-painted sign reading "Fable
& Ink" centred above the door. Two customers browse a table of secondhand books
on the pavement. Warm window light against a blue evening street, 35mm
documentary photograph.

#### Pitfalls

- The CLIP window truncates at 77 tokens, so the opening sentence has to carry
  the composition on its own.
- Counting is dependable for a handful of objects and drifts for a crowd
  (community observation).
- A checkpoint running at guidance 1.0 ignores the negative prompt entirely.
- Naming the subject in the negative prompt removes it from the image.

#### CLI

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

#### Sources

- https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo
- https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_3
- Community: crowd-count drift is user-reported and not documented by Stability AI.

<!-- families/z-image.md -->

Default word limit: 150. Also accepted on the wire as `zimage`, `z_image`.

### Z-Image prompting

Manifest family: `z-image`.

#### Prompt style

The text encoder is a Qwen3 language model, so write grammatical sentences and
not a tag list. Long, detailed prompts are what the model was tuned for: one
continuous scene description built from subject, action, spatial arrangement,
setting, light, and finally medium or style. Keep it to one composition and
stay under 150 words. Write in English or Chinese; both are read
natively and neither needs translating.

#### Syntax

No weighting syntax. Text rendering is a headline strength in both Chinese and
English, so put visible lettering in quotation marks and name its position,
script, and style. The negative prompt is honoured only on the undistilled base
checkpoint, which runs guidance between 3 and 5. There is no reference-image
addressing to write.

#### Generation context

1024x1024 is the reference canvas and any aspect ratio between roughly 512 and
2048 on a side is supported, so state the framing that suits the chosen shape.
Turbo runs nine steps at guidance 0. Because the sampler is short, one clear
composition resolves better than several competing ones on the same canvas.

#### Examples

Input: astronaut in an underwater cave

Output: An astronaut floats through a bioluminescent underwater cave. The visor
reflects blue coral below while a shaft of sunlight falls from the opening
above. Wide science-fiction illustration with a crisp silhouette.

Input: a tea shop sign in Chinese and English

Output: A narrow tea shop at dusk on a wet city street. The lantern-lit sign
above the door reads "云间茶室" in gold brush script with "CLOUD ROOM TEA" in
small white capitals beneath it. Warm interior light spills onto the pavement,
realistic photography.

#### Pitfalls

- Turbo is guidance-distilled and runs at guidance 0, so a negative prompt has
  no effect at all. Remove unwanted content by describing what belongs in the
  frame instead.
- Comma-separated tag salad underperforms an equivalent sentence.
- Very long prompts can exceed the encoder sequence limit and lose the tail.
- Repeating the same prompt and seed produces near-identical output, so vary
  the wording rather than only the seed (community observation).

#### CLI

```bash
# Turbo at its published recipe
mold run z-image-turbo:q8 \
  "An astronaut floats through a bioluminescent underwater cave, helmet visor reflecting blue coral below and a shaft of sunlight above, wide science-fiction illustration, crisp silhouette" \
  --steps 9 --seed 777

# Bilingual signage, quoted so the lettering is rendered verbatim
mold run z-image-turbo:bf16 \
  'A narrow tea shop at dusk on a wet city street, the lantern-lit sign reading "云间茶室" in gold brush script with "CLOUD ROOM TEA" beneath it, realistic photography' \
  --steps 9 --seed 778
```

#### Sources

- https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8
- https://github.com/Tongyi-MAI/Z-Image
- Community: seed-versus-prompt variation is reported in the discussion thread.

<!-- families/hunyuan3d.md -->

Default word limit: 40.

### Hunyuan3D prompting

Manifest family: `hunyuan3d`.

#### Prompt style

Write no prompt. There is no text encoder anywhere in this family. The source
image is the entire conditioning, nothing typed in the prompt field reaches
the model, and a request without an image is refused rather than answered
from nothing. The 40-word budget is therefore unused: spend the
effort on the image instead. `mold expand` and `mold remix` say so and answer
with the image advice below instead of calling a language model.

#### Syntax

Nothing in the text field reaches the model, so there is no weighting, no
negative prompt, no quoted text, and no reference addressing to write. The one
image is passed as the source, not named in prose.

#### Generation context

Three properties of the image move the result, and none of them are prose.

- One object, centred, filling most of the frame. The model reconstructs what
  it can see; a subject occupying a tenth of the frame reconstructs at a tenth
  of the detail.
- A plain or removed background. There is no segmentation stage, so a busy
  background is read as geometry. An image with an alpha channel is the best
  input, because mold letterboxes on the cutout.
- A three-quarter view. A straight-on photograph gives no depth cue for the
  sides.

#### Examples

Input: 3-D model of my dining chair, photo attached

Output: No prompt. Supply chair.png cropped so the chair fills the frame,
background removed to alpha, shot from a three-quarter angle.

Input: turn this asset concept into a mesh

Output: No prompt. Supply concept.png with the single object centred on a plain
ground and every other prop cropped away.

#### Pitfalls

- Frames, fps, masks, ControlNet, and an explicit canvas are refused for this
  family rather than ignored.
- Output is always binary glTF, so `-o` must name a `.glb` file or `-` for
  stdout; a raster, video, or audio extension is refused before any weight is
  read. An explicit `png` in a request is coerced to `glb`, not refused.
- OBJ, STL, and PLY exist only as gallery exports of the stored glTF, never as
  generation targets, because each loses something the glTF carries.
- The same picture gives a different mesh here than in ComfyUI. mold prepares
  the image the way Tencent's `ImageProcessorV2` does: crop to the alpha
  bounding box, then letterbox on a white square, so nothing is cut away.
  ComfyUI's `clip_preprocess` drops the alpha channel and, with CLIP Vision
  Encode's default `crop: center`, centre-crops the shorter side to a square,
  so an off-centre or wide subject loses its edges there and keeps them here
  (`crop: none` squashes to a square instead, distorting rather than
  cropping); a threshold tuned on one is a fair start on the other, a crop
  is not.
- The web, desktop, and mobile apps' export menu offers the same OBJ, STL,
  and PLY geometry exports and the GIF, APNG, and WebP turntables as
  `mold library export` and the `export_mesh` MCP tool.
- Texturing, the 2.1 shape model, multi-view input, and text-to-3D are not
  supported, so today's result is geometry only.
- Detail is bought with the octree resolution and its cost is cubic.

#### CLI

```bash
# The default tier: 0.6B, step-distilled, ~5 GB VRAM
mold run hunyuan3d-mini-turbo --image chair.png -o chair.glb

# Undistilled 1.1B, 30 guided steps, higher detail
mold run hunyuan3d --image chair.png --octree 320 -o chair.glb

# Recover thin features by lowering the surface threshold
mold run hunyuan3d-turbo --image lamp.png --mesh-threshold 0.4 -o lamp.glb

# Export a saved mesh from the gallery as STL, OBJ, or PLY
mold library export chair.glb --format stl -o chair.stl

# Share a turntable: the poster spun a full turn as an animated GIF (or apng, webp)
mold library export chair.glb --format gif
mold library export chair.glb --format gif --playback bounce --repeat once --frames 24
```

`--octree`, `--mesh-threshold`, and `--target-faces` are the three mesh
controls, and the model's generation profile is the authority on their
values: its `capabilities.mesh` block advertises the octree allowlist and
default, the threshold range and default, and the face bounds, so read them
from the profile (`/api/models`) rather than from this page. `--octree` is
the detail knob and its cost is cubic. `--mesh-threshold` moves the extracted
surface: lower recovers thin features and adds noise; it is the same `[0, 1]`
occupancy scale ComfyUI's `VoxelToMesh` thresholds, so a value tuned there
carries over. `--target-faces` decimates after extraction and is absent
until asked for. `mold library export`, the `export_mesh` MCP tool, and the
gallery export menu all transcode the same stored `.glb`.

#### Sources

- https://github.com/Tencent-Hunyuan/Hunyuan3D-2
  (`hy3dgen/shapegen/preprocessors.py`, `ImageProcessorV2.recenter`: the
  alpha-bounding-box crop and white letterbox mold mirrors)
- https://github.com/comfyanonymous/ComfyUI (`comfy/clip_model.py`
  `clip_preprocess`: the centre crop; `comfy_extras/nodes_hunyuan3d.py`
  `VoxelToMesh`: the threshold scale)
- Best practice: the centred-subject, cutout-background, three-quarter-view
  image advice is community practice, not a published upstream rule.

<!-- families/wuerstchen.md -->

Default word limit: 50. Also accepted on the wire as `wuerstchen-v2`.

### Wuerstchen v2 prompting

Manifest family: `wuerstchen`.

#### Prompt style

Name a broad subject, its dominant colours, the lighting, and the medium, in
one or two sentences under 50 words. Both cascade stages read a
CLIP text encoder with a 77-token window, so extra clauses are truncated rather
than honoured. Ask for mood and palette. Do not ask for small, countable, or
finely structured detail.

#### Syntax

No weighting syntax. The negative prompt is supported and applied by the prior
stage, which runs guidance above 1; mold ships a defect-suppression default.
Use it for blur, muddy colour, and watermarks. There is no on-image text worth
quoting and no reference image to address.

#### Generation context

1024x1024 is the default canvas and resolution moves in 128-pixel steps, so the
next size up is 1152x1152. Choose an aspect ratio that suits one large subject
filling the frame.

#### Examples

Input: a lighthouse at sunset

Output: A lighthouse on a rocky coast during a dramatic sunset, bold oil
painting, vibrant orange and purple sky, crashing surf.

Input: a forest in autumn

Output: A dense autumn forest in low golden light, deep amber and rust foliage,
painterly landscape illustration with soft atmospheric haze.

#### Pitfalls

- The 42x latent compression costs fine detail, and it shows first in faces and
  hands.
- The model cannot render correct text in an image.
- Output is often not photorealistic, so painterly and illustrative requests
  suit it best.
- Difficult compositional prompts, meaning several named objects in stated
  positions, are unreliable.

#### CLI

```bash
# Painterly subject with a real negative prompt
mold run wuerstchen-v2:fp16 \
  "A lighthouse on a rocky coast during a dramatic sunset, bold oil painting, vibrant orange and purple sky, crashing surf" \
  --negative-prompt "fine text, watermark, muddy colors" --seed 42

# Next canvas up the 128-pixel ladder
mold run wuerstchen-v2:fp16 "A dense autumn forest in low golden light, painterly landscape illustration" --width 1152 --height 1152 --seed 43
```

#### Sources

- https://arxiv.org/abs/2306.00637
- https://huggingface.co/docs/diffusers/main/en/api/pipelines/wuerstchen
- https://huggingface.co/warp-ai/wuerstchen

<!-- families/qwen-image.md -->

Default word limit: 180. Also accepted on the wire as `qwen_image`.

### Qwen-Image prompting

Manifest family: `qwen-image`.

#### Prompt style

Rewrite the request into a complete, expressive prompt that preserves its
original meaning. For an overly brief input, reasonably infer and add detail so
the frame is visually complete, without altering the core content. Refine four
things in order: subject characteristics, visual style, spatial relationships,
and shot composition. Match the prompt to a precise, niche style aligned with
the stated intent; when no style is given, choose the most appropriate one,
usually realistic photography. Keep the rewrite under 180 words,
then append `Ultra HD, 4K, cinematic composition` (a Chinese prompt appends
`超清，4K，电影级构图`).

#### Syntax

Enclose on-image text in quotation marks and name its position, such as the
top-left corner, along with its style. Quoted text is never translated and
never altered. Do not add lettering the user did not ask for. Avoid negation in
the positive prompt: describe what should be present rather than what should be
absent. The negative prompt is honoured whenever guidance stays above 1. No
weighting syntax, and no reference-image addressing on this family.

#### Generation context

1328x1328 is the default canvas; match the composition to the requested aspect
ratio. The base and 2512 checkpoints at fifty steps are the quality reference
and carry a dense art-direction brief. The few-step distills need a simpler
prompt and run their own fixed step-and-guidance recipe, described in their
model leaf.

#### Examples

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

#### Pitfalls

- A vague text request such as "a sign with the date" renders as garbled
  lettering. Write the exact string the image should carry.
- Negation words leak into the frame; rephrase them as what should be there.
- Dense small text, hair-fine texture, and very complex scenes degrade on the
  few-step distills.

#### CLI

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

#### Sources

- https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/src/examples/tools/prompt_utils.py
- https://github.com/QwenLM/Qwen-Image
- https://huggingface.co/Qwen/Qwen-Image

<!-- families/qwen-image-edit.md -->

Default word limit: 100. Also accepted on the wire as `qwen_image_edit`.

### Qwen-Image Edit prompting

Manifest family: `qwen-image-edit`.

#### Prompt style

Write one imperative edit instruction, direct and specific, under
100 words. Keep the core intention of the request unchanged and add
only clarity, rationality, and visual feasibility. For an add, delete, or
replace, supplement a vague request with minimal but sufficient detail:
category, colour, size, orientation, position. Phrase a swap as "Replace Y with
X" and briefly describe X's key visual features. Every added object must match
the logic and style of the source image. Name what must stay identical, such as
geometry, camera position, composition, and any subject the edit does not
touch.

#### Syntax

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

#### Generation context

The source image owns appearance, geometry, and canvas, so prompt only the
change. A person edit keeps ethnicity, gender, age, hairstyle, expression, and
outfit consistent, and any expression, beauty, or make-up change must be
natural and subtle, never exaggerated. When a style is transferred from a
second reference, describe that style in key visual features and place the
style clause last.

#### Examples

Input: make the bakery photo wintry

Output: Change the sunny spring morning into a snowy winter blue hour. Add
fresh snow to the exterior and turn on warm interior lights. Preserve the exact
teal facade, the readable "MOLD & FLOUR" sign, windows, door, camera position,
geometry, and composition.

Input: put the person from the second photo into the first

Output: Replace the woman in the first image with the man from the second
image, matching the first image's lighting and camera angle. Keep the first
image's background, furniture, and composition unchanged.

#### Pitfalls

- Describing the whole scene instead of the change makes the model repaint what
  you described. State the edit, then the invariants.
- Omitting what must stay identical is the usual cause of drifted geometry.
- Translating quoted text or changing its case rewrites the sign.
- Exaggerated face or make-up edits break identity.
- One to three input images are the tested quality range. Additional images
  remain accepted, but may reduce consistency.

#### CLI

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

#### Sources

- https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/src/examples/tools/prompt_utils.py
- https://github.com/QwenLM/Qwen-Image
- https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- https://huggingface.co/Qwen/Qwen-Image-Edit

<!-- families/ltx-video.md -->

Default word limit: 150. Also accepted on the wire as `ltx_video`, `ltxvideo`.

### LTX-Video 0.9.x prompting

Manifest family: `ltx-video`.

#### Prompt style

Write one flowing paragraph, chronological, literal and precise, the way a
cinematographer describes a shot list. Start directly with the main action in a
single sentence. Then add the specific movements and gestures, precise
appearances of the characters and objects, the background and environment, the
camera angle and movement, lighting and colors, and last any change or sudden
event. Keep within 150 words. Prefer visible action over abstract
mood: concrete observable detail steers this model, adjectives about feeling do
not.

#### Syntax

No weighting or attention syntax. A guidance-based checkpoint takes a negative
prompt for visible defects; a distilled checkpoint is pinned at guidance 1.0 and
takes none, so do not write one for it. Visible lettering is unreliable, so keep
on-image text out of the prompt. Mold's legacy LTX-Video engine is text-to-video
only: it rejects source images and requires a prompt. Use LTX-2.3 or LTX-2.5 for
image-conditioned or continuation work. Camera vocabulary is plain English:
the camera slowly tilts upward, tracks alongside, pushes in, pulls back, or
holds locked off. Use one move per clip.

#### Generation context

Frames follow the 8n+1 grid and both dimensions are multiples of 32. The default
canvas is 1216x704 at 30 fps with 25 frames, which is under a second, so one
beat is all that fits; 49 frames is about 1.6 seconds and 97 about 3.2. A longer
one-shot remains a single denoise through the 257-frame engine ceiling; it is
never automatically split at 97 because the legacy engine cannot pass motion
context across a seam. Explicitly authored sequences remain separate clips, so
write each stage as an intentional shot and use a cut where independent motion
is acceptable.

#### Examples

Input: northern lights over a frozen lake

Output: Northern lights ripple from left to right over a frozen lake; green and
violet ribbons reflect in the ice while the camera slowly tilts upward, one
continuous time-lapse shot.

Input: a chef plating a dish

Output: A chef's hands lower a seared scallop onto a white plate with tongs,
then trail a spoon of green oil around the rim. Steam rises from the plate.
Stainless counters and hanging pans fill the blurred background under warm
overhead light. The camera holds a locked-off overhead close-up.

#### Pitfalls

Legacy LTX-Video renders silent video, so never promise speech, music, or sound
effects. It is not LTX-2, and a prompt written for that audio-video family does
not transfer. Avoid cuts, a second shot, and more action than a short clip can
hold. The 0.9.6 distilled checkpoint is the safest default, the 0.9.8
checkpoints run the full multiscale refinement path, and the 13B BF16 tiers need
a 40 GB-class GPU.

#### CLI

```bash
# Basic clip (25 frames, MP4 default in a build carrying the mp4 feature)
mold run ltx-video-0.9.6-distilled:bf16 "a cat walking across a windowsill" --frames 25
# Frame counts are 8n+1 (9, 17, 25, 33, 49, ...)
mold run ltx-video-0.9.8-2b-distilled:bf16 "ocean waves at sunset" --frames 49
# Explicit MP4 output
mold run ltx-video-0.9.6-distilled:bf16 "a campfire at night" --frames 17 --format mp4
# GIF (256 colors)
mold run ltx-video-0.9.6-distilled:bf16 "a sunset" --frames 17 --format gif -o sunset.gif
# Animated WebP output (needs the webp feature)
mold run ltx-video-0.9.6-distilled:bf16 "a waterfall" --frames 9 --format webp -o waterfall.webp
# A finished chronological prompt
mold run ltx-video-0.9.6-distilled:bf16 \
  "Northern lights ripple from left to right over a frozen lake; green and violet ribbons reflect in the ice while the camera slowly tilts upward, one continuous time-lapse shot" \
  --frames 33 --seed 1234
```

#### Sources

- https://github.com/Lightricks/LTX-Video (README prompt engineering section)
- https://huggingface.co/Lightricks/LTX-Video
- Community: https://github.com/Lightricks/ComfyUI-LTXVideo

<!-- families/ltx2.md -->

Default word limit: 200. Also accepted on the wire as `ltx-2.3`, `ltx-2.5`, `ltx2.3`, `ltx2.5`.

### LTX-2 prompting

Manifest family: `ltx2`.

#### Prompt style

Write one flowing paragraph under 200 words, four to eight
sentences. Start directly with the action, never with "The scene opens" or "We
see". Add movement, appearance, environment, camera, lighting, and colour,
then any sudden change. Use present-progressive verbs and strict chronological
order joined by "as", "then", and "while". Describe only what is visible or
audible: physical cues, not emotion labels. Keep wording restrained and
colours plain, so "red dress" rather than "vibrant red dress".

#### Syntax

No weighting, no negative prompt. When the style is known, prefix the
paragraph with `Style: <style>,`. LTX-2 generates audio, so integrate the
soundscape beside the actions rather than appending it. Quote speech exactly,
keep the user's words verbatim, and name voice, delivery, accent, and language
when it is not English. Camera vocabulary is plain film language: static
frame, pans, tilts, pushes in, pulls back, tracks, arcs, handheld. Never
invent camera motion or dialogue the user did not request, and never write
timestamps or cuts.

#### Generation context

Frames sit on the `8k+1` grid at 24 fps, so 121 frames is about five seconds.
Both axes must be multiples of 32, and of 64 for lip dub. Size the action to
that duration, keep one continuous take per clip, and keep lip sync inside a
single clip. With a source image attached, describe only what changes from it;
restating the frame inaccurately makes the model cut. Distilled LTX-2.5 fixes
guidance at 1.0. Audio renders by default on MP4 output, one-shots and
sequences alike, so write the soundscape unless the request says silent.

#### Examples

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

#### Pitfalls

Readable on-screen text is unreliable, so avoid signage and logos. Avoid fast
twisting motion, crowded layered scenes, and conflicting light sources.

#### CLI

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

#### Sources

- https://github.com/Lightricks/LTX-2 (README, "Prompting for LTX-2")
- LTX-2 official prompt-enhancer system prompts, `packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/` in the same repository
- https://ltx.io/blog/prompting-guide-for-ltx-2
- https://docs.ltx.io/open-source-model/usage-guides/prompting-guide

<!-- families/wan.md -->

Default word limit: 100. Also accepted on the wire as `wan2.1`, `wan2.2`, `wan21`, `wan22`.

### Wan prompting

Manifest family: `wan`.

#### Prompt style

Write one English shot description of about 100 words. Infer missing
detail for a short input without changing its intent. Enhance what the user
named: appearance, expression, quantity, posture, visual style, spatial
relationships, shot scale. Never add a subject the input lacks. Detail how the
action unfolds and give background elements their own motion. Use simple direct
verbs. Emphasize motion and camera movement. Skip literary mood writing. Name a
style only when the user did, and put it first; a 2D style takes no cinematic
terms.

#### Syntax

Wan has no weighting syntax. The server prefills the model's tuned default
negative prompt, a standard Chinese quality list; never restate its terms in the
positive. Keep quoted text verbatim.

Add at most four cinematic settings, drawn from time, light, tone, composition,
shot size, and camera angle. Shot size: Extreme close-up shot, Close-up shot, Medium close-up shot, Medium
shot, Medium wide shot, Wide shot, Extreme wide shot; default Medium shot or
Wide shot. Camera angle: Over-the-shoulder shot, Low angle shot, High angle
shot, Dutch angle shot, Aerial shot, Overhead shot; skip it when the request
already gives a camera move.

#### Generation context

Wan runs at 16 fps on a 4k+1 frame grid, so 49 frames is about three seconds and
81 about five. Fit the action to that: two or three beats, one shot, one camera
move. A longer request renders as chained clips, and the seam carries motion
only on an image-conditioned checkpoint.

#### Examples

Input: a fox in the snow

Output: Medium wide shot, day time, side lighting. A red fox trots through fresh
snow in a pine forest, breath drifting back, powder lifting from each paw. The
camera tracks alongside.

Input: a paper boat in a rain gutter

Output: Close-up shot, overcast lighting. A folded paper boat drifts down a
gutter, spins once against a leaf, then straightens and speeds up as the camera
pushes forward.

#### Pitfalls

Wan generates silent video. Never request dialogue or sound, or claim
synchronized audio. Wan S2V is a separate speech-to-video model. Wan 2.2 A14B
drives two experts from this one prompt, so keep it internally consistent. Read
one task leaf: text-to-video for T2V identities, image-conditioned for I2V and
TI2V identities or any source frame.

#### CLI

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
# 3-step DMD distill of the same 5B, text-to-video only (steps/solver/shift pinned, shift-5 table)
mold run wan22-ti2v-5b:dmd "waves on a black sand beach" --width 1280 --height 704 --frames 121 --fps 24
# Sequences: past the per-clip envelope this auto-chains and stitches one MP4
# delivering exactly the requested total (keep --frames on the 4k+1 grid).
# The seam continues only on an image-conditioned checkpoint; clips are 4k+1.
# A text-to-video tier refuses the split instead: it carries nothing across a
# seam, so a longer --frames would repeat the clip. Stay inside its clip size.
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

#### Sources

- https://github.com/Wan-Video/Wan2.2
- https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/system_prompt.py
- https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/prompt_extend.py
- https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/shared_config.py

<!-- families/minimax-h3.md -->

Default word limit: 250.

### MiniMax H3 prompting

Manifest family: `minimax-h3`.

#### Prompt style

Write the three core fields in order: `integrated_multimodal_description`,
`overall_soundscape`, `non_diegetic_music`. Every detail must be visible or
audible. Open `[Shot 1]` with the visual style and initial composition, then
subjects, scene, and actions. Prefer concrete detail over abstract words.
Write in English, preserving the original language of dialogue, lyrics, and
visible text. Stay within 250 words.

#### Syntax

`[Shot 1]` takes no timestamp. Later shots open `[Shot n] At MM:SS.mmm,` with
a strictly increasing cut time inside the duration. Write camera motion as
natural English in the shot: type, amplitude, speed. The types are Zoom
In/Out, Push In/Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down,
Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake
Slightly/Strongly, POV, and Roll Clockwise/Counterclockwise. Omit amplitude
and speed when medium and normal. Give each vocal source a stable id such as
`(S1)`, with its identifying phrase and delivery outside the tag. Inside
`<d>[English] ...</d>` put only the language tag and the verbatim words. For
voiceover use the exact phrase "says in an off-screen voiceover", then state
that the character's lips remain closed. Use `<scenetrans>` where a line
crosses a cut and `<cutoff>` where speech is truncated by the ending. Quote
visible on-screen text verbatim. Reference labels `<Picture n>`, `<Video n>`,
and `<Audio n>` keep one meaning across every section.

#### Generation context

Match the described duration to the requested four to fifteen seconds.
`overall_soundscape` is one to four sentences of ambience, action sounds, and
non-verbal sounds, never repeating dialogue or music. `non_diegetic_music`
covers instrumentation, tempo, and dynamics; `N/A` when no score is wanted.

#### Examples

Input: a baker opens the shutters before sunrise and says one line

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening a street bakery's shutters before sunrise. The camera pushes in at slow speed as the baker with a calm, raspy voice (S1) says: <d>[English] First batch of the morning.</d>

overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo.
```

#### Pitfalls

Avoid plot summaries, unresolved labels, and timing that misses the duration.
Expansion produces this grammar when given the H3 route. Check live runtime
availability before promising speech.

#### CLI

Read exactly one direct task leaf from `SKILL.md`: base modes for FL2VA
identities, or Ref2VA for reference-conditioned identities.

```bash
# Feed a written Context-IR prompt from a file
mold run minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p "$(cat h3-prompt.txt)" --first-frame presenter.png --duration 5 --seed 83009
```

#### Sources

- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md
- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt
- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt
- https://huggingface.co/MiniMaxAI/MiniMax-H3

<!-- families/upscaler.md -->

Default word limit: 20.

### Upscaler prompting

Manifest family: `upscaler`.

#### Prompt style

Write no prompt. Real-ESRGAN upscalers have no text encoder, so the
20-word budget is unused and the model choice does all the work.

#### Syntax

No weighting, no negative prompt, no quoted text, and no reference addressing.

#### Generation context

The scale factor is fixed by the chosen model, 2x or 4x, and the canvas follows
the source image. Pick the general model for photographs and mixed artwork, or
an anime model for line art. Lower the tile size when VRAM is tight.

#### Examples

Input: make this photo bigger

Output: No prompt. Run the general 4x model and compare against the original.

Input: upscale this anime cel

Output: No prompt. Run an anime 4x model so the line art stays clean.

#### Pitfalls

- Judge fine texture, halos, and facial detail against the original at 100
  percent zoom.
- Upscaling cannot recover detail the source never held.

#### CLI

```bash
# Default general model, auto-downloads on the host that runs the job
mold upscale photo.png

# Anime line art
mold upscale photo.png -m real-esrgan-x4plus-anime:fp16

# Custom output path with an inline preview
mold upscale photo.png -o photo_4x.png --preview

# Smaller tile size for limited VRAM
mold upscale large_photo.png --tile-size 256
```

#### Sources

- https://github.com/xinntao/Real-ESRGAN

## Task leaves

A task leaf is added below its family base when the model identity or the expansion task selects it.

<!-- minimax-h3/base-modes.md -->

Family `minimax-h3`; tasks: `text-to-video`, `image-to-video`, `keyframe-interpolation`; word limit: 250.

### MiniMax H3 base modes

#### Prompt style

T2VA builds the whole timeline from text. I2VA, FL2VA, and L2VA prepend one
instruction line, a blank line, then the three core fields:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

Preserve the frame's identity, clothing, layout, lighting, and composition,
then describe one continuous observable path forward. Never contradict it.

#### Examples

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, the presenter shown in <Picture 1> retains the exact face, clothing, workstation layout, lighting, and framing. The camera trucks right at slow speed as the presenter turns toward the lens. The presenter with a bright voice (S1) says: <d>[English] With Mold, your ideas render right here.</d> Their lips synchronize; they gesture once.

overall_soundscape: Low workstation airflow continues beneath clean close-miked speech.

non_diegetic_music: N/A
```

#### CLI

```bash
# First-frame conditioning; the prompt file holds the instruction line plus the three fields
mold run minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p "$(cat h3-prompt.txt)" --first-frame presenter.png --duration 5 --seed 83009
```

#### Sources

- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt

<!-- minimax-h3/ref2va.md -->

Family `minimax-h3`; tasks: `reference-to-audio-video`; word limit: 300.

### MiniMax H3 Ref2VA

#### Prompt style

Ref2VA replaces the three core fields with six, in order:
`subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`. Establish
the style in one sentence before `[Shot 1]`. State which traits each reference
owns, and never let one reference silently replace another's identity, motion,
or sound role.

#### Syntax

Label references `<Picture n>`, `<Video n>`, and `<Audio n>` in the order
supplied, counting each category independently. A reference video carrying a
soundtrack also takes the next `<Audio n>` label first, so strip unwanted
audio before upload. Retention markers are `fully_preserved`,
`partially_preserved`, `attribute_transfer`, and `weak_reference`; audio uses
`fully_copy`, `partially_copy`, `reference`, and `weak_reference`.

#### Generation context

A five-second clip normally stays one continuous shot. Keep speech short
enough to finish before the final pose.

#### CLI

References upload through authenticated endpoints, so `MOLD_API_KEY` must be
set. The supplied order is part of the render.

```bash
# Ordered reference set: images, then video, then audio
mold run minimax-h3-ref2va:comfy-pruned-int8 "$(cat h3-ref-prompt.txt)" --reference image=a.png --reference video=b.mp4 --reference audio=c.wav
```

#### Sources

- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt

<!-- wan/text-to-video.md -->

Family `wan`; tasks: `text-to-video`; word limit: 100.

### Wan text-to-video

#### Prompt style

Use for T2V identities. Open with the shot size, then the subject and setting,
then the action in two or three beats, then background motion and lighting.
Close with the one camera move. Keep to about 100 words.

#### Generation context

One continuous shot per clip, no cuts. A T2V checkpoint cannot carry a chain
seam, so clips in a longer sequence are independent shots. Ask for an
image-conditioned checkpoint when motion must continue across clips.

#### Examples

Input: a fox in the snow, tracking shot

Output: Medium wide shot at sunrise, side lighting. A red fox trots through
fresh snow in a pine forest; breath drifts back and powder lifts from each paw.
The camera tracks alongside in one continuous move.

#### Pitfalls

Do not name two camera moves or a second unrelated action.

#### CLI

```bash
mold run wan22-t2v-a14b:q4 \
  "Medium-wide wildlife shot at sunrise. A red fox trots steadily through fresh snow in a quiet pine forest; visible breath drifts back and powder lifts from each paw. The camera tracks alongside at eye level in one continuous move. Soft side light, realistic fur, stable background, no text." \
  --width 832 --height 480 --frames 49 --fps 16 --seed 83005
mold run wan21-t2v-1.3b "a red fox trotting through snow" --frames 81 --fps 16
```

#### Sources

- https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/prompt_extend.py
- https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/system_prompt.py

<!-- wan/image-conditioned.md -->

Family `wan`; tasks: `image-to-video`, `keyframe-interpolation`; word limit: 80.

### Wan image-conditioned video

#### Prompt style

Use for I2V and TI2V identities and any source frame. Stay under 80
words and write dynamics only. Drop description already visible in the frame,
which owns appearance. Supplement a bare action with the visible subject, so
"dancing" becomes "the girl is dancing". Keep and emphasize camera phrases such
as "the camera pans up".

#### Generation context

Name what must stay stable. For a first and last frame pair, describe a
physically continuous path between them and the change that happens: walking
into, appearing, turning into, camera left, camera right. Check the model row
for its supported source roles.

#### Examples

Input: dancing

Output: The girl is dancing, turning slowly on the spot as her skirt lifts. The
camera pushes forward.

#### Pitfalls

Do not redescribe or contradict the frame.

#### CLI

```bash
mold run wan22-i2v-a14b:q5 "the balloon lifts off" --image balloon.png
mold run wan22-i2v-a14b:q5 "the sapling grows into an oak" --image sapling.png --last-image oak.png
mold run wan22-ti2v-5b:q8 "the paper boat drifts on down the gutter" --image boat.png --frames 100 --clip-frames 49
```

#### Sources

- https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/system_prompt.py
- https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/prompt_extend.py

<!-- ltx2/dub-it.md -->

Family `ltx2`; tasks: `audio-driven-video`; word limit: 120.

### LTX-2 Dub-It

#### Prompt style

The reference video owns timing, visible speakers, cuts, and mouth motion.
Write replacement speech that fits the observed speaking windows, identify
each speaker consistently, and describe voice, accent, emotion, and delivery.
Do not change the visual action. Preserve the existing ambience unless asked
to replace it.

#### Generation context

Duration is not yours to choose: the reference supplies frames and fps, and
both axes must be multiples of 64. Fit the new line inside the same speaking
window, because a longer line desynchronizes the mouth. The reference must
carry an audio track, since its speech is what the dub imitates.

#### Examples

Input: make the presenter say the new tagline in a calmer tone

Output: The same speaker says in a calm, lower voice, "We render it locally
now." The room tone underneath is unchanged.

#### CLI

Inspect `mold --help` and the selected host's capabilities for the current
Dub-It invocation; do not infer support from the family name alone. Verify the
returned clip before reporting success.

```bash
# Re-voice a clip of someone speaking; frames and fps come from the reference
mold run ltx-2.3-22b-distilled:fp8 "she says: the harbour freezes every winter" --ic-lora-control lipdub --video speaker.mp4 --width 704 --height 448
```

#### Sources

- https://github.com/Lightricks/LTX-2 (`DubItPipeline`, `packages/ltx-pipelines/src/ltx_pipelines/dubit.py`)

<!-- ltx2/text-to-audio.md -->

Family `ltx2`; tasks: `text-to-audio`; word limit: 120.

### LTX-2 text-to-audio

#### Prompt style

Describe the audible result rather than a camera scene: sound sources, their
sequence, the space, distance, dynamics, and texture. For speech, give the
exact short line and describe voice and delivery. Keep dialogue, ambience,
effects, and music in separate clauses so their roles do not conflict.

#### Generation context

Duration comes from `--frames` and `--fps`, so 121 frames at 24 fps is about
five seconds. Write only as much sound as fits that span. Text-to-audio
rejects every conditioning input and renders no picture, so describe nothing
visual.

#### Examples

Input: rain on a roof

Output: Heavy rain drums on a tin roof at a steady rate as water spills from a
gutter to the left. Distant thunder rolls twice, seconds apart, with no music.

#### CLI

Inspect `mold --help` and the selected host's capabilities for the current
text-to-audio invocation. Verify the returned artifact is audio before
reporting success.

```bash
# Audio only, 16-bit stereo WAV; duration is frames divided by fps
mold run ltx-2.3-22b-dev:fp8 "heavy rain on a tin roof, distant thunder" --pipeline t2a --frames 121 --fps 24 --output rain.wav
```

#### Sources

- https://github.com/Lightricks/LTX-2 (audio VAE and vocoder, `packages/ltx-pipelines`)

## Model leaves

A model leaf carries quirks of one checkpoint and is added after the task leaf.

<!-- models/flux-schnell.md -->

Models: `flux-schnell`.

### FLUX.1 Schnell prompting

#### Prompt style

A four-step distilled FLUX.1 for drafts and thumbnails. Give it one subject,
one action, one setting, and one style word. Guidance is fixed at 0.

#### Syntax

A negative prompt does nothing: Schnell has no negative branch, and true CFG
needs an identity reference. Quoted lettering is unreliable at four steps.

#### Pitfalls

Piling on detail returns the average of it rather than all of it. Move to Dev
or Krea when materials, small text, or exact spatial relationships matter.

#### CLI

```bash
mold run flux-schnell:q8 "a red fox asleep on a mossy log, soft morning light" --steps 4
mold run flux-schnell:bf16 "a lighthouse in a storm, dramatic illustration" --steps 4 --seed 12
```

#### Sources

- https://huggingface.co/black-forest-labs/FLUX.1-schnell
- https://docs.bfl.ai/guides/prompting_unified_basics

<!-- models/flux2-klein-base.md -->

Models: `flux2-klein-base`, `flux2-klein-base-9b`.

### FLUX.2 Klein Base prompting

#### Prompt style

The undistilled 4B and 9B Klein weights, at 50 steps and guidance 4.0, so they
carry a far denser brief than four-step Klein. Keep FLUX.2's Subject + Action +
Style + Context order.

#### Syntax

The only FLUX.2 tier where a negative prompt works: above guidance 1.0 it runs
a real unconditional branch. `--guidance 1` skips it and the negative prompt
goes inert.

#### Pitfalls

Guidance above 1.0 costs two forward passes per step. Reusing a four-step Klein
recipe wastes the model.

#### CLI

```bash
mold run flux2-klein-base:q8 "a brass orrery on a walnut desk, low winter sun through tall windows" \
  --steps 50 --guidance 4.0 --negative-prompt "blurry, warped rings, text, watermark"

mold run flux2-klein-base-9b:q6 "an empty greenhouse at dawn, condensation on the glass, terracotta pots" \
  --steps 50 --guidance 4.0
```

#### Sources

- https://docs.bfl.ai/guides/prompting_guide_flux2
- https://huggingface.co/black-forest-labs/FLUX.2-klein-base

<!-- models/sdxl-turbo.md -->

Models: `sdxl-turbo`.

### SDXL Turbo prompting

#### Prompt style

One simple idea in a short sentence: subject, setting, treatment. Turbo renders
in four steps at guidance 0, so extra clauses average out instead of resolving.
Its native canvas is 512x512.

#### Syntax

Guidance 0 runs no classifier-free branch, so a negative prompt has no effect
here. Never pass true CFG. Turbo is the one SDXL checkpoint that identity
conditioning does not accept.

#### Pitfalls

Raising steps and guidance to base-SDXL values degrades Turbo rather than
improving it. Long tag lists are wasted tokens.

#### CLI

```bash
mold run sdxl-turbo:fp16 "a red canoe on a still lake at dawn" --steps 4 --seed 88
mold run sdxl-turbo:fp16 "neon ramen counter at night, documentary photograph" --width 512 --height 512
```

#### Sources

- https://huggingface.co/stabilityai/sdxl-turbo

<!-- models/pony-v6.md -->

Models: `pony-v6`, `cyberrealistic-pony`.

### Pony Diffusion V6 XL prompting

#### Prompt style

Booru tags, comma separated. Open with the score prefix `score_9, score_8_up,
score_7_up, score_6_up, score_5_up, score_4_up`, then a source tag
(`source_anime`, `source_pony`, `source_furry`, `source_cartoon`), then a
rating tag (`rating_safe`, `rating_questionable`, `rating_explicit`), then
subject tags. CyberRealistic Pony uses the same prefix, photographic tags.

#### Syntax

The author says it needs no negative prompt in most cases and warns off quality
words like masterpiece. Community practice adds `score_6, score_5, score_4`
there.

#### Pitfalls

Without the score prefix, output looks washed out. mold has no clip skip
control.

#### CLI

```bash
mold run pony-v6:fp16 \
  "score_9, score_8_up, score_7_up, source_anime, rating_safe, 1girl, red scarf, snowy street, city lights, night" \
  --negative-prompt "score_6, score_5, score_4, blurry, watermark" --seed 606

mold run cyberrealistic-pony:fp16 \
  "score_9, score_8_up, score_7_up, rating_safe, portrait photograph of a woman, freckles, soft window light, 85mm" \
  --negative-prompt "score_6, score_5, score_4, cartoon, text, watermark"
```

#### Sources

- https://civitai.com/models/257749/pony-diffusion-v6-xl
- https://huggingface.co/AstraliteHeart/pony-diffusion-v6

<!-- models/playground-v2.5.md -->

Models: `playground-v2.5`.

### Playground v2.5 prompting

#### Prompt style

An aesthetic SDXL fine-tune tuned for photographic and painterly quality. It
likes short natural prompts: one subject, one setting, one lighting note. Extra
tags and quality words do not add polish, because the aesthetic is baked in.

#### Syntax

An ordinary SDXL negative prompt applies. There are no trigger words.

#### Pitfalls

It runs the EDM DPM++ 2M scheduler at 50 steps and guidance 3.0. Pushing
guidance toward SDXL's 7.5 oversaturates and hardens the image.

#### CLI

```bash
mold run playground-v2.5:fp16 "a quiet harbour at first light, fishing boats at anchor" \
  --steps 50 --guidance 3.0 --seed 250

mold run playground-v2.5:fp16 "portrait of a ceramicist in her studio, soft north light" \
  --negative-prompt "blurry, text, watermark"
```

#### Sources

- https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic

<!-- models/sd3.5-large-turbo.md -->

Models: `sd3.5-large-turbo`.

### SD 3.5 Large Turbo prompting

#### Prompt style

One idea, one or two sentences, well under 150 words. This
checkpoint is distilled to four steps, so a dense multi-clause brief resolves
worse than a single subject with its setting and light.

#### Syntax

It runs at guidance 1.0, so the negative prompt is ignored. Exclude things by
describing what belongs in the frame instead. Quoted on-image text still works.

#### Pitfalls

Crowded scenes and fine texture degrade against SD3.5 Large. Raising the step
count does not buy back that fidelity.

#### CLI

```bash
mold run sd3.5-large-turbo:q8 "A lone sailboat on a mirror-flat lake at dawn, wide landscape photograph" --steps 4 --seed 2026
```

#### Sources

- https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo

<!-- models/qwen-image-flash.md -->

Models: `qwen-image-flash`, `qwen-image-distill`, `qwen-image-lightning`.

### Qwen-Image few-step distills prompting

Covers `qwen-image-flash`, `qwen-image-distill`, and `qwen-image-lightning`.

#### Prompt style

Simplify. Keep the official rewrite rules but drop the densest art direction
and stay well under 180 words. One subject, one setting, one style.

#### Syntax

All three run CFG-free at guidance 1.0, so the negative prompt is ignored.
Quoted on-image text still works but keep it short.

#### Pitfalls

Fixed step recipes: flash 4, distill 15, lightning 4 or 8 by tag. Dense small
text, hair-fine detail, and very complex scenes degrade against base.

#### CLI

```bash
# NVIDIA DMD2 four-step distill of base
mold run qwen-image-flash:q8 "A red enamel teapot on a sunlit windowsill, still-life photograph" --steps 4 --seed 251203

# DiffSynth 15-step distill: closer to base fidelity
mold run qwen-image-distill:q8 "A quiet harbour at dawn, fishing boats at anchor, realistic photography" --steps 15 --seed 251204

# Lightning, eight-step tag
mold run qwen-image-lightning:fp8-8step "A snowy mountain village at blue hour, warm window lights" --steps 8 --seed 251205
```

#### Sources

- https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/src/examples/tools/prompt_utils.py
- https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning

<!-- models/qwen-image-edit-lightning.md -->

Models: `qwen-image-edit-lightning`.

### Qwen-Image-Edit Lightning prompting

#### Prompt style

A four-step fused Lightning distill of Edit-2511. Keep the edit simple: one
change plus the invariants, well under 100 words. Chained or
multi-part instructions resolve poorly in four steps.

#### Syntax

It runs at guidance 1.0, so the negative prompt is ignored. Quoted text edits
and ordinal image addressing behave as on the base edit checkpoint.

#### Pitfalls

Fine texture and small lettering degrade against Edit-2511. Prefer the base
checkpoint when the edit has to survive close inspection.

#### CLI

```bash
mold run qwen-image-edit-lightning:fp8 'Replace the sign text with "CLOSED", keeping the font, colour, and placement.' --image shop.png --seed 251112
```

#### Sources

- https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning
- https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/src/examples/tools/prompt_utils.py

<!-- models/ltx-2.5.md -->

Models: `ltx-2.5-22b-dev`, `ltx-2.5-22b-distilled`.

### LTX-2.5 22B

#### Generation context

Both base names resolve to the `int8-conv` pack by default. The prompt is
encoded by a packed Gemma 4 text encoder this family requires. Audio renders
by default on MP4 output; `--audio` states it explicitly and `--no-audio`
suppresses it. `--predict-duration` lets a qualified duration head choose a
one to twenty second clip in place of `--frames`, so describe an action whose
length is not fixed.

#### Pitfalls

The distilled recipe fixes guidance at 1.0. Do not override it.

#### CLI

```bash
# Distilled default pack, duration head picks the clip length
mold run ltx-2.5-22b-distilled "a complete product reveal" --predict-duration --fps 24 --audio --format mp4

# Base checkpoint at an explicit length
mold run ltx-2.5-22b-dev "a slow flyover of a salt flat at dawn" --frames 121 --fps 24 --audio
```

#### Sources

- https://github.com/Lightricks/LTX-2 (LTX-2.5 checkpoints and duration head)
- https://huggingface.co/Lightricks/LTX-2.5

<!-- models/wan22-ti2v-5b.md -->

Models: `wan22-ti2v-5b`.

### Wan 2.2 TI2V-5B

#### Generation context

The family's 720p path: 1280x704 at 24 fps, 4k+1 frames, both dimensions a
multiple of 32. The `fp16`, `q8` and `turbo` tiers serve text-to-video and
image-conditioned work from one checkpoint, so a chain seam continues on them
and a long sequence keeps one motion. The `dmd` tier is text-to-video only: it
refuses a source image, so write its prompt to carry the whole shot.

#### Pitfalls

At 24 fps a 121-frame clip runs five seconds, so pace the action faster than on
the 16 fps tiers. Keep a `:turbo` prompt to one simple idea.

#### CLI

```bash
mold run wan22-ti2v-5b "waves on a black sand beach" --width 1280 --height 704 --frames 121 --fps 24
mold run wan22-ti2v-5b:q8 "a paper boat drifting down a rain gutter" --frames 100 --clip-frames 49
mold run wan22-ti2v-5b:turbo "waves on a black sand beach" --width 1280 --height 704
mold run wan22-ti2v-5b:dmd "waves on a black sand beach" --width 1280 --height 704 --frames 121 --fps 24
mold run wan22-ti2v-5b:q8 "the balloon lifts off" --image balloon.png --frames 49
```

#### Sources

- https://github.com/Wan-Video/Wan2.2
- https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

