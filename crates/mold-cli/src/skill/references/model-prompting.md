# Model prompting recipes

Read only the section for the selected model family. Check `mold info <model>`
or the remote `/api/models` row before choosing dimensions, frames, steps, or
conditioning: installed catalog checkpoints can differ from manifest defaults.
Use a fixed seed while refining a prompt, then try several seeds for final art.

## FLUX.1

Applies to Schnell, Dev, Krea, and FLUX.1 fine-tunes. Put the subject and action
first, then composition, lighting, lens/material cues, and style. FLUX follows
natural sentences well; avoid long tag piles and conflicting styles. Schnell is
best for drafts at four steps. Dev/Krea and quality fine-tunes reward precise
materials, spatial relationships, and lighting.

```bash
mold run flux-dev:q4 \
  "A cozy Japanese tea house interior, two ceramic cups steaming on a low cedar table, warm paper lanterns, rain visible beyond the open shoji, intimate eye-level composition, delicate watercolor texture" \
  --seed 1337
```

## FLUX.2

Applies to Klein 4B/9B, Dev, and compatible fine-tunes. Use direct prose with a
clear subject, scene geometry, action, and finish. Klein is a four-step model:
one coherent visual idea is more reliable than many competing details. Dev can
carry a denser art-direction brief.

```bash
mold run flux2-klein-9b:q4 \
  "Macro photograph of a glass bottle ship caught inside a curling ocean wave, the bottle centered and fully visible, lightning behind translucent blue water, crisp reflections, dramatic dark background" \
  --steps 4 --seed 999
```

## SD 1.5

Applies to the base model and SD1.5 fine-tunes such as DreamShaper and Realistic
Vision. Use compact comma-separated concepts, repeat the most important subject
early, and provide a real negative prompt. Keep the canvas near the model's
native 512-pixel scale unless the selected profile advertises otherwise.

```bash
mold run dreamshaper-v8:fp16 \
  "fantasy castle on floating islands above clouds, waterfalls, sunrise, wide establishing shot, luminous atmosphere, detailed concept art" \
  --negative-prompt "blurry, low detail, text, watermark, malformed architecture" \
  --seed 555
```

For ControlNet, describe the intended rendered subject; the control image owns
pose, edges, or depth, not appearance.

## SDXL

Applies to SDXL Base, Turbo, Playground, Juggernaut, RealVis, DreamShaper XL,
Pony, and other SDXL fine-tunes. Write a concise subject/scene sentence followed
by photographic or illustrative treatment. Turbo needs one simple idea at four
steps. Base and quality fine-tunes tolerate richer composition. Fine-tunes may
publish trigger words; preserve them exactly.

```bash
mold run sdxl-turbo:fp16 \
  "Vibrant Bangkok street-food market at night, steam rising from woks, neon reflected on wet pavement, bustling documentary photograph" \
  --negative-prompt "empty street, blur, text, watermark" --steps 4 --seed 88
```

## SD 3.5

SD 3.5 follows complete natural-language composition better than legacy Stable
Diffusion. State count, placement, relationships, and lighting explicitly. Use
the negative prompt for visible defects rather than restating the positive.

```bash
mold run sd3.5-large:q8 \
  "A Victorian clocktower fills the left third of a city square at sunset; its glass walls reveal interlocking brass gears while pedestrians cross below, low-angle architectural photograph, dramatic clouds" \
  --negative-prompt "illegible clock, warped buildings, text, watermark" --seed 2024
```

## Z-Image

Z-Image Turbo works best with one detailed natural-language scene and explicit
spatial relationships. Its Qwen3 encoder can follow longer descriptions, but
the turbo denoiser still benefits from a single composition. Use quoted text
only when visible lettering is central, and expect seed variation.

```bash
mold run z-image-turbo:q8 \
  "An astronaut floats through a bioluminescent underwater cave, helmet visor reflecting blue coral below and a shaft of sunlight above, wide science-fiction illustration, crisp silhouette" \
  --steps 9 --seed 777
```

## Wuerstchen v2

Use broad subjects, color, lighting, and medium. The highly compressed cascade
is less reliable for tiny objects, anatomy, readable text, or dense geometry.

```bash
mold run wuerstchen-v2:fp16 \
  "A lighthouse on a rocky coast during a dramatic sunset, bold oil painting, vibrant orange and purple sky, crashing surf" \
  --negative-prompt "fine text, watermark, muddy colors" --seed 42
```

## Qwen-Image

Applies to base, 2512, Flash, Distill, and Lightning image variants. Describe
the scene as a structured art-direction brief: subject, location, composition,
lighting, materials, and finish. Put required visible text in exact quotes and
describe its placement. Base/2512 at 50 steps are the quality reference; few-step
distills need simpler prompts and their fixed guidance/step recipe.

```bash
mold run qwen-image-2512:q8 \
  'Straight-on editorial architectural photograph of a tiny artisan bakery named "MOLD & FLOUR" on a quiet European corner, deep teal facade, three arched windows, striped awning, vintage delivery bicycle, sunny spring morning, crisp realistic detail, balanced symmetrical composition' \
  --seed 251201
```

### Qwen-Image Edit

Use an imperative edit instruction. Name what must change and what must remain
identical. Do not use img2img `--strength`; repeat `--image` for all visual
references, ordered by role.

```bash
mold run qwen-image-edit-2511:q8 \
  "Change the sunny spring morning into a cozy snowy winter blue hour. Add fresh snow to the exterior and turn on warm interior lights. Preserve the exact teal facade, readable MOLD & FLOUR sign, windows, door, bread displays, bicycle, crates, camera position, geometry, and composition." \
  --image bakery.png --seed 251111
```

## LTX-Video 0.9.x

Describe a single short shot chronologically: subject motion, environmental
motion, then one camera move. Use visible actions rather than abstract mood.
Legacy LTX-Video renders silent video, so do not promise generated speech or
sound.

```bash
mold run ltx-video-0.9.6-distilled:bf16 \
  "Northern lights ripple from left to right over a frozen lake; green and violet ribbons reflect in the ice while the camera slowly tilts upward, one continuous time-lapse shot" \
  --frames 33 --seed 1234
```

## LTX-2, LTX-2.3, and LTX-2.5

Write one flowing paragraph under 200 words with 4-8 concrete present-tense
sentences: shot scale, subject, chronological action, one camera move, coherent
lighting, ambience, and sound. For dialogue, keep one visible speaker in one
continuous take, describe voice/accent/delivery, and put the complete short line
in quotation marks. Keep the mouth unobstructed. Avoid critical generated text,
rapid cuts, and more speech than fits the clip.

```bash
mold run ltx-2.5-22b-distilled:q6 \
  "A cinematic close-up shows a woman in a red raincoat beneath a glass awning during gentle rain at night. Cyan and magenta reflections ripple behind her. The camera remains locked off as she looks into the lens and says with delighted, clear delivery, 'This was made locally with Mold, including the sound.' She gives a quick laugh after the line. Rain taps on glass beneath her clean voice; no music and no on-screen text." \
  --width 768 --height 512 --frames 121 --clip-frames 121 --fps 24 \
  --audio --seed 83007
```

For lip sync, keep the request inside one clip; raising `--frames` above the
routing default without a matching `--clip-frames` can auto-chain it. Respect
the selected checkpoint's advertised dimension alignment. Distilled LTX-2.5
recipes fix guidance at 1.0; do not override it.

Official guidance: [LTX prompting guide](https://docs.ltx.io/open-source-model/usage-guides/prompting-guide), [LTX-2 repository](https://github.com/Lightricks/LTX-2).

## Wan 2.1 and 2.2

For T2V, use roughly 60-200 words when expanding a final prompt: subject and
setting, one temporal action in 2-3 beats, subtle background motion, lighting,
shot size/angle, and one camera move. For I2V/TI2V, the source already owns
appearance: stay under about 100 words and emphasize only motion and camera.
Avoid unrelated actions or conflicting camera directions.

```bash
mold run wan22-t2v-a14b:q4 \
  "Medium-wide wildlife shot at sunrise. A red fox trots steadily through fresh snow in a quiet pine forest; visible breath drifts back and powder lifts from each paw. The camera tracks alongside at eye level in one continuous move. Soft side light, realistic fur, stable background, no text." \
  --width 832 --height 480 --frames 49 --fps 16 --seed 83005
```

Wan A14B and TI2V-5B generate silent video. Do not ask them for native dialogue
or claim synchronized audio. Wan S2V is a separate speech-to-video model and is
not interchangeable with these checkpoints.

Official guidance: [Wan 2.2 repository](https://github.com/Wan-Video/Wan2.2), [Wan prompt-extension system](https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/system_prompt.py).

## MiniMax H3

H3 generates synchronized stereo audio and video. Use the official three
sections in this exact order. Put dialogue in the visual timeline, assign a
stable speaker ID such as `(S1)`, describe voice traits outside the dialogue
tag, and place only the language tag plus verbatim words inside `<d>...</d>`.
Do not repeat dialogue in the soundscape. Use `N/A` when no score is wanted.
Write this structure yourself: generic `mold expand` does not yet produce the
H3 Context-IR shape ([#1434](https://github.com/utensils/mold/issues/1434)).

Current runtime limitation: mold's H3 tokenizer does not yet register the
official dialogue and lyrics special tokens. Keep the upstream `<d>...</d>`
syntax so prompts remain forward-compatible, but do not promise reliable
speech, lip sync, or literal handling of those delimiters until
[#1430](https://github.com/utensils/mold/issues/1430) is fixed.

For a first-frame FL2VA request, anchor `<Picture 1>` at 0.00 seconds and
preserve its identity, clothing, layout, lighting, and composition. Describe a
continuous observable path forward; do not contradict the supplied frame.

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the presenter shown in <Picture 1> retains the exact face, hairstyle, clothing, workstation layout, lighting, and framing. The camera trucks right with small amplitude at slow speed as the presenter turns toward the lens. The presenter with a confident, bright voice and relaxed pace (S1) says: <d>[English] With Mold, your ideas render right here.</d> Their lips synchronize precisely; they gesture once toward the screen, then hold a natural final expression.

overall_soundscape: Low workstation airflow continues beneath clean close-miked speech. One quiet interface chime sounds after the final word.

non_diegetic_music: N/A
```

```bash
mold run minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p \
  "$(cat h3-prompt.txt)" --first-frame presenter.png --duration 5 --seed 83009
```

For Ref2VA, name ordered references as `<Picture n>`, `<Video n>`, and
`<Audio n>` in their supplied order. A five-second clip should normally remain
one continuous shot. Keep speech short enough to finish before the final pose.
Until [#1433](https://github.com/utensils/mold/issues/1433) is fixed, a reference
video containing a soundtrack is automatically treated as an audio reference
and shifts later `<Audio n>` ordinals. Strip unwanted audio before upload.

Official guidance: [MiniMax H3 prompting skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md), [base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).

## Upscalers

Real-ESRGAN upscalers take no text prompt. Choose the general model for photos
and mixed artwork, or an anime model for line art. Judge fine texture, halos,
and facial detail against the original at 100% zoom.

```bash
mold upscale input.png --model real-esrgan-x4plus:fp16 --output output-4x.png
```
