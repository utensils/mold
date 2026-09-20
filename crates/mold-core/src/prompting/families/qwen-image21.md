# Qwen-Image 2.1 prompting

Manifest family: `qwen-image21`.

## Prompt style

Write one direct, complete description of the image. State the primary subject,
setting, action or pose, composition, lighting, and visual medium in a natural
order. Preserve the user's requested subject and style; add only details that
make an underspecified composition legible. Keep the result under
{{word_limit}} words.

## Syntax

Put requested on-image text in quotation marks and say where it belongs and how
it should look. Keep the quoted string exact. Describe positive visual content
in the prompt. A separate negative prompt is used only when the caller
explicitly enables classifier-free guidance with a value greater than one.

## Generation context

Mold currently exposes this checkpoint as text-to-image only. Use a canvas
whose width and height are multiples of 32; the native 1024x1024 recipe is the
default. The model's standard quality recipe uses forty denoising steps and
guidance 1, so no negative prompt is needed for normal generation.

## Examples

Input: a poster for a small bakery called MOLD & FLOUR

Output: Straight-on editorial photograph of a tiny artisan bakery on a quiet
European corner, deep teal facade with three arched windows and a striped
awning. A hand-painted sign above the door reads "MOLD & FLOUR" in cream serif
capitals. A vintage delivery bicycle rests at the right edge. Sunny spring
morning, crisp realistic detail, balanced symmetrical composition.

Input: a lion statue

Output: Dynamic stone lion sculpture mid-pounce in an ancient courtyard, front
legs airborne and hind legs pushing off, weathered carved surface and defined
muscles. Trees and shallow stone steps in soft morning haze, documentary
photography, low eye-level composition, finely detailed natural light.

## Pitfalls

- A vague request for lettering does not specify the characters to render;
  write the exact string in quotation marks.
- Do not rely on image-reference, source-image, mask, ControlNet, or LoRA
  wording in a prompt: those inputs are not exposed for this first Mold
  integration.
- Very crowded compositions and many independent text blocks compete for the
  same canvas. Give the principal subject and the important lettering clear
  spatial priority.

## CLI

```bash
mold run qwen-image-2.1:bf16 \
  'Straight-on editorial photograph of a tiny artisan bakery named "MOLD & FLOUR" on a quiet European corner, deep teal facade, three arched windows, striped awning, sunny spring morning, crisp realistic detail, balanced composition' \
  --seed 210001
```

## Sources

- https://huggingface.co/Qwen/Qwen-Image-2.1
- https://github.com/QwenLM/Qwen-Image
