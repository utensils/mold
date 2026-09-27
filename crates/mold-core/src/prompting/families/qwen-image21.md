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

One model covers text-to-image, editing and multi-reference composition: up
to ten ordered reference images ride the request, and none of them is a
special "source". With references attached, write the change or the new
scene and refer to the pictures by position ("the woman in image 1", "the
jacket from image 2"). An edit instruction can be short, such as "Change the
background to a sunset beach". Canvases are multiples of 32; 1024x1024 is the
default and the native 2K sizes are 2048x2048, 2400x1792, 2528x1696 and
2752x1536 (and their portrait turns). The standard recipe uses forty steps.

For a transparent background, describe only the subject: no scenery,
backdrop or floor. Mold then wraps the prompt in the model card's RGBA
recipe, so never write these two sentences yourself:
"This is an RGBA image with transparency." before it and
"The image has alpha channel and the background is transparent." after it.

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
- References are addressed by their order, not by describing their pixels;
  never invent what an attached image shows.
- A transparent render wants an isolated subject. Scenery words ("in a
  forest", "on a table") fight the transparent background.
- There is no mask or ControlNet input; describe a local edit in words.
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
