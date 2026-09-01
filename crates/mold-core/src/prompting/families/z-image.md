# Z-Image prompting

Manifest family: `z-image`.

## Prompt style

The text encoder is a Qwen3 language model, so write grammatical sentences and
not a tag list. Long, detailed prompts are what the model was tuned for: one
continuous scene description built from subject, action, spatial arrangement,
setting, light, and finally medium or style. Keep it to one composition and
stay under {{word_limit}} words. Write in English or Chinese; both are read
natively and neither needs translating.

## Syntax

No weighting syntax. Text rendering is a headline strength in both Chinese and
English, so put visible lettering in quotation marks and name its position,
script, and style. The negative prompt is honoured only on the undistilled base
checkpoint, which runs guidance between 3 and 5. There is no reference-image
addressing to write.

## Generation context

1024x1024 is the reference canvas and any aspect ratio between roughly 512 and
2048 on a side is supported, so state the framing that suits the chosen shape.
Turbo runs nine steps at guidance 0. Because the sampler is short, one clear
composition resolves better than several competing ones on the same canvas.

## Examples

Input: astronaut in an underwater cave

Output: An astronaut floats through a bioluminescent underwater cave. The visor
reflects blue coral below while a shaft of sunlight falls from the opening
above. Wide science-fiction illustration with a crisp silhouette.

Input: a tea shop sign in Chinese and English

Output: A narrow tea shop at dusk on a wet city street. The lantern-lit sign
above the door reads "云间茶室" in gold brush script with "CLOUD ROOM TEA" in
small white capitals beneath it. Warm interior light spills onto the pavement,
realistic photography.

## Pitfalls

- Turbo is guidance-distilled and runs at guidance 0, so a negative prompt has
  no effect at all. Remove unwanted content by describing what belongs in the
  frame instead.
- Comma-separated tag salad underperforms an equivalent sentence.
- Very long prompts can exceed the encoder sequence limit and lose the tail.
- Repeating the same prompt and seed produces near-identical output, so vary
  the wording rather than only the seed (community observation).

## CLI

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

## Sources

- https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8
- https://github.com/Tongyi-MAI/Z-Image
- Community: seed-versus-prompt variation is reported in the discussion thread.
