# Playground v2.5 prompting

## Prompt style

An aesthetic SDXL fine-tune tuned for photographic and painterly quality. It
likes short natural prompts: one subject, one setting, one lighting note. Extra
tags and quality words do not add polish, because the aesthetic is baked in.

## Syntax

An ordinary SDXL negative prompt applies. There are no trigger words.

## Pitfalls

It runs the EDM DPM++ 2M scheduler at 50 steps and guidance 3.0. Pushing
guidance toward SDXL's 7.5 oversaturates and hardens the image.

## CLI

```bash
mold run playground-v2.5:fp16 "a quiet harbour at first light, fishing boats at anchor" \
  --steps 50 --guidance 3.0 --seed 250

mold run playground-v2.5:fp16 "portrait of a ceramicist in her studio, soft north light" \
  --negative-prompt "blurry, text, watermark"
```

## Sources

- https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic
