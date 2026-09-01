# Pony Diffusion V6 XL prompting

## Prompt style

Booru tags, comma separated. Open with the score prefix `score_9, score_8_up,
score_7_up, score_6_up, score_5_up, score_4_up`, then a source tag
(`source_anime`, `source_pony`, `source_furry`, `source_cartoon`), then a
rating tag (`rating_safe`, `rating_questionable`, `rating_explicit`), then
subject tags. CyberRealistic Pony uses the same prefix, photographic tags.

## Syntax

The author says it needs no negative prompt in most cases and warns off quality
words like masterpiece. Community practice adds `score_6, score_5, score_4`
there.

## Pitfalls

Without the score prefix, output looks washed out. mold has no clip skip
control.

## CLI

```bash
mold run pony-v6:fp16 \
  "score_9, score_8_up, score_7_up, source_anime, rating_safe, 1girl, red scarf, snowy street, city lights, night" \
  --negative-prompt "score_6, score_5, score_4, blurry, watermark" --seed 606

mold run cyberrealistic-pony:fp16 \
  "score_9, score_8_up, score_7_up, rating_safe, portrait photograph of a woman, freckles, soft window light, 85mm" \
  --negative-prompt "score_6, score_5, score_4, cartoon, text, watermark"
```

## Sources

- https://civitai.com/models/257749/pony-diffusion-v6-xl
- https://huggingface.co/AstraliteHeart/pony-diffusion-v6
