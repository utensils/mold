# SDXL prompting

Manifest family: `sdxl`.

Write a concise subject and scene sentence followed by photographic or
illustrative treatment. Turbo needs one simple idea at four steps. Base and
quality fine-tunes tolerate richer composition; preserve published trigger
words exactly.

```bash
mold run sdxl-turbo:fp16 \
  "Vibrant Bangkok street-food market at night, steam rising from woks, neon reflected on wet pavement, bustling documentary photograph" \
  --negative-prompt "empty street, blur, text, watermark" --steps 4 --seed 88
```

With `--id-image`, let the reference own facial features and describe the new
scene. Start near `--id-weight 0.8`. SDXL uses ordinary CFG and must not receive
FLUX's `--true-cfg`; check `supports_identity` for the exact model first.
