# Z-Image prompting

Manifest family: `z-image`.

Use one detailed natural-language scene with explicit spatial relationships.
The Qwen3 encoder can follow longer descriptions, but the turbo denoiser still
benefits from a single composition. Use quoted text only when visible lettering
is central, and expect seed variation.

```bash
mold run z-image-turbo:q8 \
  "An astronaut floats through a bioluminescent underwater cave, visor reflecting blue coral below and a shaft of sunlight above, wide science-fiction illustration" \
  --steps 9 --seed 777
```
