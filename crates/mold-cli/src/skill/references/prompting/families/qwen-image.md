# Qwen-Image prompting

Manifest family: `qwen-image`.

Describe the scene as a structured art-direction brief: subject, location,
composition, lighting, materials, and finish. Put required visible text in exact
quotes and describe its placement. Base/2512 quality recipes tolerate more
detail; few-step distills need simpler prompts and their advertised fixed recipe.

```bash
mold run qwen-image-2512:q8 \
  'Straight-on editorial photograph of a tiny bakery named "MOLD & FLOUR", deep teal facade, three arched windows, sunny spring morning, crisp realistic detail, symmetrical composition' \
  --seed 251201
```
