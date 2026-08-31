# Qwen-Image Edit prompting

Manifest family: `qwen-image-edit`.

Use an imperative edit instruction. Name what must change and what must remain
identical. Do not use img2img `--strength`; repeat `--image` for all visual
references, ordered by role.

```bash
mold run qwen-image-edit-2511:q8 \
  "Change the sunny morning into a snowy winter blue hour. Preserve the exact teal facade, readable MOLD & FLOUR sign, windows, door, camera position, geometry, and composition." \
  --image bakery.png --seed 251111
```
