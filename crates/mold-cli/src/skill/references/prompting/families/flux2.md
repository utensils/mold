# FLUX.2 prompting

Manifest family: `flux2`.

Applies to Klein 4B/9B, Dev, and compatible fine-tunes. Use direct prose with a
clear subject, scene geometry, action, and finish. Klein is a four-step model:
one visual idea is more reliable than many competing details. Dev can carry a
denser art-direction brief.

```bash
mold run flux2-klein-9b:q4 \
  "Macro photograph of a glass bottle ship caught inside a curling ocean wave, the bottle centered and fully visible, lightning behind translucent blue water, crisp reflections, dramatic dark background" \
  --steps 4 --seed 999
```
