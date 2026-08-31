# LTX-Video 0.9.x prompting

Manifest family: `ltx-video`.

Describe one short shot chronologically: subject motion, environmental motion,
then one camera move. Use visible actions rather than abstract mood. Legacy
LTX-Video renders silent video, so do not promise generated speech or sound.

```bash
mold run ltx-video-0.9.6-distilled:bf16 \
  "Northern lights ripple left to right over a frozen lake; green and violet ribbons reflect in the ice while the camera slowly tilts upward, one continuous time-lapse shot" \
  --frames 33 --seed 1234
```
