# Wan text-to-video

Use for T2V identities. When expanding a final prompt, aim for roughly 60-200
words: subject and setting, one temporal action in 2-3 beats, subtle background
motion, lighting, shot size and angle, and one camera move. Avoid unrelated
actions or conflicting camera directions.

```bash
mold run wan22-t2v-a14b:q4 \
  "Medium-wide wildlife shot at sunrise. A red fox trots steadily through fresh snow in a quiet pine forest; visible breath drifts back and powder lifts from each paw. The camera tracks alongside at eye level in one continuous move. Soft side light, realistic fur, stable background, no text." \
  --width 832 --height 480 --frames 49 --fps 16 --seed 83005
```
