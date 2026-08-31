# Wuerstchen v2 prompting

Manifest family: `wuerstchen`.

Use broad subjects, color, lighting, and medium. The compressed cascade is less
reliable for tiny objects, anatomy, readable text, or dense geometry.

```bash
mold run wuerstchen-v2:fp16 \
  "A lighthouse on a rocky coast during a dramatic sunset, bold oil painting, vibrant orange and purple sky, crashing surf" \
  --negative-prompt "fine text, watermark, muddy colors" --seed 42
```
