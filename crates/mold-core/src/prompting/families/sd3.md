# SD 3.5 prompting

Manifest family: `sd3`.

Use complete natural-language composition. State count, placement,
relationships, and lighting explicitly. Use the negative prompt for visible
defects rather than restating the positive.

```bash
mold run sd3.5-large:q8 \
  "A Victorian clocktower fills the left third of a city square at sunset; its glass walls reveal interlocking brass gears while pedestrians cross below, low-angle architectural photograph" \
  --negative-prompt "illegible clock, warped buildings, text, watermark" --seed 2024
```
