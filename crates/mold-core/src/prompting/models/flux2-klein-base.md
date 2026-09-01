# FLUX.2 Klein Base prompting

## Prompt style

The undistilled 4B and 9B Klein weights, at 50 steps and guidance 4.0, so they
carry a far denser brief than four-step Klein. Keep FLUX.2's Subject + Action +
Style + Context order.

## Syntax

The only FLUX.2 tier where a negative prompt works: above guidance 1.0 it runs
a real unconditional branch. `--guidance 1` skips it and the negative prompt
goes inert.

## Pitfalls

Guidance above 1.0 costs two forward passes per step. Reusing a four-step Klein
recipe wastes the model.

## CLI

```bash
mold run flux2-klein-base:q8 "a brass orrery on a walnut desk, low winter sun through tall windows" \
  --steps 50 --guidance 4.0 --negative-prompt "blurry, warped rings, text, watermark"

mold run flux2-klein-base-9b:q6 "an empty greenhouse at dawn, condensation on the glass, terracotta pots" \
  --steps 50 --guidance 4.0
```

## Sources

- https://docs.bfl.ai/guides/prompting_guide_flux2
- https://huggingface.co/black-forest-labs/FLUX.2-klein-base
