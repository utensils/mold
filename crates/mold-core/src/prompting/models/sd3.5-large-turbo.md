# SD 3.5 Large Turbo prompting

## Prompt style

One idea, one or two sentences, well under {{word_limit}} words. This
checkpoint is distilled to four steps, so a dense multi-clause brief resolves
worse than a single subject with its setting and light.

## Syntax

It runs at guidance 1.0, so the negative prompt is ignored. Exclude things by
describing what belongs in the frame instead. Quoted on-image text still works.

## Pitfalls

Crowded scenes and fine texture degrade against SD3.5 Large. Raising the step
count does not buy back that fidelity.

## CLI

```bash
mold run sd3.5-large-turbo:q8 "A lone sailboat on a mirror-flat lake at dawn, wide landscape photograph" --steps 4 --seed 2026
```

## Sources

- https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo
