# SDXL Turbo prompting

## Prompt style

One simple idea in a short sentence: subject, setting, treatment. Turbo renders
in four steps at guidance 0, so extra clauses average out instead of resolving.
Its native canvas is 512x512.

## Syntax

Guidance 0 runs no classifier-free branch, so a negative prompt has no effect
here. Never pass true CFG. Turbo is the one SDXL checkpoint that identity
conditioning does not accept.

## Pitfalls

Raising steps and guidance to base-SDXL values degrades Turbo rather than
improving it. Long tag lists are wasted tokens.

## CLI

```bash
mold run sdxl-turbo:fp16 "a red canoe on a still lake at dawn" --steps 4 --seed 88
mold run sdxl-turbo:fp16 "neon ramen counter at night, documentary photograph" --width 512 --height 512
```

## Sources

- https://huggingface.co/stabilityai/sdxl-turbo
