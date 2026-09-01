# FLUX.1 Schnell prompting

## Prompt style

A four-step distilled FLUX.1 for drafts and thumbnails. Give it one subject,
one action, one setting, and one style word. Guidance is fixed at 0.

## Syntax

A negative prompt does nothing: Schnell has no negative branch, and true CFG
needs an identity reference. Quoted lettering is unreliable at four steps.

## Pitfalls

Piling on detail returns the average of it rather than all of it. Move to Dev
or Krea when materials, small text, or exact spatial relationships matter.

## CLI

```bash
mold run flux-schnell:q8 "a red fox asleep on a mossy log, soft morning light" --steps 4
mold run flux-schnell:bf16 "a lighthouse in a storm, dramatic illustration" --steps 4 --seed 12
```

## Sources

- https://huggingface.co/black-forest-labs/FLUX.1-schnell
- https://docs.bfl.ai/guides/prompting_unified_basics
