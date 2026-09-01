# Qwen-Image-Edit Lightning prompting

## Prompt style

A four-step fused Lightning distill of Edit-2511. Keep the edit simple: one
change plus the invariants, well under {{word_limit}} words. Chained or
multi-part instructions resolve poorly in four steps.

## Syntax

It runs at guidance 1.0, so the negative prompt is ignored. Quoted text edits
and ordinal image addressing behave as on the base edit checkpoint.

## Pitfalls

Fine texture and small lettering degrade against Edit-2511. Prefer the base
checkpoint when the edit has to survive close inspection.

## CLI

```bash
mold run qwen-image-edit-lightning:fp8 'Replace the sign text with "CLOSED", keeping the font, colour, and placement.' --image shop.png --seed 251112
```

## Sources

- https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning
- https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/src/examples/tools/prompt_utils.py
