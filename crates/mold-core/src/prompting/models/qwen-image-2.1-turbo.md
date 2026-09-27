# Qwen Image 2.1 turbo prompting

Covers `qwen-image-2.1-turbo`: the base weights plus Viggle's 6-step distilled
adapter.

## Prompt style

Write the same direct, complete description as for the base model, under
{{word_limit}} words. The student was distilled against prompt-enhanced
targets, so describing subject, setting, composition and lighting helps.

## Syntax

Guidance is fixed at 1.0, so there is no negative prompt. Quoted text,
ordinal references ("image 1") and transparency behave as on the base model.

## Pitfalls

Small or long lettering garbles more often than with the 40-step base; keep
it short and large. Complicated edits (several references, face swaps,
identity-preserving changes) can ghost subjects; ask for one clear change.

## CLI

```bash
mold run qwen-image-2.1-turbo "A studio portrait of an old fisherman mending a net, warm rim light, 85mm" --seed 0
mold run qwen-image-2.1-turbo:int8-conv "Replace the background with a sunset beach, keep the subject unchanged" --image portrait.png
```

## Sources

- https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo
- https://huggingface.co/Qwen/Qwen-Image-2.1
