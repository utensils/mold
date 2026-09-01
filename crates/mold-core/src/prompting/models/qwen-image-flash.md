# Qwen-Image few-step distills prompting

Covers `qwen-image-flash`, `qwen-image-distill`, and `qwen-image-lightning`.

## Prompt style

Simplify. Keep the official rewrite rules but drop the densest art direction
and stay well under {{word_limit}} words. One subject, one setting, one style.

## Syntax

All three run CFG-free at guidance 1.0, so the negative prompt is ignored.
Quoted on-image text still works but keep it short.

## Pitfalls

Fixed step recipes: flash 4, distill 15, lightning 4 or 8 by tag. Dense small
text, hair-fine detail, and very complex scenes degrade against base.

## CLI

```bash
# NVIDIA DMD2 four-step distill of base
mold run qwen-image-flash:q8 "A red enamel teapot on a sunlit windowsill, still-life photograph" --steps 4 --seed 251203

# DiffSynth 15-step distill: closer to base fidelity
mold run qwen-image-distill:q8 "A quiet harbour at dawn, fishing boats at anchor, realistic photography" --steps 15 --seed 251204

# Lightning, eight-step tag
mold run qwen-image-lightning:fp8-8step "A snowy mountain village at blue hour, warm window lights" --steps 8 --seed 251205
```

## Sources

- https://raw.githubusercontent.com/QwenLM/Qwen-Image/main/src/examples/tools/prompt_utils.py
- https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning
