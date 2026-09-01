# MiniMax H3 base modes

## Prompt style

T2VA builds the whole timeline from text. I2VA, FL2VA, and L2VA prepend one
instruction line, a blank line, then the three core fields:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

Preserve the frame's identity, clothing, layout, lighting, and composition,
then describe one continuous observable path forward. Never contradict it.

## Examples

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, the presenter shown in <Picture 1> retains the exact face, clothing, workstation layout, lighting, and framing. The camera trucks right at slow speed as the presenter turns toward the lens. The presenter with a bright voice (S1) says: <d>[English] With Mold, your ideas render right here.</d> Their lips synchronize; they gesture once.

overall_soundscape: Low workstation airflow continues beneath clean close-miked speech.

non_diegetic_music: N/A
```

## CLI

```bash
# First-frame conditioning; the prompt file holds the instruction line plus the three fields
mold run minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p "$(cat h3-prompt.txt)" --first-frame presenter.png --duration 5 --seed 83009
```

## Sources

- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt
