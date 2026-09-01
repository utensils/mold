# Shared prompting practice

## Prompt style

State the subject and action before style, lighting, lens, or material cues.
Keep one coherent visual or temporal idea. Preserve exact user wording unless
the user asks for expansion. When source media owns appearance or composition,
prompt only the requested change and name what must stay unchanged.

## Pitfalls

Hold one seed while refining a prompt, then vary the seed for final art. Never
promise sound, speech, or motion the selected family does not generate.

## CLI

Read this file for every generation or upscale request, then read exactly one
family base guide linked from `SKILL.md`. Read a task leaf only when that task
needs its own prompt grammar.

Confirm the selected identity with `mold info <model>` or the remote
`/api/models` row before choosing dimensions, frames, steps, guidance, or
conditioning. Installed catalog checkpoints can differ from manifest defaults.

```bash
# Confirm an identity and its advertised defaults before writing the prompt
mold info flux2-klein:q8
```
