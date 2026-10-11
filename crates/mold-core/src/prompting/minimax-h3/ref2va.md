# MiniMax H3 Ref2VA

## Prompt style

Ref2VA replaces the three core fields with six, in order:
`subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`. Establish
the style in one sentence before `[Shot 1]`. State which traits each reference
owns, and never let one reference silently replace another's identity, motion,
or sound role. Define `<Subject n>` from source `<Picture n>` or `<Video n>` traits; subjects
and assets need not map one-to-one. Reserve standalone picture entries for frame
anchors. Prefix `summary` with its task, such as `[reference generation]`;
motion guidance alone is not video editing.

## Syntax

Label references `<Picture n>`, `<Video n>`, and `<Audio n>` in the order
supplied, counting each category independently. A reference video carrying a
soundtrack also takes the next `<Audio n>` label first, so strip unwanted
audio before upload. Retention markers are `fully_preserved`,
`partially_preserved`, `attribute_transfer`, and `weak_reference`; audio uses
`fully_copy`, `partially_copy`, `reference`, and `weak_reference`.

## Generation context

A five-second clip normally stays one continuous shot. Keep speech short
enough to finish before the final pose. Describe referenced subjects inside shots. Timbre reference does not copy words;
supply intended dialogue and reuse the speaker ID in subject and audio definitions.

## CLI

With a configured API key and advertised upload support, references use
request-bound upload sessions. Otherwise the CLI sends inline media, limited to
32 MiB decoded reference bytes combined across all batch siblings. This works on hosts with authentication disabled;
auth-enabled hosts still require a valid `MOLD_API_KEY`. A rejected upload is
never retried anonymously. License acceptance belongs to the rendering server;
H3 has no separate client acceptance gate. The supplied order is part of the
render. Uploaded references require one request per session (`--batch 1`).

```bash
# Ordered reference set: images, then video, then audio
mold run minimax-h3-ref2va:comfy-pruned-int8 "$(cat h3-ref-prompt.txt)" --reference image=a.png --reference video=b.mp4 --reference audio=c.wav
```

## Sources

- https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing/references/ref-en.txt
