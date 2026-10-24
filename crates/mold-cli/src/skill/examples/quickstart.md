# Quickstart examples

These examples are parsed by Mold's CLI contract tests. Replace prompts and
paths, but keep the command shape.

```bash
mold list
mold info flux2-klein:q8
mold run flux2-klein:q8 "A red fox in falling snow" --seed 42 --output fox.png
mold run qwen-image-edit-2511:q8 "Change the chair to red leather" --image chair.png --output edited.png
mold upscale input.png --model real-esrgan-x4plus:fp16 --output output-4x.png
mold queue list
mold queue cancel job-abc123
mold server status
mold host list
mold ps --host workstation
mold run flux2-klein:q8 "A red fox in falling snow" --host workstation
mold library download --latest --host workstation
mold skill show codex
mold skill show codex references/prompting/families/flux2.md
```

For a remote host, pick a saved one from `mold host list` and pass
`--host NAME`; its saved credential is used, so nothing secret goes on the
command line. Run `mold <command> --help` for anything not covered here.

For a video up to one generation long (MiniMax H3: 14.4 s), use one prompt
and seconds. Longer than that, write a `scenes.toml` with one `[[scene]]`
(`seconds` and its own `prompt`) per part of at most one generation, as in
[Longer videos](../references/cli.md#longer-videos):

```bash
mold list --json --host workstation
mold --host workstation run MODEL "a paper boat drifts along a sunlit stream" --duration 12
mold --host workstation run MODEL --scenes scenes.toml --dry-run
mold --host workstation run MODEL --scenes scenes.toml
```

Keep the printed video UUID and inspect it before retrying after a disconnect.
For exact per-stage video scripts, copy the complete `shot.toml` from
[Scripted sequences](../references/cli.md#scripted-sequences), inspect the
selected model, then validate and run it:

```bash
mold list --json --host workstation
mold chain validate shot.toml
mold run --script shot.toml --dry-run --host workstation
mold run --script shot.toml --host workstation --output walk.mp4
mold jobs show job-abc123 --json --host workstation
```

Keep the printed job ID; inspect it before retrying a disconnected render.
