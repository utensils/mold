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
