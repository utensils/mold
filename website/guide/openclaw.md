# OpenClaw

[OpenClaw](https://docs.openclaw.ai) is an agentic coding assistant and runtime
that can load workspace-specific skills. Mold's CLI-native interface makes it a
natural agent tool: commands remain inspectable and reproducible outside the
agent, while the same local or remote engine is also available through MCP and
REST/SSE.

## How It Fits Together

The usual setup is:

1. Run `mold serve` on the machine with the GPU.
2. Pair the machine OpenClaw runs on with that host (`mold host add`).
3. Install the mold skill into the workspace so agents can call the CLI or API.

That keeps image generation on the GPU host while the rest of your agent work
happens from a laptop, devbox, or remote shell.

## Install the Skill

Install the embedded skill directly into OpenClaw's user-wide path:

```bash
mold skill install openclaw
```

For a project-local workspace, run `mold skill install openclaw --project` from
its root or pass `--dir <PATH>`. Mold writes `.agents/skills/mold/SKILL.md` for
project installs and `~/.openclaw/skills/mold/SKILL.md` user-wide. The
OpenClaw render includes its `{baseDir}` path convention and runtime metadata,
plus a concise router, operational safety guidance, tested examples, and one
shared prompting guide plus one base file under `references/prompting/families/`
for each manifest family. H3, Wan, and LTX-2 add only their necessary task
leaves. This keeps LTX-2.x and MiniMax H3 speech/audio grammar available without
loading every family's instructions at once.

## Configure the Connection

Pairing is done by the user, not the agent. In their own terminal, the user
runs `mold host invite` on the GPU host and `mold host add` on the OpenClaw
machine, then pastes the link at the prompt. An agent never pastes pairing links
or API keys; it only uses the saved host by name:

```bash
mold host list                      # saved hosts and the default
mold list --host workstation
mold run flux-schnell:q8 "a cinematic product shot" --host workstation
```

`mold host use workstation` makes it the default so `--host` can be omitted;
`MOLD_HOST=workstation` also works where an environment variable fits better.
If a command reports "No secure pairing for this host", ask the user to pair it.

## Notes

- If the server is remote, `HF_TOKEN` must be set on the server side for gated
  model pulls.
- `mold ps --host workstation` is the quickest way to confirm the OpenClaw
  environment can reach the server.
- The same setup works well for agent workflows that want prompt expansion,
  gallery generation, or remote batch jobs.
