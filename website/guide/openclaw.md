# OpenClaw

[OpenClaw](https://docs.openclaw.ai) is an agentic coding assistant and runtime
that can load workspace-specific skills. Mold's CLI-native interface makes it a
natural agent tool: commands remain inspectable and reproducible outside the
agent, while the same local or remote engine is also available through MCP and
REST/SSE.

## How It Fits Together

The usual setup is:

1. Run `mold serve` on the machine with the GPU.
2. Configure `MOLD_HOST` wherever OpenClaw is running.
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
project installs and `~/.openclaw/skills/mold/SKILL.md` user-wide.

## Configure the Connection

Point OpenClaw at a running mold server:

```bash
export MOLD_HOST=http://gpu-host:7680
```

Then normal mold commands work from the OpenClaw environment:

```bash
mold list
mold run flux-schnell:q8 "a cinematic product shot"
```

## Notes

- If the server is remote, `HF_TOKEN` must be set on the server side for gated
  model pulls.
- `mold ps` is the quickest way to confirm the OpenClaw environment can reach
  the server.
- The same setup works well for agent workflows that want prompt expansion,
  gallery generation, or remote batch jobs.
