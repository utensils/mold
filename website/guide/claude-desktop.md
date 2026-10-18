# Claude Desktop

Claude Desktop (and claude.ai) can use mold through two pieces that one command
sets up:

- **The mold skill**: instructions and prompting guides Claude loads when you
  ask for images or video. Desktop keeps skills on your Claude account, so the
  skill is uploaded as a ZIP and then follows you to the web and mobile apps.
- **The mold connector**: a local MCP server (`mold mcp`) that Desktop starts on
  your computer. Chat skills run in Anthropic's cloud sandbox, which has no
  `mold` binary and cannot reach your machines, so the connector is how Claude
  actually lists, generates and fetches prints.

## Install

Pair your GPU host first if it is not this computer (see
[Remote Workflows](/guide/remote-workflows)), then run:

```bash
mold skill install claude-desktop
```

It does three things:

1. Writes `~/Downloads/mold-skill.zip`, the skill folder zipped as Claude
   expects (`mold/SKILL.md` at the top level).
2. Adds `mcpServers.mold` to Claude Desktop's `claude_desktop_config.json`
   (`~/Library/Application Support/Claude/` on macOS, `%APPDATA%\Claude\` on
   Windows). The previous file is kept as
   `claude_desktop_config.json.mold-backup`. The connector runs this `mold`
   binary as `mold mcp` and talks to your default host (`mold host use NAME`);
   when `MOLD_HOME` is set, it is passed through, since Desktop does not read
   your shell environment.
3. Opens [Customize > Skills](https://claude.ai/customize/skills) and shows the
   ZIP in your file manager.

Then finish in Claude Desktop:

1. Quit and reopen Claude Desktop so it starts the connector.
2. **Customize > Skills > Add**, upload `mold-skill.zip`, and turn it on.
   Remove or replace an older `mold` skill there first.
3. Make sure **Settings > Capabilities > Code execution and file creation** is
   on; Claude needs it to use any skill.

Ask something like "what's on my mold host's queue?" or "show my latest
prints". Claude loads the skill and answers with the connector's tools
(`server_status`, `list_gallery`, `generate_image`, …).

There is no API for uploading a skill to a personal Claude account, which is
why the upload step is manual. Re-run the command after upgrading mold. It
replaces its own ZIP and connector entry. `mold skill list` shows whether the
connector is in place and whether the ZIP is current.

## Options

```bash
mold skill install claude-desktop --host workstation       # pin the connector to one saved host
mold skill install claude-desktop --output ./mold-skill.zip
mold skill install claude-desktop --no-connector           # only write the ZIP
mold skill install claude-desktop --no-open                # print the steps, open nothing
mold skill show claude-desktop                             # the SKILL.md that goes in the ZIP
mold skill uninstall claude-desktop
```

Mold never overwrites a file it did not write. Overwriting an `--output` path
that holds some other file needs `--force`. So does replacing an
`mcpServers.mold` entry that mold did not write, or one edited since mold wrote
it. Mold records the entry it wrote in
`claude_desktop_config.json.mold-connector`. `uninstall` removes only that
recorded entry and the ZIP mold put in Downloads. Remove the uploaded skill
yourself under Customize > Skills.

## Connector without the skill

To wire `mold mcp` into Desktop by hand instead, add this to
`claude_desktop_config.json` and restart Desktop:

```json
{
  "mcpServers": {
    "mold": {
      "command": "/path/to/mold",
      "args": ["mcp", "--host", "workstation"]
    }
  }
}
```

The connector never renders on this computer: `mold mcp` talks to a mold
server over HTTP and has no local-GPU fallback.
