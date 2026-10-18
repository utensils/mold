- **`mold skill install claude-desktop`.** Builds the mold skill as a ZIP for
  Claude Desktop and claude.ai (`~/Downloads/mold-skill.zip`). It adds a local
  `mold mcp` connector to `claude_desktop_config.json`, backing up the old
  file, using the default host or the one given with `--host`. It then opens
  **Customize > Skills** so you can upload the ZIP. The Desktop skill tells
  Claude to work through the connector tools, because Desktop chat runs in a
  cloud sandbox with no `mold` binary. `mold skill list` shows the connector
  and whether the ZIP is current, and
  `mold skill uninstall claude-desktop` removes only what mold wrote.
