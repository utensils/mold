# Discord Bot

mold includes a built-in Discord bot that connects to `mold serve`, allowing
users to generate images via slash commands.

## Running

```bash
# Server + bot in one process
MOLD_DISCORD_TOKEN="your-token" mold serve --discord

# Or run the bot separately (connects to a remote server)
MOLD_HOST=http://gpu-host:7680 MOLD_DISCORD_TOKEN="your-token" mold discord
```

## Setup

1. Create a Discord application at the
   [Developer Portal](https://discord.com/developers/applications)
2. Create a bot user and copy the token
3. Invite with:
   `https://discord.com/api/oauth2/authorize?client_id=YOUR_APP_ID&permissions=51200&scope=bot%20applications.commands`
   (Send Messages, Attach Files, Embed Links + slash command registration)
4. No privileged intents are needed (slash commands only)

## Slash Commands

| Command              | Description                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------- |
| `/generate`          | Generate an image or video (prompt, model, source_image, video_format, frames, fps, width, height, steps, guidance, seed, strength, audio, pipeline, negative_prompt) |
| `/expand`            | Expand a short prompt into detailed generation prompts                                   |
| `/models`            | List available models with download/loaded status                                        |
| `/status`            | Show server health, GPU info, uptime                                                     |
| `/quota`             | Check your remaining daily generation quota                                              |
| `/admin reset-quota` | Reset a user's daily quota (requires Manage Server)                                      |
| `/admin block`       | Temporarily block a user from generating (requires Manage Server)                        |
| `/admin unblock`     | Unblock a previously blocked user (requires Manage Server)                               |

## Configuration

| Variable                     | Default                 | Description                                                             |
| ---------------------------- | ----------------------- | ----------------------------------------------------------------------- |
| `MOLD_DISCORD_TOKEN`         | —                       | Bot token (required; falls back to `DISCORD_TOKEN`)                     |
| `MOLD_HOST`                  | `http://localhost:7680` | mold server URL                                                         |
| `MOLD_DISCORD_COOLDOWN`      | `10`                    | Per-user cooldown (s)                                                   |
| `MOLD_DISCORD_ALLOWED_ROLES` | —                       | Comma-separated role names/IDs for access control (unset = all)         |
| `MOLD_DISCORD_DAILY_QUOTA`   | —                       | Max generations per user per UTC day (unset = unlimited; 0 = block all) |

::: tip Video generation
Running `/generate` against a video model (`ltx-video-*`, `ltx-2-*`) produces an
MP4 by default. Pass `video_format: Animated GIF` to receive a GIF instead. You
can also attach a `source_image` for img2img on regular models, or as the first
frame for LTX-2 image-to-video. When the rendered MP4 exceeds Discord's upload
ceiling the bot falls back to the always-bundled GIF preview.
:::

::: info Block List
The `/admin block` command stores blocks in memory. Blocks clear when the bot
restarts. For permanent restrictions, use role-based access via
`MOLD_DISCORD_ALLOWED_ROLES`.
:::

## NixOS

```nix
services.mold.discord = {
  enable = true;
  # tokenFile is loaded via systemd EnvironmentFile —
  # the file must contain: MOLD_DISCORD_TOKEN=your-token-here
  tokenFile = config.age.secrets.discord-token.path;
  moldHost = "http://localhost:7680";
  cooldownSeconds = 10;
};
```
