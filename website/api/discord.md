# Discord Bot

mold includes a built-in Discord bot that connects to `mold serve`, allowing
users to generate images and videos via slash commands.

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

| Command              | Description                                                                                                           |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `/generate`          | Generate an image or video, including attachment-driven LTX-2 audio-to-video, retake, and keyframe modes              |
| `/sequence`          | Submit 2–16 `\|`-separated prompts as a durable LTX-2 sequence, with per-clip progress and final MP4 delivery         |
| `/expand`            | Expand a short prompt into detailed generation prompts                                                                |
| `/models`            | List available models with download/loaded status                                                                     |
| `/status`            | Show server health, queue summary, and every GPU/MIG device; large fleets paginate across limit-safe follow-up embeds |
| `/quota`             | Check your remaining daily generation quota                                                                           |
| `/admin reset-quota` | Reset a user's daily quota (requires Manage Server)                                                                   |
| `/admin block`       | Temporarily block a user from generating (requires Manage Server)                                                     |
| `/admin unblock`     | Unblock a previously blocked user (requires Manage Server)                                                            |

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

LTX-2 specialized modes are selected by their attachments: `audio_file` starts
audio-to-video, `source_video` plus both retake times regenerates that time
range, and two or three `keyframe_*` images are spaced across the requested
frame count for interpolation. These modes are mutually exclusive in one
command and do not need the `pipeline` option.
:::

::: tip Durable sequences
`/sequence` accepts prompts separated by `|` and queues them through the
server's durable chain-job API. A normal bot-authored channel message follows
clip progress and attaches the completed MP4 when it fits, so delivery does not
expire with the slash-command interaction token. Larger outputs receive a deep
link to the exact print in the server's Library instead of failing Discord
upload.
:::

::: warning `prompt` is now optional — re-register the commands
`/generate`'s `prompt` option changed from required to optional so an LTX-2
image-to-video run can be submitted with just a `source_image`. Discord caches
command definitions, so **the bot's slash commands must be re-registered** before
users see the new signature; until then Discord keeps enforcing the old required
`prompt`. The relaxation is guarded on both ends: the bot only skips the
up-front check when visual conditioning (`source_image`, `source_video`, or
keyframes) is attached, and the server's family-aware validator still rejects
an empty prompt for every image family and for unconditioned text-to-video.
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
