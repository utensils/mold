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

| Command              | Description                                                                                                                                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/generate`          | Generate an image or video, including attachment-driven LTX-2 audio-to-video, retake, and keyframe modes and ordered MiniMax H3 references                                                                                      |
| `/identity`          | Generate an image conditioned on a face reference photo (PuLID), with `identity_strength` and `identity_start_step`                                                                                                             |
| `/sequence`          | Submit 2–16 `\|`-separated prompts as a durable video sequence (LTX-2, LTX-Video (which joins independently rendered clips) or Wan, per the model's advertised sequence support), with per-clip progress and final MP4 delivery |
| `/expand`            | Expand a short prompt into detailed generation prompts                                                                                                                                                                          |
| `/models`            | List available models with download/loaded status                                                                                                                                                                               |
| `/status`            | Show server health, queue summary, and every GPU/MIG device; large fleets paginate across limit-safe follow-up embeds                                                                                                           |
| `/quota`             | Check your remaining daily generation quota                                                                                                                                                                                     |
| `/admin reset-quota` | Reset a user's daily quota (requires Manage Server)                                                                                                                                                                             |
| `/admin block`       | Temporarily block a user from generating (requires Manage Server)                                                                                                                                                               |
| `/admin unblock`     | Unblock a previously blocked user (requires Manage Server)                                                                                                                                                                      |

### `/identity`

Face-identity conditioning has its own command rather than options on
`/generate`. Discord caps a chat-input command at **25 options** and
`/generate` is already at exactly 25, and identity is qualified only for a
fixed list of FLUX and SDXL checkpoints (see the
[Identity Photos guide](/guide/identity)), so none of `/generate`'s video and
conditioning options apply to it anyway.

| Option                | Purpose                                                                                        |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| `prompt`              | Required. What to render.                                                                      |
| `identity`            | Required. Face reference photo as a PNG or JPEG attachment.                                    |
| `model`               | Identity-capable checkpoint. Autocompletes only models the server advertises; defaults to one. |
| `identity_strength`   | `0.0`–`3.0`, default `1.0`.                                                                    |
| `identity_start_step` | First identity-conditioned denoise step; must be under the resolved step count. Default `0`.   |
| `width` / `height`    | Output size in pixels.                                                                         |
| `steps`               | Inference steps.                                                                               |
| `guidance`            | Guidance scale.                                                                                |
| `seed`                | Seed for reproducibility.                                                                      |

Preconditions are checked in cost order, so an impossible request never takes a
quota slot or a download: the declared attachment size and container and the
strength range first, then the model gate against the server's advertised
`/api/models[].supports_identity` (an absent field is read as "no", which
covers both a server too old for identity conditioning and one whose binary
cannot execute it) then the start step against the resolved step count, and
finally the downloaded bytes. A server advertising no identity-capable model at
all says so instead of guessing a checkpoint. The result embed carries an
**Identity** row naming the photo, the strength, and the start step.

## Configuration

| Variable                     | Default                 | Description                                                             |
| ---------------------------- | ----------------------- | ----------------------------------------------------------------------- |
| `MOLD_DISCORD_TOKEN`         | --                      | Bot token (required; falls back to `DISCORD_TOKEN`)                     |
| `MOLD_HOST`                  | `http://localhost:7680` | mold server URL                                                         |
| `MOLD_DISCORD_COOLDOWN`      | `10`                    | Per-user cooldown (s)                                                   |
| `MOLD_DISCORD_ALLOWED_ROLES` | --                      | Comma-separated role names/IDs for access control (unset = all)         |
| `MOLD_DISCORD_DAILY_QUOTA`   | --                      | Max generations per user per UTC day (unset = unlimited; 0 = block all) |

::: tip Video generation
Running `/generate` against a video model (`ltx-video-*`, `ltx-2-*`, `wan*`)
produces an MP4 by default. Pass `video_format: Animated GIF` to receive a GIF instead. You
can also attach a `source_image` for img2img on regular models, or as the first
frame for LTX-2 image-to-video. When the rendered MP4 exceeds Discord's upload
ceiling the bot falls back to the always-bundled GIF preview.

Use `duration` for the simple path: `duration: 10` uses the selected model's
default FPS and converts ten seconds to the nearest frame count on the selected
model's advertised grid (`8n+1` for the LTX families, `4n+1` for Wan). The
bot uses the same advertised frame/FPS defaults as Studio, supports LTX-2's
full 20-second single-generation limit, and rejects a duration beyond the
selected model's limit. The existing `frames` and `fps` options remain
available for precise control; `duration` and `frames` cannot be combined.

LTX-2 specialized modes are selected by their attachments: `audio_file` starts
audio-to-video, `source_video` plus both retake times regenerates that time
range, and two or three `keyframe_*` images are spaced across the requested
frame count for interpolation. These modes are mutually exclusive in one
command and do not need the `pipeline` option.

MiniMax H3's ordered references are their own pair of attachments,
`reference_1` and `reference_2`, each an image, an H.264 MP4, or a WAV.
Ordering is explicit, so `reference_1` must be present before `reference_2`.
They cannot be combined with `source_image`, `source_video`, `audio_file`,
the `keyframe_*` images, either retake time, or the `pipeline` option.

Negative prompts: leaving `negative_prompt` unset applies the model's
advertised default negative (Wan ships a tuned one). To explicitly disable the
negative prompt, pass `negative_prompt: none` (case-insensitive; `-` also
accepted); Discord's 25-option cap means the opt-out rides the existing
option as a sentinel rather than its own toggle.
:::

::: tip Durable sequences
`/sequence` accepts prompts separated by `|` and queues them through the
server's durable chain-job API. A normal bot-authored channel message follows
clip progress and attaches the completed MP4 when it fits, so delivery does not
expire with the slash-command interaction token. Larger outputs receive a deep
link to the exact print in the server's Library instead of failing Discord
upload.
:::

::: warning Re-register after command-option changes
`/generate`'s `prompt` option changed from required to optional so an LTX-2
image-to-video run can be submitted with just a `source_image`. Discord caches
command definitions, so **the bot's slash commands must be re-registered** after
upgrading before users see the optional prompt or new `duration` option. The
prompt relaxation is guarded on both ends: the bot only skips the up-front
check when visual conditioning (`source_image`, `source_video`, or keyframes)
is attached, and the server's family-aware validator still rejects an empty
prompt for every image family and for unconditioned text-to-video.
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
  # tokenFile is loaded via systemd EnvironmentFile.
  # the file must contain: MOLD_DISCORD_TOKEN=your-token-here
  tokenFile = config.age.secrets.discord-token.path;
  moldHost = "http://localhost:7680";
  cooldownSeconds = 10;
};
```
