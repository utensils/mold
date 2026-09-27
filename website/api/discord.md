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

| Command              | Description                                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `/generate`          | Generate an image or video, including attachment-driven LTX-2 audio-to-video, retake, and keyframe modes and ordered MiniMax H3 references |
| `/mesh`              | Generate a Hunyuan3D GLB from one source image or semantic front/left/back/right multiview attachments                                     |
| `/transparent`       | Render a subject on a transparent background (PNG or WebP with alpha), optionally cut out of up to three ordered reference images          |
| `/identity`          | Generate an image conditioned on a face reference photo (PuLID), with `identity_strength` and `identity_start_step`                        |
| `/expand`            | Expand a short prompt into detailed generation prompts                                                                                     |
| `/remix`             | Rewrite a prompt into subject-preserving alternatives, one creative dimension each (`dimensions`, `style`, `variations` 1-5)               |
| `/models`            | List available models with download/loaded status                                                                                          |
| `/status`            | Show server health, queue summary, and every GPU/MIG device; large fleets paginate across limit-safe follow-up embeds                      |
| `/quota`             | Check your remaining daily generation quota                                                                                                |
| `/admin reset-quota` | Reset a user's daily quota (requires Manage Server)                                                                                        |
| `/admin block`       | Temporarily block a user from generating (requires Manage Server)                                                                          |
| `/admin unblock`     | Unblock a previously blocked user (requires Manage Server)                                                                                 |

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

### `/transparent`

Transparent-background renders have their own command for the same reason as
`/identity`: `/generate` is already at Discord's 25-option ceiling. It offers
only models whose recipe advertises `capabilities.transparency` (today Qwen
Image 2.1; the model option autocompletes those, downloaded first) and sends
`transparent_background: true`.

| Option                        | Purpose                                                                             |
| ----------------------------- | ----------------------------------------------------------------------------------- |
| `prompt`                      | Required. The subject alone — no scenery or backdrop.                               |
| `model`                       | Transparency-capable model; defaults to one the server advertises.                  |
| `format`                      | `PNG` (default) or `WebP`. JPEG is not offered because it cannot carry alpha.       |
| `reference_1` … `reference_3` | Ordered reference images (PNG, JPEG or WebP), e.g. a picture to cut a subject from. |
| `seed`                        | Seed for reproducibility.                                                           |
| `width` / `height`            | Output size. Without them the last reference's aspect ratio is used.                |
| `steps`                       | Inference steps.                                                                    |

The prompt you type is what is stored; the model's RGBA wording is added by
the engine. `reference_2` needs `reference_1`, and `reference_3` needs both, so
the order the prompt names ("image 1", "image 2") is never ambiguous.

### `/mesh`

Mesh generation has its own compact command because `/generate` already uses
Discord's 25-option ceiling. Attach `source` for a single-view Hunyuan3D model,
or any non-empty subset of `front`, `left`, `back`, and `right` for a 2mv
checkpoint. The two forms are mutually exclusive. If `model` is omitted,
`source` selects the ordinary Hunyuan3D default and named views select the
five-step `hunyuan3d-2mv-turbo:fp16` tier. Optional `seed`, `texture`,
`octree`, `threshold`, and `target_faces` values ride the same generation
request and admission checks as every other client. The response includes the
poster and GLB using the existing Discord mesh delivery policy.

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

`/generate`'s `reference_1` and `reference_2` attachments are routed by the
selected model's capability, never its name. On a model that advertises
`capabilities.reference_images` (FLUX.2, Qwen-Image-Edit, Qwen Image 2.1, and
the SD 1.5 / SDXL image prompt) they become ordered `edit_images`; for Qwen
Image 2.1 they may be PNG, JPEG or WebP, and without `width`/`height` the
output takes the last reference's aspect ratio. Only an image-prompt model
also keeps `source_image`; everywhere else a source image beside references is
refused. For MiniMax H3 Ref2VA they are its ordered references, each an image,
an H.264 MP4, or a WAV, and they cannot be combined with `source_image`,
`source_video`, `audio_file`, the `keyframe_*` images, either retake time, or
the `pipeline` option. Any other model refuses them with the server's own
sentence. Ordering is explicit, so `reference_1` must be present before
`reference_2`.

Negative prompts: leaving `negative_prompt` unset applies the model's
advertised default negative (Wan ships a tuned one). To explicitly disable the
negative prompt, pass `negative_prompt: none` (case-insensitive; `-` also
accepted); Discord's 25-option cap means the opt-out rides the existing
option as a sentinel rather than its own toggle.
:::

::: warning Re-register after command-option changes
`/generate`'s `prompt` option changed from required to optional so an LTX-2
image-to-video run can be submitted with just a `source_image`. Discord caches
command definitions, so **the bot's slash commands must be re-registered** after
upgrading before users see the optional prompt or new `duration` option. The
new `/mesh` command also appears only after re-registration. The
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
