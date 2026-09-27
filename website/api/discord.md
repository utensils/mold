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
| `/upscale`           | Upscale an attached PNG, JPEG, or WebP with a server-side Real-ESRGAN model; the result is private to the requester                        |
| `/search`            | Search the external model catalog read-only; results are sanitized, bounded, and ephemeral                                                 |
| `/models`            | List available models with download/loaded status                                                                                          |
| `/status`            | Show server health, queue summary, and every GPU/MIG device; large fleets paginate across limit-safe follow-up embeds                      |
| `/quota`             | Check your remaining daily generation quota                                                                                                |
| `/queue`             | Inspect or control the host-global queue (guild-only, Manage Server, ephemeral)                                                            |
| `/downloads`         | Inspect or cancel host-global model downloads (guild-only, Manage Server, ephemeral)                                                       |
| `/video-upscale`     | Manage Library-backed video upscales (guild-only, Manage Server, ephemeral)                                                                |
| `/gallery`           | Browse host-global Library metadata read-only (guild-only, Manage Server, ephemeral)                                                       |
| `/admin reset-quota` | Reset a user's daily quota (requires Manage Server)                                                                                        |
| `/admin block`       | Temporarily block a user from generating (requires Manage Server)                                                                          |
| `/admin unblock`     | Unblock a previously blocked user (requires Manage Server)                                                                                 |

## Capability and exclusion matrix

Discord parity is defined by user capability, not by copying every CLI flag
into a slash command. The bot is an HTTP-only client of one Mold server and
uses that server's advertised generation profile as the authority for model
availability, defaults, ranges, fixed controls, accepted inputs, and output
formats; Discord does not define a separate availability allowlist. If an
additive capability block is absent, the bot treats that as an older server and
either uses the documented legacy behavior or gives a clear unsupported
response.

The following matrix accounts for every current `GenerateRequest` field.
"Server/default" means Discord deliberately leaves the field unset or sends a
fixed transport value so the selected recipe or server remains authoritative.

| `GenerateRequest` field(s)                                      | Discord status                               | Discord surface or reason                                                                                                                                                                                        |
| --------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`, `model`                                               | Implemented                                  | `/generate`; focused commands also select a model and provide the prompt shape appropriate to that capability.                                                                                                   |
| `negative_prompt`                                               | Implemented                                  | `/generate`; `none` or `-` sends an explicit empty negative, while omission preserves the recipe default.                                                                                                        |
| `width`, `height`, `steps`, `guidance`, `seed`                  | Implemented                                  | `/generate`, `/identity`, and the applicable focused commands. Omitted values come from the server-advertised recipe.                                                                                            |
| `batch_size`                                                    | Intentional exclusion                        | Discord creates one result per interaction. Batch authoring belongs to the CLI and Studio clients.                                                                                                               |
| `output_format`                                                 | Implemented with delivery limits             | `/generate` chooses MP4 or GIF for video; `/transparent` chooses PNG or WebP; images default to PNG; mesh output is pinned to GLB. Other still containers remain CLI/API choices.                                |
| `embed_metadata`                                                | Server/default                               | Discord does not override the host's metadata policy.                                                                                                                                                            |
| `offload`, `placement`                                          | Intentional exclusion                        | Host memory policy and component placement are operator and CLI/API controls.                                                                                                                                    |
| `scheduler`, `cfg_plus`                                         | Intentional exclusion                        | Advanced sampler controls remain CLI/API controls.                                                                                                                                                               |
| `source_image`, `source_image_name`                             | Implemented                                  | `/generate` accepts a bounded PNG/JPEG attachment and records its sanitized upload name.                                                                                                                         |
| `source_fit`                                                    | Intentional exclusion                        | Discord has no interactive crop/pad editor; it preserves source aspect through the shared fit helpers.                                                                                                           |
| `id_image`, `id_image_name`, `id_weight`, `id_start_step`       | Implemented                                  | `/identity` accepts one PNG/JPEG identity photo and exposes strength and start step.                                                                                                                             |
| `id_images`, `id_image_names`, `true_cfg`, `cfg_start_step`     | Intentional exclusion                        | Multiple identity photos and PuLID true-CFG controls remain CLI/API and Studio capabilities; `/identity` is already a focused, bounded Discord form.                                                             |
| `edit_images`                                                   | Implemented with a Discord bound             | `/generate` exposes two ordered reference attachments; `/transparent` exposes three. The recipe may accept more through CLI/API.                                                                                 |
| `reference_weight`                                              | Intentional exclusion                        | IP-Adapter reference strength remains a CLI/API and Studio control.                                                                                                                                              |
| `transparent_background`                                        | Implemented                                  | `/transparent`, offered only when the selected recipe advertises transparency and a compatible alpha format.                                                                                                     |
| `references`                                                    | Implemented for chat-suitable media          | `/generate` carries two ordered MiniMax H3 image/video/audio references; `/mesh` uses named image references for ordinary multiview generation. Mesh-reference workflows are excluded.                           |
| `strength`                                                      | Implemented                                  | `/generate`, with the selected recipe's source-strength semantics and admission bounds.                                                                                                                          |
| `mesh`                                                          | Partly implemented                           | `/mesh` exposes ordinary image-to-mesh `texture`, `octree_resolution`, `threshold`, `target_faces`, and `matting`. `texture_resolution`, `texture_view_count`, and `delight` remain CLI/API and Studio controls. |
| `mask_image`                                                    | Intentional exclusion                        | Inpainting mask authoring needs a canvas/editor and remains in the CLI/API and Studio clients.                                                                                                                   |
| `control_image`, `control_model`, `control_scale`               | Intentional exclusion                        | ControlNet setup remains a CLI/API and Studio capability.                                                                                                                                                        |
| `expand`, `original_prompt`, `prompt_transform`                 | Intentional exclusion on generation requests | `/expand` is a separate prompt tool. It returns text for review rather than silently transforming a generation request.                                                                                          |
| `save_to_gallery`                                               | Server/default                               | Discord generations follow the host's normal durable publication behavior.                                                                                                                                       |
| `title`, `tags`, `collection`                                   | Intentional exclusion                        | Library filing and naming remain in the CLI/API and Studio clients.                                                                                                                                              |
| `batch_id`, `batch_index`, `batch_count`                        | Intentional exclusion                        | These are prepared-batch provenance fields, and Discord does not author batches.                                                                                                                                 |
| `mesh_workflow`                                                 | Not client-authorable                        | This is server-minted workflow provenance. Durable mesh workflows are CLI/API-only.                                                                                                                              |
| `lora`, `loras`                                                 | Intentional exclusion                        | Single and stacked LoRA authoring remain CLI/API and Studio controls.                                                                                                                                            |
| `frames`, `fps`                                                 | Implemented                                  | `/generate`; `duration` is converted through the advertised temporal grid and limits.                                                                                                                            |
| `upscale_model`                                                 | Intentional exclusion on generation requests | Chained post-generation upscale remains a CLI/API and Studio control. The separate `/upscale` command covers attachment-based image upscaling.                                                                   |
| `gif_preview`                                                   | Implemented transport policy                 | The bot requests a GIF preview for video so it can fall back when the primary MP4 exceeds Discord's upload ceiling.                                                                                              |
| `enable_audio`                                                  | Implemented                                  | `/generate` exposes synchronized audio where the recipe supports it; omission preserves the recipe default.                                                                                                      |
| `video_only`                                                    | Intentional exclusion                        | The advanced LTX-2 video-only execution mode remains CLI/API.                                                                                                                                                    |
| `audio_file`                                                    | Implemented                                  | `/generate` attachment for LTX-2 audio-to-video.                                                                                                                                                                 |
| `audio_file_path`                                               | Intentional exclusion                        | Server-local paths are trusted-deployment API/CLI inputs and are never accepted from chat.                                                                                                                       |
| `source_video`, `retake_range`                                  | Implemented                                  | `/generate` attachment plus start/end options for LTX-2 retake.                                                                                                                                                  |
| `source_video_path`                                             | Intentional exclusion                        | Server-local paths are trusted-deployment API/CLI inputs.                                                                                                                                                        |
| `extend_video`, `extend_video_path`, `extend_overlap_frames`    | Intentional exclusion                        | Video continuation and its server-local path form remain CLI/API and Studio capabilities.                                                                                                                        |
| `keyframes`                                                     | Implemented with a Discord bound             | `/generate` accepts two or three LTX-2 keyframes, or the two endpoint images supported by Wan.                                                                                                                   |
| `hdr_exr_dir`, `hdr_exr_full_float`                             | Intentional exclusion                        | Host-local EXR sidecar output is a CLI/API workflow.                                                                                                                                                             |
| `pipeline`                                                      | Implemented                                  | `/generate` exposes ordinary LTX-2 recipes; attachment-driven audio, retake, and keyframe modes select their recipe automatically.                                                                               |
| `ic_lora_control`, `guidance_overrides`                         | Intentional exclusion                        | LTX-2 camera/HDR adapters and detailed guider overrides remain CLI/API controls.                                                                                                                                 |
| `spatial_upscale`, `temporal_upscale`                           | Intentional exclusion                        | LTX-2 latent upscale controls remain CLI/API and Studio capabilities.                                                                                                                                            |
| `sample_shift`, `distill_strength_high`, `distill_strength_low` | Intentional exclusion                        | Advanced Wan recipe controls remain CLI/API and Studio capabilities.                                                                                                                                             |

The profile-level controls follow the same boundary:

| Generation profile contract                                                                         | Discord handling                                                                                                                                                                              |
| --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recipe selector and defaults                                                                        | Reads the selected recipe's pipeline selector and defaults; focused attachments select specialized LTX-2 recipes.                                                                             |
| Resolution domain, alignment, bounds, aspect, and presets                                           | Accepts explicit width/height, derives a source/reference aspect where documented, and leaves final validation to the advertised recipe. Discord does not reproduce the Studio preset picker. |
| Step and guidance controls                                                                          | Exposes the numeric values and uses advertised defaults; fixed or hidden controls remain fixed or hidden.                                                                                     |
| Temporal frames, FPS, duration, and grid                                                            | Exposes frames/FPS plus a duration convenience option, snapping and bounding through the advertised temporal contract.                                                                        |
| Prompt requirement                                                                                  | Honors required, optional, or ignored prompt behavior. Ordinary `/mesh` needs image conditioning because Hunyuan3D ignores text.                                                              |
| Source image and strength                                                                           | Exposes both when the recipe accepts them and rejects incompatible combinations before upload where possible.                                                                                 |
| Negative prompt and audio                                                                           | Exposes both when supported; omission preserves recipe policy.                                                                                                                                |
| Ordered reference images                                                                            | Exposes ordered attachments within Discord's option and request-size limits; server admission owns the recipe ceiling and format rules.                                                       |
| Identity and transparency                                                                           | Exposed through `/identity` and `/transparent`, gated by server-advertised capability.                                                                                                        |
| Mesh generation                                                                                     | Exposes ordinary image-to-mesh and named multiview controls through `/mesh`. Supplied-mesh, text-to-mesh, texture-only, mesh-roundtrip, and other durable workflow modes are CLI/API-only.    |
| LoRA, ControlNet, masks, schedulers, sequence authoring, continuation, and advanced family controls | Intentionally excluded as listed above. Scripted chain jobs and their durable lifecycle are CLI/API-only.                                                                                     |
| Output formats                                                                                      | Narrows the advertised formats to containers Discord can deliver safely; the server still performs final admission.                                                                           |
| Provenance and delivery notes                                                                       | Display metadata only; they are not user-editable controls.                                                                                                                                   |

Auto-chaining a single long video request is an engine implementation detail,
not Discord sequence authoring. Discord has no `/sequence`, chain-job lifecycle,
or durable mesh-workflow commands. Use `mold run --script`, `mold jobs`,
`mold mesh-workflow`, or the corresponding HTTP APIs for those workflows.

### Access boundary for host-global operations

Queue rows, download jobs, Library filenames, saved prompts, and video-upscale
jobs belong to the Mold host; the current server contract does not attach a
Discord user owner to them. `/queue`, `/downloads`, `/video-upscale`, and
`/gallery` are therefore guild-only commands that require Discord's **Manage
Server** permission and always reply ephemerally. Global queue pause/resume and
per-item pause/resume are separate actions and call separate server endpoints;
one never stands in for the other. Gallery access is read-only, and Discord
does not expose destructive Library actions.

| Operator command | Subcommands                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| `/queue`         | `list`, `show`, global `pause` / `resume`, per-item `pause-item` / `resume-item`, `cancel`, and `retry` |
| `/downloads`     | `list` and `cancel`; starting an install is deliberately absent                                         |
| `/video-upscale` | `create`, `list`, `status`, `cancel`, and `resume`, using an exact Library filename as the source       |
| `/gallery`       | Read-only `list` and `show`; `show` may attach a bounded thumbnail                                      |

`/search` is a read-only user command. It applies the normal allowed-role
check, bounds and sanitizes external catalog text, and replies ephemerally.
Installing a catalog result or starting a model pull remains outside Discord so
license acceptance stays explicit. An operator may cancel an already-created
download with `/downloads cancel`; catalog license acceptance is never inferred
from a search.

### `/upscale`

`/upscale` accepts one PNG, JPEG, or WebP attachment up to 10 MiB, an optional
upscaler model, and an optional tile size. It uses the same generation access,
cooldown, daily-quota, and failure-refund policy as the rendering commands. The
bot returns a PNG in an ephemeral response and refuses a result above its 24 MiB
Discord delivery ceiling. This command calls the standalone image-upscale API;
it does not set `GenerateRequest.upscale_model` on a new generation.

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
