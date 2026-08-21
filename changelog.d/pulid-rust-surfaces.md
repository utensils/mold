- **Identity photos in the TUI.** The Create form's Advanced accordion gains an
  **Identity photo** section — a local PNG/JPEG path plus Strength (`0.0`–`3.0`,
  default `1.0`) and Start step (`0`–`steps-1`) — shown only for a checkpoint the
  connected server advertises as identity-capable. The photo is opened
  no-follow and bounds-checked when you enter it, so an unreadable, symlinked,
  oversized, or non-image file is refused in the picker instead of after a
  queue slot. Switching to a model that cannot take the photo keeps it and
  refuses the render with the server's own wording rather than quietly
  rendering someone else's face; Reset to model defaults clears it. Library
  Details and the full print view show the reference's name, short digest,
  strength, and start step
  ([#1231](https://github.com/utensils/mold/issues/1231)).
- **`/identity` on the Discord bot.** A new slash command generates from a face
  reference photo: `identity` (PNG/JPEG attachment), `identity_strength`, and
  `identity_start_step`, plus the usual prompt/model/size/steps/guidance/seed
  options. It refuses an oversized or wrong-container upload before
  downloading it, checks both knobs and the model gate against the server's
  advertised `supports_identity`, and names the reference in the result embed.
  It is a separate command because `/generate` already sits at Discord's hard
  25-option ceiling — and identity is qualified only for the FLUX dev tiers,
  so none of `/generate`'s video and conditioning options apply
  ([#1231](https://github.com/utensils/mold/issues/1231)).
