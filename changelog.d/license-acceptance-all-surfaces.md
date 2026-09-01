- **Accept model licenses from every surface, not just the CLI.** Picking a
  license-gated model such as Hunyuan3D on web, desktop or mobile used to dead-end
  at "model not installed" with no way to read or accept the terms — a placement
  preview only carried license terms for dependency bundles, never for the
  checkpoint itself. Previews now carry the requested model's own outstanding
  terms, so the existing consent dialog appears wherever you generate, and
  installing a gated model from the Models page prompts for consent instead of
  failing with a raw error (this also fixes PuLID installs from that page).
- **Accept terms without downloading gigabytes.** New `POST /api/licenses/accept`
  and `mold licenses accept <id>...` record consent on its own; the Settings
  license panel gains a matching "Accept terms" action beside the existing
  review-and-download button. `mold pull --accept-license` may now be repeated
  for a bundle covered by more than one agreement.
- **Accept on behalf of a remote host from desktop Settings.** The license panel
  gains a machine selector, so consent can be recorded on the server that will
  do the downloading, matching what mobile and generate-time routing already did.
- **Review licenses from the TUI.** The command palette gains "Review model
  licenses…", listing each agreement, its pinned terms and what needs it, for
  whichever host the TUI is pointed at.
- **A refused pull is no longer reported as a server fault.** Auto-pull paths
  mapped a license refusal to `500 INTERNAL_ERROR` and discarded the pinned
  terms; they now return the structured `403 LICENSE_NOT_ACCEPTED` a client can
  act on.
- **Hunyuan3D 2.1 PBR paint weights are installable.** The `hunyuan3d-paint`
  bundle ships so the Tencent 2.1 agreement is required by something and can be
  accepted; previously it was listed on every surface and acceptable on none.
  The paint engine is not implemented yet, so these weights satisfy the license
  gate but do not render.
