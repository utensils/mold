- **FLUX.2 [klein] reference-image editing.** Every Klein tier (4B and 9B, distilled
  and base, BF16/FP8/GGUF) now accepts up to four ordered reference images —
  `mold run flux2-klein "…" --reference a.png --reference b.png`, the References
  strip on web, desktop, and iPhone, and `edit_images` on `POST /api/generate` —
  using the same protocol FLUX.2 [dev] already ran (references VAE-encoded and
  appended to the sequence at time coordinates 10, 20, …; the Qwen3 encoder never
  sees them). Klein keeps its img2img, inpaint, and LoRA paths; one render carries
  either a source image or references, never both, and the Create rail parks
  whichever well is not in use instead of refusing.
- **Reference images are advertised on the wire.** The generation profile carries
  a new `capabilities.reference_images` block (`mode`, `required`, `max_count`,
  `primary_is_target`, `source_relation`, per-image pixel ceilings) built from one
  core decision, and admission, the CLI, the TUI, and every Studio surface read it
  instead of matching model names. Absence of the block means an older server, so
  existing FLUX.2 [dev] and Qwen-Image-Edit clients keep working; the TUI now
  routes a FLUX.2 [dev] picture into `edit_images` rather than a refused
  `source_image`.
- **Dropped images land on the well you dropped them on.** On desktop, an OS file
  drag is routed to the well under the cursor (source, references, identity,
  opening image, end frame, or a MiniMax H3 slot) instead of always overwriting
  the source or attachment slot; a drop onto a references strip appends rather
  than replacing it, and H3 drops reach the H3 authoring fields. On web, a file
  dropped outside a well no longer navigates the browser away from Studio, and a
  drop lands on the well you aimed at, the References strip included.
