- **Share a 3-D print as a turntable GIF, APNG or WebP.** Nothing outside a
  3-D tool opens a `.glb`, and the gallery poster shows one view.
  `mold library export chair.glb --format gif` now renders that poster set
  spinning — the same camera, lighting and slate background swept a full turn
  around the mesh, so the first frame IS the poster — and writes it as an
  animation you can drop into a chat, a README or a browser. The flags are the
  gallery video export's own: `--playback loop|bounce`, `--repeat forever|once`
  and `--max-dimension` (240–2048, default 512), plus `--frames` (8–180,
  default 36, a 10° step) and `--fps` (1–30, default 10). A loop renders one
  full turn whose last frame stops one step short of the first, so the wrap is
  seamless; a bounce renders half a turn that the GIF encoder plays back, so
  the reversal reads as deliberate rather than a full turn snapping into
  reverse. Bounce and `once` are GIF contracts, exactly as for a video. The
  same options are on `POST /api/gallery/export/:filename` (`playback`,
  `repeat`, `max_dimension`, `frames`, `fps`; every bound a `422` naming it,
  and the frame buffer capped at the video export's 256 MiB) and the
  `export_mesh` MCP tool; `capabilities.mesh.export_formats` and
  `/api/gallery/export-options` advertise `gif`, `apng` and — on a build with
  the `webp` feature — `webp` beside the geometry containers, and a client
  skips an export format it does not know instead of failing the read. The
  TUI's `x` picker lists whatever the owning host advertises, renders a local
  print in-process through the same code at the fixed defaults, and says so.
  A flat mesh's edge-on frames render as background instead of failing the
  sequence.
