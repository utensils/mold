- **Hunyuan3D generates and renders from the web, desktop, and iPhone/Android
  apps, not just the CLI.** Picking a Hunyuan3D model reshapes Create from the
  recipe's own generation profile, the same one the TUI and Discord already
  read: Shape, Resolution, exact-size, Fit to canvas, Strength, Mask, and
  Negative all disappear because the profile is canvasless, strengthless,
  maskless, and reads none of them, and a **Mesh** control group takes their
  place — Octree detail over the advertised allowlist (128/192/256/320/384,
  256 default), an Iso threshold slider, and an optional Target faces field
  within the advertised bounds that keeps the raw surface when left blank.
  The prompt bed becomes an optional note (placeholder text says the model has
  no text encoder and renders from the source image) and Generate submits with
  no prompt; a source image is still required. Requests are pinned to GLB and
  carry only the mesh controls that differ from the advertised defaults, and
  Reuse settings restores the recorded octree, threshold, and face target from
  the print's `metadata.mesh` instead of a form's leftovers
  ([#1496](https://github.com/utensils/mold/issues/1496)).
- **A finished mesh renders in place, not just in the Library.** The shared
  WebGL2 `MeshViewer` now mounts directly in the Create result area as well as
  the Lightbox and the mobile viewer sheet: it auto-rotates until touched,
  honours `prefers-reduced-motion`, and adds a fullscreen toggle and a
  wireframe toggle beside its `tris · verts · bounds` caption. The recent strip
  and the Library carry a 3D badge and a "3D" kind filter alongside Images,
  Video, and Audio on every surface, and the lightbox refuses **Use as
  source** for a mesh — there is no raster to stage as conditioning. A
  reloaded web page hydrates a mesh print from its poster and recorded counts
  exactly like any other kind.
- **Export as OBJ, STL, or PLY without leaving the app.** The lightbox and the
  mobile viewer sheet offer the host's advertised
  `capabilities.mesh.export_formats` — the same conversions the
  `mold library export` command and the `export_mesh` MCP tool perform.
  Desktop saves the converted file through its normal download path; mobile
  hands it to the native share sheet. A host that advertises an animated
  turntable poster container opens the existing GIF-export options sheet
  (playback direction, repeat, max dimension) for that entry instead.
