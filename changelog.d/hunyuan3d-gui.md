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
  WebGL `MeshViewer` now mounts directly in the Create result area as well as
  the Lightbox and the mobile viewer sheet, with a wireframe toggle beside its
  `tris · verts · bounds` caption (the toggle is disabled, with a reason, for
  a mesh with no edges to outline). In the three Create result areas — web,
  desktop, and iPhone/Android — it also auto-rotates until touched, honours
  `prefers-reduced-motion` (parking or resuming the moment the setting
  changes), and offers a fullscreen toggle; the Lightbox and the viewer
  sheets do not auto-rotate or go fullscreen, and fullscreen is unavailable
  inside the iOS WKWebView, where the button never appears. The recent strip
  and every Library tile carry a 3D badge; the web and desktop Library also
  gain a "3D" kind filter alongside Images, Video, and Audio (the iPhone
  Library has the badge only). The lightbox refuses **Use as source** and
  offers no **Upscale** for a mesh — there is no raster to stage as
  conditioning or to enlarge. The empty Create canvas explains such a model
  in its own words (prepare the source image) instead of the optional-prompt
  wording about motion, and the web rail's Target faces field warns inline
  when a value falls outside the advertised bounds instead of letting
  Generate fail. A
  reloaded web page hydrates a mesh print from its poster and recorded counts
  exactly like any other kind.
- **Export as OBJ, STL, or PLY without leaving the app.** The lightbox and the
  mobile viewer sheet offer the host's advertised
  `capabilities.mesh.export_formats` — the same conversions the
  `mold library export` command and the `export_mesh` MCP tool perform — and
  never the stored GLB itself, which Download already covers. Desktop saves
  the converted file through its normal download path; mobile hands it to the
  native share sheet. A host that advertises an animated turntable container
  (GIF, APNG, WebP) collapses those into one **Export turntable…** entry that
  opens the existing GIF-export options sheet (playback direction, repeat,
  max dimension, frame rate) instead. The web lightbox shows a refused
  export beside the menu on both its layouts and clears it when you move to
  the next print.
- **Expand and Remix are refused for a model that reads no prompt.** On a
  recipe whose profile advertises `prompt.mode: ignored`, both controls render
  disabled on web, desktop, and iPhone with the one sentence "This model reads
  no prompt; prepare the image instead.", the keyboard, menu, and recovery
  paths answer with the same sentence instead of sending a request, and the
  missing-expander pull is never offered. The host answers such a transform
  with a single image-preparation note rather than a batch of variants, so
  every client validator now accepts that one result for a prompt-ignoring
  recipe instead of failing the batch.
