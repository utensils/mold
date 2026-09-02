- **3-D exports come out print-ready.** Exporting a mesh as OBJ, STL, or PLY
  no longer hands back Hunyuan3D's raw normalized-unit-cube geometry, which
  slicers read as a 2 mm blob and refuse ("object too small") and which
  Blender's STL/PLY importers land on their side. `mold library export
  chair.glb --format stl --size-mm 120 --up-axis y --origin center` (and the
  matching `POST /api/gallery/export/:filename` fields, MCP `export_mesh`
  args, and the web/desktop/iPhone export options sheet) now scale, orient,
  and position the mesh, defaulting to what each format's own tools expect
  (100 mm, Z-up, floor for STL and PLY; unscaled, Y-up, floor for OBJ). A
  server that doesn't advertise `capabilities.mesh.export_geometry` gets none
  of the new fields, and the options are refused rather than ignored on `glb`
  or an animated turntable.
- **Turntables no longer breathe.** A GIF/APNG/WebP turntable export used to
  refit its scale to every frame's own silhouette, so a spinning mesh visibly
  grew and shrank (up to ~41% on a square footprint) and popped at the
  x/y crossover. The sweep is now fitted once for the whole turn, so the
  mesh holds one size throughout.
