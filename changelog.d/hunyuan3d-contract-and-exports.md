- **3-D generation is a first-class contract, and meshes export as OBJ, STL and
  PLY.** Hunyuan3D worked from the CLI and nowhere else, because every other
  surface asked the wrong questions of it. The generation profile is now the
  single authority for all three: it says whether a recipe requires, accepts,
  or IGNORES a prompt (this family has no text encoder at all, so
  `mold run hunyuan3d-mini-turbo --image cutout.png -o chair.glb` needs none),
  whether `strength` changes the render, and — for a mesh recipe — the octree
  resolutions, iso-threshold, and face bounds a client should offer instead of
  a resolution picker. Server validation and every client read those fields
  rather than carrying their own family lists
  ([#1496](https://github.com/utensils/mold/issues/1496)).
- **An unavailable output format is a 422 at the door, not a job that holds.**
  Durable admission now checks an explicit `output_format` against the resolved
  recipe before the row is written, so a client naming a container the model
  cannot deliver is told at submit time instead of watching a print hold and
  then fail. A mesh model is the deliberate exception: it stores binary glTF
  and nothing else, so an explicit `png` from an older client is COERCED to
  `glb` rather than refused — the same rule `mold run` already applied.
- **`mold library export <file> --format obj|stl|ply`** transcodes a stored
  `.glb` into a file other tools read: OBJ for Blender and MeshLab, STL for
  3-D printers and CAD, PLY for point-and-mesh tooling. The gallery keeps its
  glTF, which is the only form carrying geometry, UVs, normals and textures in
  one file. The same conversions are on `POST /api/gallery/export/:filename`
  and the `export_mesh` MCP tool, and `/api/capabilities.mesh.export_formats`
  advertises them.
- **The CLI stops describing a 3-D render as an image.** A mesh run no longer
  resamples the source to a canvas the engine ignores or announces an img2img
  strength nothing reads; it prints the octree resolution and threshold that
  actually shape the mesh. Every progress line is printed once with its own
  duration — encode, sampling, volume decode, surface extraction, and write
  were previously printed twice each with the whole render attributed to the
  decode — and `-o` refuses a filename a 3-D render cannot write, pointing at
  the export command for `.obj`, `.stl`, and `.ply`.
