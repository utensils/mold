- **Hunyuan3D image-to-3D.** `mold run hunyuan3d-mini-turbo --image photo.png -o
  chair.glb` turns a single photograph into a 3-D mesh. The result is a durable
  library print like any other: it lands in the gallery with a rendered poster
  tile, and it lists, downloads, restores from trash and reuses its settings
  exactly like an image or a clip. Three Hunyuan3D 2.0 tiers ship —
  `hunyuan3d-mini-turbo` (0.6B, ~5 GB VRAM, the default), `hunyuan3d-turbo`, and
  the undistilled `hunyuan3d` — and the weights are gated behind an explicit
  acceptance of Tencent's community licence, which does not apply in the EU, the
  UK or South Korea ([#1495](https://github.com/utensils/mold/issues/1495)).
- **`--octree`, `--mesh-threshold` and `--target-faces` control the geometry.**
  `--octree` is the detail knob and its cost is cubic; `--mesh-threshold` moves
  the extracted surface; `--target-faces` decimates. Meshes are stored as binary
  glTF with the geometry, normals and materials in one self-contained file
  ([#1495](https://github.com/utensils/mold/issues/1495)).
