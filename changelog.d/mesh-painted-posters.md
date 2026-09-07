- **Painted 3-D prints show their colours.** A Hunyuan3D print with PBR
  materials now renders its gallery thumbnail and its turntable GIF, APNG or
  WebP export in the baked texture's own colours instead of the bare grey
  placeholder surface. The poster reader had been dropping the `.glb`'s
  embedded `baseColorTexture` and vertex colours, so a painted mesh was the
  only thing in mold that looked different in the Library than in the 3-D
  viewer. Cached grey tiles are re-rendered automatically.
