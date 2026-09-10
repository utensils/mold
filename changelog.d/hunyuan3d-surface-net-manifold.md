- **Surface extraction no longer webs thin geometry together, so a frame or a
  set of spokes textures in seconds instead of stalling.** mold's port of
  ComfyUI's surface nets emitted a quad wherever four cells around a grid edge
  happened to be active, rather than where that edge actually crosses the
  surface. On a shape whose surfaces never fold back into the same voxel the two
  rules agree exactly; on the thin, self-touching geometry Hunyuan3D produces
  for a pram frame or a wheel the difference was most of the mesh. The shape
  from [#1666](https://github.com/utensils/mold/issues/1666) came out with
  508,838 triangles, 78% of its edges shared by more than two of them, which
  then defeated both stages behind it: decimation stopped at 96,052 triangles
  instead of the 40,000 it was asked for, and UV unwrapping took 1,133 s. The
  same shape now extracts 381,080 triangles with 0.2% of edges non-manifold and
  no open boundary at all, decimates to exactly 40,000, and unwraps in 4.1 s —
  280x. The surplus quads were visible too: the flat ground plate under that
  render was covered in overlapping sheets that are simply gone. A mesh you
  upload to a texture-only workflow can still arrive non-manifold, and that case
  is still slow to unwrap — it reports its progress, stops within seconds of a
  cancel, and now logs a warning when decimation cannot reach the face budget it
  was given ([#1669](https://github.com/utensils/mold/issues/1669)).
