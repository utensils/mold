- **MiniMax H3 references are never upscaled.** Ref2VA image references
  smaller than a 2048 px short edge now keep their native geometry (32-aligned)
  and video references keep theirs when they fit the reference canvas, matching
  ComfyUI's `min(1.0, 2048/short)` policy. Two phone photographs (582x1200) were
  being inflated to 2048x4224 each — about 67,000 vision-patch rows — and refused
  with an impossible ~82.7 GB host-memory demand on a 64 GB host; the same print
  now admits at roughly a quarter of the memory. The Create crop hint on web,
  desktop, and iPhone mirrors the new arithmetic, and `REFERENCE_PREPROCESS_VERSION`
  is bumped so a job held under the old shapes re-derives them on retry.
