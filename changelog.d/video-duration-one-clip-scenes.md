- **MiniMax H3 Duration videos up to 14.4 s render as one clip.** A Duration
  request now plans H3 clips at its full single-generation length (345 frames)
  instead of the 124-frame default, so `--duration 14` is one clip rather than
  three short clips replaying the same prompt.
- **Scenes for longer videos, from the CLI too.** `mold run MODEL --scenes
  scenes.toml` takes one `[[scene]]` per part (`seconds` plus its own
  `prompt`); the video length is their sum and the dry run prints the scene
  timeline. Every scene's prompt is validated before anything renders, and the
  plan now warns when a scene or single prompt is longer than one generation
  (its action would repeat) or shorter than the shortest generation.
- **H3 continuation clips are prompted as continuations.** Each clip that
  starts from the previous clip's last frame opens with MiniMax's upstream
  frame-alignment instruction, so the model treats the carried frame as the
  opening picture. Videos already in progress resume unchanged.
- **Video jobs report what they are actually doing.** `/api/video-jobs` adds
  `progress_phase` (`waiting`, `rendering`, `joining`) and `/api/activity`
  reports a waiting video as queued, so Queue no longer shows every accepted
  video as running while only some render.
- **Chained videos no longer get darker with every clip.** Every H.264
  encode and decode now uses one rounded BT.601 conversion instead of the
  codec library's truncating helpers, which shaved about one code off every
  channel per pass. The frame carried into each next clip went through
  several passes, so a dark H3 scene lost roughly 13% of its brightness at
  every clip boundary.
