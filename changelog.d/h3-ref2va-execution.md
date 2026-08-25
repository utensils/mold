- **MiniMax H3 Ref2VA renders on the shipping build.** The compact
  `minimax-h3-ref2va:comfy-pruned-int8` checkpoint now generates video with
  synchronized audio from an ordered set of image, video, and audio references,
  wherever the H3 engine is built. It carries its own compiled runtime
  qualification — a separate schema, decision string, envelope, and memory
  bounds record derived per request from the reference set's real preprocessing
  shapes — so a set the device cannot hold is refused with numbers instead of
  being refused by name. `/api/models` reports `runtime_available: true` on the
  row, and the earlier "Ref2VA execution is not available in any released
  build" sentence is gone
  ([#825](https://github.com/utensils/mold/issues/825)).
