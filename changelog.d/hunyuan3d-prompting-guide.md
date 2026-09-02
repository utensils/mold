- **Expand and remix know when a model reads no prompt.** `mold expand`,
  `mold remix`, `--expand`, `POST /api/expand`, `POST /api/remix`, the MCP
  `expand_prompt` / `remix_prompt` tools, and the TUI's Expand and Remix used
  to hand a Hunyuan3D request to the expansion model with a guide that says
  "write no prompt". They now ask the generation profile's one prompt rule
  first and, for a family that ignores its prompt, answer with the guide's
  own image-preparation advice as the single result — no expansion model is
  created, activated, or pulled, and generation-time expansion is skipped
  instead of rewriting provenance. The `GENERATION CONTEXT` block states
  when the prompt is not read, and `ExpandContext.prompt_mode` carries the
  resolved contract on the wire (additive).
- **The Hunyuan3D prompting guide matches the merged contract.** The family
  guide, the agent skill, and the website page now say that the mesh
  controls come from the model's profile, that OBJ/STL/PLY are gallery-side
  exports, and that the same picture meshes differently in ComfyUI because
  mold letterboxes on the alpha bounding box the way Tencent's preprocessor
  does while ComfyUI centre-crops
  ([#1528](https://github.com/utensils/mold/issues/1528)).
