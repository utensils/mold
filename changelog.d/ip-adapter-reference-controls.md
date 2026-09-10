- **Create draws both image wells at once for an SD 1.5 or SDXL reference.**
  IP-Adapter is the first recipe whose references ADD to the conditioning
  instead of replacing it, so neither well parks and the request carries the
  source image and the reference together — every earlier reference family
  (Qwen Image Edit, FLUX.2 [dev] and [klein]) behaves exactly as it did. A drop
  with no well under the cursor lands on the Source well, and the batch is no
  longer coerced to one print
  ([#1573](https://github.com/utensils/mold/issues/1573)).
- **Reference strength control on web, desktop and phone.** It renders only
  where the recipe declares an adapter and takes its bounds from the host, so
  an older machine or a checkpoint without one shows nothing, and it stays off
  the wire until you move it — a default render is unchanged
  ([#1573](https://github.com/utensils/mold/issues/1573)).
