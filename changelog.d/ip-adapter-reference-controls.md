- **A reference picture can now ride WITH a source image, a mask, ControlNet
  and a LoRA.** IP-Adapter on SD1.5 and SDXL is the first recipe whose
  references add to the conditioning instead of replacing it, so Create draws
  both wells live at once and parks neither — every earlier reference family
  (Qwen Image Edit, FLUX.2 [dev] and [klein]) still behaves exactly as it did.
  A drop with no well under the cursor lands on the Source well, the request
  carries `source_image` and `edit_images` together, and the batch is no
  longer coerced to one print, because an image prompt broadcasts across every
  row.
- **New Reference strength slider** beside the reference strip, with the
  bounds the host itself advertises. It appears only where the recipe declares
  an adapter — an older host or a checkpoint without one shows nothing — and
  stays off the wire until you move it, so a default render is unchanged.
