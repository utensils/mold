- **Ref2VA image references no longer hold or fail at execution.** Admission
  counted a reference's Qwen vision rows as merged 32-px pads while the runtime
  prepared 16-px ViT patches (four per pad), so every image reference was
  admitted, dispatched, and then held as "prepared rows differ from frozen
  admission" (16,384 vision rows against 4,096). Both sides now count patches
  — the grid FL2VA's frozen value and the qualification record already used —
  the conditioner text budget keeps the merged pads, and the memory grant
  scales on the same composition the workspace was measured over. Past that
  check, the conditioner's vision-row validation expected the bare pad count
  and refused every visual reference ("returned 4098 vision rows for 4096
  presentation pads"); it now expects the two flanking
  `<|vision_start|>`/`<|vision_end|>` rows of every span, which upstream tags
  as vision rows too. And the visual-VAE reference condition, whose official
  seed-42 sample round-trips through FP16 on the host, is now moved onto the
  frozen device before validation exactly as FL2VA's endpoint is, instead of
  being refused for the encoder's own placement. Finally, the runtime-bound
  observer attributes the condition-VAE workspace to whichever encoder phase
  ran — Ref2VA's reference encode as well as FL2VA's endpoint encode — instead
  of failing a fully muxed Ref2VA print at the last step for lacking FL2VA's
  phase, and Ref2VA now reports its staged encoded-video and thumbnail
  capacities to that observer as FL2VA does, whose zero-byte guard names the
  offending field ([#1418](https://github.com/utensils/mold/issues/1418)).
