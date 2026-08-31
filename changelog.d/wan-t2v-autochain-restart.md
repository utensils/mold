- **A long text-to-video Wan render is refused instead of repeating itself.**
  `mold run wan22-t2v-a14b --frames 201` chained into three stages and handed
  back roughly the same four-second clip three times, because a text-to-video
  checkpoint has no image conditioning to carry motion across a clip boundary —
  every stage re-derived the scene from the same prompt and seed. It now
  refuses up front and names both ways forward: a single continuous clip within the
  model's own budget, or an image-to-video tier, whose continuations are seeded
  with the previous clip's final frame. Image-conditioned and unclassified
  checkpoints chain exactly as before.
