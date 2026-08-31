- **Keep gallery source media reusable.** Authored images, videos, audio, masks,
  keyframes, and references now remain encrypted after a durable generation
  finishes, survive restarts and trash/restore, and are released only when the
  last owning print is permanently deleted. Authenticated clients can inspect
  availability and restore every retained role—single and collection images,
  masks, controls, audio, source/extension video, keyframes, and references—
  without exposing server filesystem paths.
