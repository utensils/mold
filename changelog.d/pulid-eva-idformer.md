- **PuLID's identity encoders now run in candle.** Mold ports the
  EVA02-CLIP-L-14-336 vision tower and PuLID's IDFormer resampler to pure
  Rust, parity-tested against the upstream reference on the SHA-256-pinned
  checkpoints — the IDFormer output matches to 1.5e-7 of its scale and the
  tower's CLIP projection to 1.3e-5 on a unit vector. Nothing user-facing
  changes yet; this is the encoder half of face-identity conditioning
  ([#1229](https://github.com/utensils/mold/issues/1229)).
- **The EVA02-CLIP release is converted, never loaded as a pickle.** Its
  official distribution is a torch pickle, so Mold converts it once to
  vision-only safetensors from the SHA-verified source — opened without
  following symlinks, hashed through the retained descriptor, inode-fenced
  across the read, and published through an fsynced atomic rename with the
  derived digest recorded in a sidecar. The runtime only ever loads the
  derived file ([#1229](https://github.com/utensils/mold/issues/1229)).
