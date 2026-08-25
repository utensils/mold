- **Truthful generation progress.** MiniMax and other multi-stage renders now keep
  showing their real denoising step while transformer blocks stream, then name
  video/audio decoding, encoding, muxing, and saving as finalization instead of
  appearing stuck at a premature 100%.
