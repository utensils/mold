- **MiniMax H3 placement recovers host memory and starts faster after restart.**
  Private admission now unloads Mold's idle model cache before refusing a print,
  while centralized artifact attestations avoid re-hashing unchanged model files
  on every process start without weakening pinned-digest verification.
