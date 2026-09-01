- **Keep queued generations moving through harmless GPU telemetry changes.**
  Multiple prepared MiniMax H3 jobs no longer leave an idle GPU stuck in a
  high-CPU planning loop when CUDA context memory changes between planning and
  dispatch; genuinely insufficient VRAM and changed execution plans still stop
  the grant.
