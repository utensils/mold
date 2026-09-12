- **LTX-2 IC-LoRA control renders work at the tier's default resolution.** A
  `ref0.5` control adapter (union, motion-track) conditions on a reference video at
  half the conditioned stage's size, and that half must still land on the video
  VAE's 32 px latent grid. `ltx-2.3-22b-distilled:fp8` defaults to 1216x704, whose
  stage-1 grid is an odd 19x11, so `--ic-lora-control union` with no explicit
  `--width/--height` failed inside the VAE — after paying for the full text encode —
  with a reshape mismatch. Admission now snaps such a render down onto the grid the
  reference can be encoded on (1216x704 becomes 1152x640) and says so, on the
  server and under `--local` alike; the engine's own guard refuses an unsnapped
  canvas by name instead of dying in a tensor reshape.
