- **Mold desktop app on the AUR.** Two new Arch packages install the desktop app
  beside any `mold-ai*` CLI package: `mold-ai-desktop-bin`, a prebuilt GPU-free
  build for remote GPU hosts, and `mold-ai-desktop`, which builds the CUDA app
  from source (`CUDA_COMPUTE_CAP=86`, `89` or `120`). Both ship a launcher
  entry, AppStream metadata and icons. Tagged releases now also attach the
  GPU-free Linux desktop archive, `mold-desktop-x86_64-unknown-linux-gnu-cpu.tar.gz`.
