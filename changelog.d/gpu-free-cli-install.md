### Fixed

- Linux clients can install and update Mold without an NVIDIA GPU or CUDA libraries. Stable and nightly releases now include a GPU-free CLI archive; the installer selects it when no GPU is available and supports `MOLD_BACKEND=cpu` for explicit remote-only use.
- `mold-ai-bin` on AUR now installs the GPU-free CLI instead of the CUDA-linked SM89 archive, avoiding missing `libcudart.so.12` at startup. Existing users needing local GPU generation should switch to the `mold-ai` / `mold-ai-git` source packages or a matching CUDA release archive.
