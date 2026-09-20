- **AUR packages no longer execute the CUDA-linked binary to generate shell
  completions.** `mold-ai-bin 0.30.1` failed inside `package()` with
  `libcudart.so.12: cannot open shared object file` because the recipe ran
  `mold completions` under fakeroot. The Linux CUDA release archives now ship
  the bash, zsh and fish completion scripts beside the binary
  (`completions/mold.bash`, `completions/_mold`, `completions/mold.fish`),
  generated at release time from that exact binary and verified against it;
  `mold-ai-bin` installs those files, the source recipes generate theirs in
  `build()` against the toolkit's library directory, and
  `scripts/aur/test-in-docker.sh` now creates the `mold-ai-bin` package with
  no CUDA library present before installing anything. The prebuilt archive
  still links the CUDA 12 runtime while Arch's `extra/cuda` ships CUDA 13, so
  the installed binary needs a CUDA 12 runtime to load
  ([#1742](https://github.com/utensils/mold/issues/1742)).
