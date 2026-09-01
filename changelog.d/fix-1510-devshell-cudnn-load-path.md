- **Devshell builds with `cudnn` now run.** The Nix devshell left cuDNN off
  `LD_LIBRARY_PATH`, and a binary built in the shell from the shipping Linux
  feature set carries no RUNPATH, so it linked fine and then died at startup
  with `libcudnn.so.9: cannot open shared object file`. Both `LIBRARY_PATH` and
  `LD_LIBRARY_PATH` now carry cuDNN, and a new `devshell-cuda-load-path` flake
  check holds the devshell's own advertised feature set to being runnable in it
  ([#1510](https://github.com/utensils/mold/issues/1510)).
