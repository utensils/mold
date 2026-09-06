# xatlas

Unmodified `source/xatlas/xatlas.cpp`, `source/xatlas/xatlas.h` and `LICENSE` from
[jpcy/xatlas](https://github.com/jpcy/xatlas) revision
`f700c7790aaa030e794b52ba7791a05c085faf0c`. This is the submodule used by
`mworchel/xatlas-python` v0.0.9 (`ff6541ec7ed1b5131dd7bf5447da6174e6c82621`),
the version pinned by the Hunyuan3D 2.1 executable oracle.

The `mesh-texture` feature compiles this CPU UV library through a narrow C ABI.
It does not add a Python dependency. Source is vendored for offline Nix builds.
`bridge.cpp` is mold-owned glue; the upstream implementation is unchanged.
