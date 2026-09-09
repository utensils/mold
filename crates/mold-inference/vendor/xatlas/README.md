# xatlas

`source/xatlas/xatlas.cpp`, `source/xatlas/xatlas.h` and `LICENSE` from
[jpcy/xatlas](https://github.com/jpcy/xatlas) revision
`f700c7790aaa030e794b52ba7791a05c085faf0c`. This is the submodule used by
`mworchel/xatlas-python` v0.0.9 (`ff6541ec7ed1b5131dd7bf5447da6174e6c82621`),
the version pinned by the Hunyuan3D 2.1 executable oracle.

The `mesh-texture` feature compiles this CPU UV library through a narrow C ABI.
It does not add a Python dependency. Source is vendored for offline Nix builds.
`bridge.cpp` is mold-owned glue. `xatlas.h` and `LICENSE` are unmodified.

## Build flags

`crates/mold-inference/build.rs` compiles with `-std=c++17 -O3 -DNDEBUG`, which
is what `CMAKE_CXX_STANDARD 17` plus `CMAKE_BUILD_TYPE=Release` gives the
pinned oracle. Compiling without `NDEBUG` leaves all 155 `XA_DEBUG_ASSERT`
sites in — a measured ~17% of the unwrap, and a divergence from the revision
this file pins.

It also compiles with `-DXA_MULTITHREADED=0`, upstream's own switch
(`xatlas.cpp:71-73`) for the inline scheduler at `:3304-3400`. The threaded
`TaskScheduler` sizes its pool at `hardware_concurrency() - 1` with no cap and
waits in a bare `std::this_thread::yield()` loop (`:3235-3236`). Mold submits
one mesh, so one connected shape is one chart group and one worker ever has
work: measured on a 128-core host, 127 threads and 3.06 cores consumed against
1.00 core here, for 7% more wall clock and byte-identical output.

## Divergence from upstream: `xatlas.cpp` only

`mold-cancellable-merge.patch` is the complete diff against the pinned
revision, and is the file to re-apply when the pin moves — it applies with
`patch -p1` (or `git apply`) from the repository root against a pristine
`xatlas.cpp` and reproduces this file exactly. Every hunk is marked
`MOLD DIVERGENCE` in the source. Line numbers quoted in this file and in
`build.rs` are the PINNED REVISION's, so they are what the patch and any
upstream comparison refer to; they sit 14 lines earlier than the same code in
the patched file.

It makes `segment::ClusteredCharts::mergeCharts` cancellable, and nothing else.
That phase rescans every chart pair after each successful merge, and each
candidate pair is re-parameterized and grid-intersected; upstream neither
advances a counter through it nor polls for cancellation. On a non-manifold
shape — which the surface-net mesher produces from thin, self-touching geometry
— it runs for minutes to hours with the progress callback never invoked, so a
user's cancel is never observed (#1666). `Progress::poll()` re-fires the
callback at the percent already reported, because `update()` fires only when
the whole percent changes and therefore never writes `cancel` during such a
phase; `mergeCharts` calls it once per rescan and every 1024 charts within one.

This is scheduling only. When nothing cancels, every merge decision and every
output UV is unchanged — pinned by `uv::tests` (the 1e-7 xatlas-python oracle
and the determinism assertion) and by byte-identical unwrap output across the
patched and unpatched builds on retained 226k-, 264k- and 351k-triangle
Hunyuan3D meshes.
