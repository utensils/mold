- **A textured 3-D render decimates before it unwraps, and takes seconds instead
  of minutes.** Tencent's own paint pipeline remeshes to 40,000 triangles before
  UV unwrapping; mold had no face budget, so the raw surface-net mesh — 226k to
  455k triangles in practice — went into xatlas whole and the `Unwrapping mesh`
  stage ran for minutes to over an hour. Measured 11-14x faster end to end
  (59.9 s to 5.4 s on a 227k-triangle mesh). An explicit `--target-faces` still
  wins, geometry-only exports keep the full-density surface, and the budget is
  advertised as `capabilities.mesh.target_faces_texture_default` so every client
  shows the number it will get. Reusing the settings of a textured print made
  before this change re-renders it at the budget, because it recorded no face
  count of its own; set `--target-faces` explicitly to reproduce the original
  density
  ([#1666](https://github.com/utensils/mold/issues/1666)).
- **UV unwrapping no longer spawns one thread per core and burns two of them in
  a spin loop.** On a 128-core host the stage used 127 threads and 3.06 cores to
  do one core's work; it now uses exactly one, with byte-identical output. The
  vendored xatlas is also compiled the way its pinned oracle compiles it
  (`-std=c++17 -O3 -DNDEBUG`), worth a further ~17%
  ([#1666](https://github.com/utensils/mold/issues/1666)).
- **`Unwrapping mesh` and a new `Simplifying mesh` stage report real progress
  and can be cancelled.** The unwrap forwards xatlas's own phase and percentage,
  which the native bridge previously received and discarded, and the chart-merge
  phase — which reported nothing and observed no cancel at all, so a cancel
  during it was silently ignored — is now polled
  ([#1666](https://github.com/utensils/mold/issues/1666)).
- **A running job whose stage has a name reports `running` rather than
  `loading`.** Long stages that carry no step counter, including `Sampling`,
  `Writing mesh` and the PBR bake, were mislabelled for their whole duration.
  `/api/activity` and `/api/queue` now ask one shared predicate, so they cannot
  disagree about the same job; weight loading and downloading still read as
  `loading`, which is what keeps their byte counters formatted as sizes
  ([#1666](https://github.com/utensils/mold/issues/1666)).
- **A finish estimate that has already passed is hidden instead of rendering as
  "Finishes in 0s".** The estimate is stamped once when a job is leased and
  never refreshed, so an overrunning job showed `0s` indefinitely
  ([#1666](https://github.com/utensils/mold/issues/1666)).
- **Known issue:** a shape whose surface is heavily non-manifold — thin,
  self-touching geometry such as an open frame or spokes — can still spend a
  long time in `Unwrapping mesh` whatever its face count, because xatlas's
  chart-merge pass rescans every chart pair after each merge. That stage now
  reports its progress and stops within seconds of a cancel, which it
  previously could not do at all
  ([#1669](https://github.com/utensils/mold/issues/1669)).
