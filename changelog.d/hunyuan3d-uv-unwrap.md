### Fixed

- **3-D:** A textured Hunyuan3D render now decimates before it unwraps, matching
  Tencent's own paint pipeline, which remeshes to 40,000 triangles before UV
  unwrapping. mold had no face budget, so the raw surface-net mesh — 226k to
  455k triangles in practice — went into xatlas whole and the `Unwrapping mesh`
  stage took minutes to over an hour. Measured 11-14x faster end to end (59.9 s
  to 5.4 s on a 227k-triangle mesh). An explicit `--target-faces` still wins,
  geometry-only exports keep the full-density surface, and the budget is
  advertised as `capabilities.mesh.target_faces_texture_default` so every client
  shows the number it will get. Reusing the settings of a textured print made
  before this change re-renders it at the budget, because it recorded no face
  count of its own; set `--target-faces` explicitly to reproduce the original
  density.
- **3-D:** UV unwrapping no longer spawns one thread per core and burns two of
  them in a spin loop. On a 128-core host the stage used 127 threads and 3.06
  cores to do one core's work; it now uses exactly one. Unwrap output is
  byte-identical. The vendored xatlas is also compiled the way its pinned
  oracle compiles it (`-std=c++17 -O3 -DNDEBUG`), worth a further ~17%.
- **3-D:** `Unwrapping mesh` and a new `Simplifying mesh` stage report real
  progress and can be cancelled. The unwrap forwards xatlas's own phase and
  percentage, which the native bridge previously received and discarded, and
  the chart-merge phase — which reported nothing and observed no cancel at all,
  so a cancel during it was silently ignored — is now polled.
- **Queue:** A running job whose stage has a name now reports `running` rather
  than `loading`. Long stages that carry no step counter, including `Sampling`,
  `Writing mesh` and the PBR bake, were mislabelled for their whole duration.
  `/api/activity` and `/api/queue` ask one shared predicate, so they can no
  longer disagree about the same job; weight loading and downloading still read
  as `loading`, which is what keeps their byte counters formatted as sizes.
- **Queue:** A finish estimate that has already passed is hidden instead of
  rendering as "Finishes in 0s". The estimate is stamped once when a job is
  leased and never refreshed, so an overrunning job showed `0s` indefinitely.

### Known issue

- A shape whose surface is heavily non-manifold — thin, self-touching geometry
  such as an open frame or spokes — can still spend a long time in
  `Unwrapping mesh` whatever its face count, because xatlas's chart-merge pass
  rescans every chart pair after each merge. That stage now reports its
  progress and stops within seconds of a cancel, which it previously could not
  do at all.
