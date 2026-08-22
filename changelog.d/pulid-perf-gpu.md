- **Face-identity conditioning is 7x faster to set up.** PuLID extraction now
  runs on the render's own GPU instead of the host, and the derived
  EVA02-CLIP tower is proven once per process instead of being re-hashed on
  every conditioned request. One extraction on an M4 Max went from 3,074 ms to
  **395 ms**; a CPU-only host still improves to 1,907 ms. Repeating the same
  reference photograph within a session is served from a small in-memory cache
  in **under 2 ms**, opening no models at all
  ([#1227](https://github.com/utensils/mold/issues/1227)).
- **Identity extraction is a scheduled phase.** It is reported as its own
  progress stage, its runtime feeds the queue's learned time estimates
  (`mold.db` schema v22), and its device memory is charged to the plan the
  scheduler admits, so a conditioned render is queued against what it actually
  needs rather than gated by a second, separate memory check
  ([#1227](https://github.com/utensils/mold/issues/1227)).
- **`pulid_face_probe bench` measures the device path.** New `--device
  cpu|metal|cuda`, a `--regress-against-full` check stated over the whole
  extraction rather than the face stack alone, and three rows decomposing the
  tower's setup cost
  ([#1227](https://github.com/utensils/mold/issues/1227)).
