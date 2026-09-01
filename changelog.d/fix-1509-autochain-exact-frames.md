- **`--frames N` past one clip now renders N frames, not a whole extra
  clip's worth.** An auto-chained long video used to round the request up to
  whole clips and silently render the overshoot (`--frames 145` on
  `wan22-ti2v-5b` rendered 241 frames, +66% GPU time); the last stage is now
  exact-fitted to the requested total on every surface, and the CLI
  discloses a total the frame lattice cannot land on exactly before
  rendering ([#1509](https://github.com/utensils/mold/issues/1509)).
