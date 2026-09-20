- **Enabled MiniMax H3 across shipped Apple Silicon surfaces.** Promoted the
  compact Metal runtime to supported, taught the shared web/desktop/mobile
  capability parser and inventory UI to accept the server's `cuda-or-metal`
  contract, and retained exact per-request memory admission for shapes that do
  not fit safely.
- **Made MiniMax H3 Metal qualification fail closed.** Added opt-in
  machine-readable phase and memory capture, exact prepared-request budget
  sidecars, a required native-allocation ceiling, allocation-free preflight,
  macOS process probes, and an external watchdog enforcing the campaign's
  availability, swap, pressure, deadline, lock, and cleanup gates. The
  instrumentation is inactive outside a campaign.
- **Reduced MiniMax H3 Metal attention residency.** Chunked dense attention now
  converts only the active query slice, transposes keys directly into their
  consumed layout, releases the fused QKV projection before attention, bounds
  each F32 score matrix at 768 MiB, and uses the fused last-dimension softmax so
  runtime retains the single probability matrix its workspace prices. CUDA
  dispatch and attention arithmetic remain unchanged.
