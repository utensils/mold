- **MiniMax H3 says which models this build can actually run, before you
  download them.** `/api/models` now derives `runtime_available` from the same
  authority generation consults — the build's own engine, the task partition,
  and the checkpoint's weight layout — and carries a new
  `runtime_unavailable_reason` naming the obstacle. So an RTX 3090/A40, B200,
  RTX 50-series, or Windows build no longer advertises H3 as runnable, Ref2VA
  is honest that it executes on no released binary, and web, desktop, and iPhone
  show a "Download only" badge and the server's own sentence on Discover rows
  and detail panes rather than after a 21-42 GB pull. Generation refuses with
  the same sentence over HTTP 501 (`MINIMAX_H3_RUNTIME_UNAVAILABLE`, never the
  compliance 451), `mold pull` prints the reason instead of a `mold run` hint
  it would refuse, and `mold run --local` stops before loading any weights.
  Downloading, verifying, inventorying, repairing, and removing every H3 model
  is unchanged ([#1276](https://github.com/utensils/mold/issues/1276)).
