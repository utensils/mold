- **A queued generation now says when it is preparing, and a memory shortfall
  no longer holds it forever.** `GET /api/queue` reports a job whose own
  dependency preparation is running as `preparing` — with how long it has been
  running and, for MiniMax H3's artifact authentication pass, which component
  and how far through — instead of the generic `dependency_wait` every other
  not-ready job gets; the server also logs the start and the elapsed
  completion. An H3 admission refused for device or unified memory now carries
  a typed shortfall like the host one already did, so the scheduler parks the
  job while the machine could still free that memory and refuses it with both
  numbers when it could not. Park or refuse now turns on the resource rather
  than on which worker happened to be busy at the instant admission sampled,
  and a park that outlives an idle scheduler is answered with its own
  shortfall numbers rather than waiting indefinitely on an idle GPU.
