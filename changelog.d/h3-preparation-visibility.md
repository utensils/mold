- **A queued generation now says when it is preparing, and a memory shortfall
  no longer holds it forever.** `GET /api/queue` reports a job whose own
  dependency preparation is running as `preparing` — with how long it has been
  running and, for MiniMax H3's artifact authentication pass, which component
  and how far through — instead of the generic `dependency_wait` every other
  not-ready job gets. MiniMax H3's admission additionally names and times each
  of its phases — request contract, conditioner support load, conditioning
  normalization, artifact authentication, adapter authentication, runtime
  qualification, checkpoint opens, execution-plan freeze, plus reference
  binding and any host-reclaim wait — logging both edges of every one at INFO
  with its own elapsed time, and reporting the current phase and its own age on
  the queue. An H3 admission refused for device or unified memory now carries
  a typed shortfall like the host one already did, so the scheduler parks the
  job while the machine could still free that memory and refuses it with both
  numbers when it could not. Park or refuse now turns on the resource rather
  than on which worker happened to be busy at the instant admission sampled,
  and a park that outlives an idle scheduler is answered with its own
  shortfall numbers rather than waiting indefinitely on an idle GPU.
- **MiniMax H3 no longer charges one host peak for memory it uses at two
  different moments.** The conditioner's load staging and its forward
  activation cannot coexist — the staging buffers are freed before the first
  forward allocates — so the phase peak now takes the larger rather than their
  sum, and the staging term charges one largest tensor instead of two on top of
  a parameter total that already contains it. FL2VA also pays for its own
  conditioner sequence instead of the largest canvas the family admits, using
  the same measured per-row cost and margin policy Ref2VA already used and
  clamped so the reviewed shape is unchanged. At the reviewed 1344x768 shape
  the host charge falls from 22.89 GB to 21.34 GB; smaller canvases fall
  further.
- **`GET /api/queue/:id` returns one queued job in full.** The queue listing is
  payload-free by construction — it never reads a request body per row — so a
  durably admitted job showed no settings at all until it was dispatched. The
  new endpoint reads that one body and returns the same metadata shape a
  replayed job describes itself with, plus the planner's work item for the job;
  unknown ids return `404`.
