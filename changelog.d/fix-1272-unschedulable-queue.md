- **A queued print is never silently abandoned.** A generation whose execution
  plan could not be resolved used to be reported as an untyped
  `no_schedulable_device` and retried forever with the real reason discarded, so
  an H3 print sat `queued` on an idle RTX 4090 past its client's forty-minute
  timeout with no failure and no message. The scheduler still retries — a job
  waiting behind running work is untouched and is dispatched the moment capacity
  returns — but once nothing is leased or preparing it settles the job after a
  grace window with the plan's own named shortfall instead of leaving it queued
  with no explanation. A MiniMax H3 job blocked on host RAM is also now
  identified as a host-memory shortfall, naming required and available bytes,
  rather than being filed as a VRAM shortfall it did not have
  ([#1272](https://github.com/utensils/mold/issues/1272)).
