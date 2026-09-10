- **Queue-aware Auto routing.** Web, desktop, iPhone, and Android now account for
  schedulable GPUs, active work, and the incoming batch size when choosing a
  machine, keeping busy multi-GPU servers available for queued work.
- **Send held jobs to another machine.** Queue details offer a destination picker
  when another connected machine is available. The terminal Machines queue and
  `mold queue send JOB-ID --to HOST` also support transfers. Original settings and
  reference media are preserved, retries recover the same destination job, and
  the held original is removed only after durable acceptance. Machine-local
  adapters and independent workflow stages are refused rather than changed.
