- **Account for unified memory in Metal server admission.** Queues and placement
  previews include CPU-parked encoders and concurrent host allocations in the
  device budget. Chain stages retain consistent lease accounting, and server
  memory samples use the same available-memory authority as worker preflight
  ([#1059](https://github.com/utensils/mold/issues/1059)).
