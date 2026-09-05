### Fixed

- Metal server queues and placement previews account for CPU-parked encoders and concurrent host allocations in the unified memory budget. Chain stages retain consistent lease accounting, and server memory samples use the same available-memory authority as worker preflight.
