- **Durable batch queues stay responsive at any depth.** Heterogeneous batch
  admission now commits directly to SQLite and a capacity-bounded feeder
  hydrates work only as scheduler room becomes available, preserving FIFO,
  cancellation, restart recovery, and idempotent retries without one waiting
  task per child.
