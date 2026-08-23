interface StreamWaiter {
  signal: AbortSignal;
  start: (release: () => void) => void;
  cancelled: () => void;
  state: "waiting" | "active" | "done";
  release: () => void;
}

interface TargetPool {
  active: number;
  waiting: StreamWaiter[];
}

/**
 * Session-only generation streams are deliberately admitted by the client.
 * The server never sees a waiting request's media until this pool grants its
 * target a connection slot. Pools are keyed by the frozen target so one host
 * cannot consume another host's stream budget.
 */
export class TargetStreamSlots {
  private readonly pools = new Map<string, TargetPool>();

  constructor(private readonly limitPerTarget: number) {
    if (!Number.isInteger(limitPerTarget) || limitPerTarget < 1) {
      throw new RangeError(
        "stream limit per target must be a positive integer",
      );
    }
  }

  active(target: string): number {
    return this.pools.get(target)?.active ?? 0;
  }

  waiting(target: string): number {
    return this.pools.get(target)?.waiting.length ?? 0;
  }

  schedule(
    target: string,
    signal: AbortSignal,
    start: (release: () => void) => void,
  ): () => void {
    return this.enqueue(target, signal, start, () => undefined);
  }

  acquire(target: string, signal: AbortSignal): Promise<(() => void) | null> {
    if (signal.aborted) return Promise.resolve(null);
    return new Promise((resolve) => {
      let started = false;
      this.enqueue(
        target,
        signal,
        (release) => {
          started = true;
          resolve(release);
        },
        () => {
          if (!started) resolve(null);
        },
      );
    });
  }

  private enqueue(
    target: string,
    signal: AbortSignal,
    start: (release: () => void) => void,
    cancelled: () => void,
  ): () => void {
    if (signal.aborted) {
      cancelled();
      return () => undefined;
    }
    let pool = this.pools.get(target);
    if (!pool) {
      pool = { active: 0, waiting: [] };
      this.pools.set(target, pool);
    }

    const waiter = {} as StreamWaiter;
    const release = () => {
      if (waiter.state === "done") return;
      if (waiter.state === "waiting") {
        const index = pool.waiting.indexOf(waiter);
        if (index >= 0) pool.waiting.splice(index, 1);
        cancelled();
      } else {
        pool.active -= 1;
      }
      waiter.state = "done";
      signal.removeEventListener("abort", release);
      this.promote(target, pool);
      this.clean(target, pool);
    };
    Object.assign(waiter, {
      signal,
      start,
      cancelled,
      state: "waiting" as const,
      release,
    });

    signal.addEventListener("abort", release, { once: true });
    pool.waiting.push(waiter);
    this.promote(target, pool);
    return release;
  }

  private promote(target: string, pool: TargetPool): void {
    while (pool.active < this.limitPerTarget && pool.waiting.length > 0) {
      const waiter = pool.waiting.shift()!;
      if (waiter.signal.aborted) {
        waiter.release();
        continue;
      }
      waiter.state = "active";
      pool.active += 1;
      try {
        waiter.start(waiter.release);
      } catch (error) {
        waiter.release();
        throw error;
      }
    }
    this.clean(target, pool);
  }

  private clean(target: string, pool: TargetPool): void {
    if (pool.active === 0 && pool.waiting.length === 0) {
      this.pools.delete(target);
    }
  }
}
