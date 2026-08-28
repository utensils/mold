export type ThumbnailPriority = "visible" | "near" | "background";

export interface ThumbnailRequest<T> {
  /** Immutable physical-media identity, including host and content version. */
  key: string;
  hostKey: string;
  priority: ThumbnailPriority;
  run: (signal: AbortSignal) => Promise<T>;
}

export interface ThumbnailHandle<T> {
  promise: Promise<T>;
  cancel: () => void;
  setPriority: (priority: ThumbnailPriority) => void;
}

interface ScheduledEntry<T> {
  key: string;
  hostKey: string;
  priority: ThumbnailPriority;
  sequence: number;
  consumers: number;
  state: "queued" | "running";
  controller: AbortController;
  run: (signal: AbortSignal) => Promise<T>;
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
}

export interface ThumbnailSchedulerOptions {
  concurrency?: number;
  perHostConcurrency?: number;
  backgroundConcurrency?: number;
}

const PRIORITIES: readonly ThumbnailPriority[] = [
  "visible",
  "near",
  "background",
];

const priorityRank: Record<ThumbnailPriority, number> = {
  visible: 0,
  near: 1,
  background: 2,
};

function abortError(): Error {
  if (typeof DOMException !== "undefined")
    return new DOMException("Thumbnail cancelled", "AbortError");
  const error = new Error("Thumbnail cancelled");
  error.name = "AbortError";
  return error;
}

/**
 * FIFO with an advancing head, so dequeue is O(1) and the array is compacted
 * only once the dead prefix outweighs the live tail.
 */
class Fifo<T> {
  private items: T[] = [];
  private head = 0;

  push(item: T): void {
    this.items.push(item);
  }

  shift(): T | undefined {
    if (this.head >= this.items.length) return undefined;
    const item = this.items[this.head];
    this.items[this.head] = undefined as unknown as T;
    this.head += 1;
    if (this.head > 1024 && this.head * 2 > this.items.length) {
      this.items = this.items.slice(this.head);
      this.head = 0;
    }
    return item;
  }

  get size(): number {
    return this.items.length - this.head;
  }
}

/**
 * Shared policy for every gallery surface. Adapters own transport and cache;
 * this class only bounds, prioritizes, deduplicates, and cancels work.
 *
 * Dispatch is O(hosts): one FIFO per (priority, host), visited highest
 * priority first and round-robin across hosts with a free slot. Entries whose
 * priority was raised or that were cancelled while queued are skipped lazily
 * when they surface, so nothing ever re-sorts the queue — the previous
 * implementation copied and sorted every queued entry per dispatch, which
 * made draining Q tiles O(Q² log Q).
 */
export class ThumbnailScheduler {
  private readonly concurrency: number;
  private readonly perHostConcurrency: number;
  private readonly backgroundConcurrency: number;
  private readonly entries = new Map<string, ScheduledEntry<unknown>>();
  private readonly runningByHost = new Map<string, number>();
  /** priority → host → FIFO of queued entries (lazily invalidated). */
  private readonly queues: Record<
    ThumbnailPriority,
    Map<string, Fifo<ScheduledEntry<unknown>>>
  > = { visible: new Map(), near: new Map(), background: new Map() };
  private running = 0;
  private runningBackground = 0;
  private sequence = 0;
  private pumpQueued = false;
  private dispatchScans = 0;

  constructor(options: ThumbnailSchedulerOptions = {}) {
    this.concurrency = Math.max(1, options.concurrency ?? 12);
    this.perHostConcurrency = Math.max(1, options.perHostConcurrency ?? 6);
    this.backgroundConcurrency = Math.max(
      0,
      Math.min(this.concurrency, options.backgroundConcurrency ?? 2),
    );
  }

  schedule<T>(request: ThumbnailRequest<T>): ThumbnailHandle<T> {
    let entry = this.entries.get(request.key) as ScheduledEntry<T> | undefined;
    if (entry && (entry.controller.signal.aborted || entry.consumers <= 0)) {
      if (this.entries.get(request.key) === entry)
        this.entries.delete(request.key);
      entry = undefined;
    }
    if (!entry) {
      let resolve!: (value: T) => void;
      let reject!: (error: unknown) => void;
      const promise = new Promise<T>((done, failed) => {
        resolve = done;
        reject = failed;
      });
      entry = {
        ...request,
        sequence: this.sequence++,
        consumers: 0,
        state: "queued",
        controller: new AbortController(),
        promise,
        resolve,
        reject,
      };
      this.entries.set(request.key, entry as ScheduledEntry<unknown>);
      this.enqueue(entry as ScheduledEntry<unknown>);
    } else if (priorityRank[request.priority] < priorityRank[entry.priority]) {
      this.raise(entry as ScheduledEntry<unknown>, request.priority);
    }
    entry.consumers += 1;
    this.queuePump();

    let active = true;
    return {
      promise: entry.promise,
      cancel: () => {
        if (!active) return;
        active = false;
        entry!.consumers -= 1;
        if (entry!.consumers > 0) return;
        entry!.controller.abort();
        if (entry!.state === "queued") {
          // The FIFO slot stays behind and is skipped when it surfaces.
          this.entries.delete(entry!.key);
          entry!.reject(abortError());
        }
      },
      setPriority: (priority) => {
        if (!active || priorityRank[priority] >= priorityRank[entry!.priority])
          return;
        this.raise(entry! as ScheduledEntry<unknown>, priority);
        this.queuePump();
      },
    };
  }

  get stats(): Readonly<{
    queued: number;
    running: number;
    background: number;
    keys: number;
    /** FIFO slots examined by dispatch so far — the perf guard's budget. */
    dispatchScans: number;
  }> {
    let queued = 0;
    for (const entry of this.entries.values())
      if (entry.state === "queued") queued += 1;
    return {
      queued,
      running: this.running,
      background: this.runningBackground,
      keys: this.entries.size,
      dispatchScans: this.dispatchScans,
    };
  }

  private enqueue(entry: ScheduledEntry<unknown>): void {
    const byHost = this.queues[entry.priority];
    let fifo = byHost.get(entry.hostKey);
    if (!fifo) {
      fifo = new Fifo();
      byHost.set(entry.hostKey, fifo);
    }
    fifo.push(entry);
  }

  /** Re-file a queued entry under a higher priority; the old slot goes stale. */
  private raise(
    entry: ScheduledEntry<unknown>,
    priority: ThumbnailPriority,
  ): void {
    entry.priority = priority;
    if (entry.state === "queued") this.enqueue(entry);
  }

  private queuePump(): void {
    if (this.pumpQueued) return;
    this.pumpQueued = true;
    queueMicrotask(() => {
      this.pumpQueued = false;
      this.pump();
    });
  }

  private pump(): void {
    while (this.running < this.concurrency) {
      const next = this.nextRunnable();
      if (!next) return;
      this.start(next);
    }
  }

  /** A slot is live only while it is still the entry's current filing. */
  private isLive(
    entry: ScheduledEntry<unknown>,
    priority: ThumbnailPriority,
  ): boolean {
    return (
      entry.state === "queued" &&
      entry.consumers > 0 &&
      entry.priority === priority &&
      this.entries.get(entry.key) === entry
    );
  }

  private nextRunnable(): ScheduledEntry<unknown> | null {
    for (const priority of PRIORITIES) {
      if (
        priority === "background" &&
        this.runningBackground >= this.backgroundConcurrency
      ) {
        continue;
      }
      for (const [hostKey, fifo] of this.queues[priority]) {
        if ((this.runningByHost.get(hostKey) ?? 0) >= this.perHostConcurrency)
          continue;
        for (;;) {
          const entry = fifo.shift();
          if (!entry) {
            this.queues[priority].delete(hostKey);
            break;
          }
          this.dispatchScans += 1;
          if (this.isLive(entry, priority)) return entry;
        }
      }
    }
    return null;
  }

  private start(entry: ScheduledEntry<unknown>): void {
    entry.state = "running";
    this.running += 1;
    if (entry.priority === "background") this.runningBackground += 1;
    this.runningByHost.set(
      entry.hostKey,
      (this.runningByHost.get(entry.hostKey) ?? 0) + 1,
    );
    const startedAsBackground = entry.priority === "background";
    void entry
      .run(entry.controller.signal)
      .then(entry.resolve, entry.reject)
      .finally(() => {
        if (this.entries.get(entry.key) === entry)
          this.entries.delete(entry.key);
        this.running -= 1;
        if (startedAsBackground) this.runningBackground -= 1;
        const hostRunning = (this.runningByHost.get(entry.hostKey) ?? 1) - 1;
        if (hostRunning > 0) this.runningByHost.set(entry.hostKey, hostRunning);
        else this.runningByHost.delete(entry.hostKey);
        this.queuePump();
      });
  }
}

export const galleryThumbnailScheduler = new ThumbnailScheduler();
