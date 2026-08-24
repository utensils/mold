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
 * Shared policy for every gallery surface. Adapters own transport and cache;
 * this class only bounds, prioritizes, deduplicates, and cancels work.
 */
export class ThumbnailScheduler {
  private readonly concurrency: number;
  private readonly perHostConcurrency: number;
  private readonly backgroundConcurrency: number;
  private readonly entries = new Map<string, ScheduledEntry<unknown>>();
  private readonly runningByHost = new Map<string, number>();
  private running = 0;
  private runningBackground = 0;
  private sequence = 0;
  private pumpQueued = false;

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
    } else if (priorityRank[request.priority] < priorityRank[entry.priority]) {
      entry.priority = request.priority;
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
          this.entries.delete(entry!.key);
          entry!.reject(abortError());
        }
      },
      setPriority: (priority) => {
        if (!active || priorityRank[priority] >= priorityRank[entry!.priority])
          return;
        entry!.priority = priority;
        this.queuePump();
      },
    };
  }

  get stats(): Readonly<{
    queued: number;
    running: number;
    background: number;
    keys: number;
  }> {
    let queued = 0;
    for (const entry of this.entries.values())
      if (entry.state === "queued") queued += 1;
    return {
      queued,
      running: this.running,
      background: this.runningBackground,
      keys: this.entries.size,
    };
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

  private nextRunnable(): ScheduledEntry<unknown> | null {
    const candidates = [...this.entries.values()]
      .filter(
        (entry) =>
          entry.state === "queued" &&
          entry.consumers > 0 &&
          (this.runningByHost.get(entry.hostKey) ?? 0) <
            this.perHostConcurrency &&
          (entry.priority !== "background" ||
            this.runningBackground < this.backgroundConcurrency),
      )
      .sort(
        (left, right) =>
          priorityRank[left.priority] - priorityRank[right.priority] ||
          left.sequence - right.sequence,
      );
    return candidates[0] ?? null;
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
