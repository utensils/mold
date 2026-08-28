import { describe, expect, it, vi } from "vitest";
import { expectOpsUnder } from "./galleryPerfBudget";
import {
  ThumbnailScheduler,
  type ThumbnailPriority,
} from "./thumbnailScheduler";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => (resolve = done));
  return { promise, resolve };
}

async function turn(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

describe("ThumbnailScheduler", () => {
  it("deduplicates physical media while retaining independent consumers", async () => {
    const scheduler = new ThumbnailScheduler({ concurrency: 2 });
    const work = deferred<string>();
    let calls = 0;
    const request = () =>
      scheduler.schedule({
        key: "host|a.png|v1",
        hostKey: "host",
        priority: "visible",
        run: () => {
          calls += 1;
          return work.promise;
        },
      });
    const first = request();
    const second = request();
    first.cancel();
    await turn();
    expect(calls).toBe(1);
    work.resolve("url");
    await expect(second.promise).resolves.toBe("url");
  });

  it("reserves capacity by limiting background work", async () => {
    const scheduler = new ThumbnailScheduler({
      concurrency: 3,
      backgroundConcurrency: 1,
    });
    const blockers = [deferred<void>(), deferred<void>(), deferred<void>()];
    const started: string[] = [];
    const add = (key: string, priority: ThumbnailPriority, index: number) =>
      scheduler.schedule({
        key,
        hostKey: key,
        priority,
        run: () => {
          started.push(key);
          return blockers[index]!.promise;
        },
      });
    add("background-1", "background", 0);
    add("background-2", "background", 1);
    await turn();
    expect(started).toEqual(["background-1"]);
    add("visible", "visible", 2);
    await turn();
    expect(started).toEqual(["background-1", "visible"]);
    blockers.forEach(({ resolve }) => resolve());
  });

  it("drops queued work and aborts running work after its final consumer cancels", async () => {
    const scheduler = new ThumbnailScheduler({ concurrency: 1 });
    const running = deferred<void>();
    let runningSignal: AbortSignal | null = null;
    const first = scheduler.schedule({
      key: "first",
      hostKey: "host",
      priority: "visible",
      run: (signal) => {
        runningSignal = signal;
        return running.promise;
      },
    });
    const queued = scheduler.schedule({
      key: "queued",
      hostKey: "host",
      priority: "near",
      run: async () => undefined,
    });
    void first.promise.catch(() => {});
    void queued.promise.catch(() => {});
    await turn();
    queued.cancel();
    first.cancel();
    expect((runningSignal as AbortSignal | null)?.aborted).toBe(true);
    expect(scheduler.stats.queued).toBe(0);
    running.resolve();
  });

  it("promotes queued work when it becomes visible", async () => {
    const scheduler = new ThumbnailScheduler({ concurrency: 1 });
    const blocker = deferred<void>();
    const order: string[] = [];
    const active = scheduler.schedule({
      key: "active",
      hostKey: "a",
      priority: "visible",
      run: () => blocker.promise,
    });
    const older = scheduler.schedule({
      key: "older",
      hostKey: "b",
      priority: "near",
      run: async () => void order.push("older"),
    });
    void older.promise.catch(() => {});
    const promoted = scheduler.schedule({
      key: "promoted",
      hostKey: "c",
      priority: "background",
      run: async () => void order.push("promoted"),
    });
    promoted.setPriority("visible");
    await turn();
    blocker.resolve();
    await active.promise;
    await turn();
    expect(order[0]).toBe("promoted");
    older.cancel();
  });

  it("drains a 10 000-tile backlog in O(Q) dispatch work without sorting", async () => {
    const Q = 10_000;
    const scheduler = new ThumbnailScheduler({
      concurrency: 12,
      perHostConcurrency: 6,
      backgroundConcurrency: 2,
    });
    const sortSpy = vi.spyOn(Array.prototype, "sort");
    const priorities: ThumbnailPriority[] = ["visible", "near", "background"];
    const handles = [];
    let completed = 0;
    for (let i = 0; i < Q; i++) {
      handles.push(
        scheduler.schedule({
          key: `tile-${i}`,
          hostKey: `host-${i % 3}`,
          priority: priorities[i % 3]!,
          run: async () => {
            completed += 1;
          },
        }),
      );
    }
    // Every other tile scrolls into view, and every fifth leaves the window.
    for (let i = 0; i < Q; i += 2) handles[i]!.setPriority("visible");
    for (let i = 0; i < Q; i += 5) {
      void handles[i]!.promise.catch(() => {});
      handles[i]!.cancel();
    }
    // Drain: each turn completes the running batch and pumps the next.
    for (let spins = 0; spins < Q && scheduler.stats.keys > 0; spins++)
      await turn();
    sortSpy.mockRestore();

    expect(scheduler.stats.keys).toBe(0);
    expect(completed).toBe(Q - Q / 5);
    expect(sortSpy).not.toHaveBeenCalled();
    // One live slot plus at most one stale slot (the raised copy) per tile.
    expectOpsUnder(
      "scheduler dispatch scans",
      scheduler.stats.dispatchScans,
      2 * Q,
    );
  });

  it("starts a fresh generation when a cancelled key is requested again", async () => {
    const scheduler = new ThumbnailScheduler({ concurrency: 2 });
    const stale = deferred<string>();
    const first = scheduler.schedule({
      key: "same",
      hostKey: "host",
      priority: "visible",
      run: () => stale.promise,
    });
    void first.promise.catch(() => {});
    await turn();
    first.cancel();
    const second = scheduler.schedule({
      key: "same",
      hostKey: "host",
      priority: "visible",
      run: async () => "fresh",
    });
    await expect(second.promise).resolves.toBe("fresh");
    stale.resolve("stale");
    await turn();
    expect(scheduler.stats.keys).toBe(0);
  });
});
