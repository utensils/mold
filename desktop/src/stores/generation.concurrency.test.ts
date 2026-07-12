import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../lib/api/sse", () => ({ sseStream: vi.fn() }));
vi.mock("../lib/notify", () => ({
  notifyGenerated: vi.fn(),
  notifyGenerationFailed: vi.fn(),
}));

import { sseStream } from "../lib/api/sse";
import { runWithConcurrency, useGenerationStore } from "./generation";
import type { GenerateRequest } from "../lib/api/types";

describe("runWithConcurrency", () => {
  it("never exceeds the concurrency limit", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const releases: Array<() => void> = [];
    const tasks = Array.from({ length: 6 }, () => () => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      return new Promise<void>((resolve) => {
        releases.push(() => {
          inFlight--;
          resolve();
        });
      });
    });

    const done = runWithConcurrency(tasks, 2);
    await flushPromises();
    expect(inFlight).toBe(2);

    // Drain oldest-first; each release frees a slot for the next task.
    while (releases.length) {
      releases.shift()!();
      await flushPromises();
    }
    await done;

    expect(maxInFlight).toBe(2);
    expect(inFlight).toBe(0);
  });

  it("resolves results in task order regardless of completion order", async () => {
    const tasks = [
      () => new Promise<string>((r) => setTimeout(() => r("a"), 30)),
      () => Promise.resolve("b"),
      () => new Promise<string>((r) => setTimeout(() => r("c"), 5)),
    ];
    expect(await runWithConcurrency(tasks, 2)).toEqual(["a", "b", "c"]);
  });

  it("a rejected task does not stall its siblings", async () => {
    const tasks = [
      () => Promise.reject(new Error("boom")),
      () => Promise.resolve("b"),
      () => Promise.resolve("c"),
    ];
    const results = await runWithConcurrency(tasks, 2);
    expect(results[1]).toBe("b");
    expect(results[2]).toBe("c");
  });
});

describe("submitBatch connection cap", () => {
  const mockSse = vi.mocked(sseStream);
  const streams: Array<{ seed: number; resolve: () => void }> = [];

  function resolveStream(seed: number) {
    const idx = streams.findIndex((c) => c.seed === seed);
    streams[idx]!.resolve();
    streams.splice(idx, 1);
  }

  const req: GenerateRequest = {
    prompt: "a lighthouse",
    model: "flux-schnell:q8",
    width: 1024,
    height: 1024,
    steps: 4,
    seed: 100,
  };

  beforeEach(() => {
    setActivePinia(createPinia());
    streams.length = 0;
    mockSse.mockReset();
    // Each POST parks open until the test resolves it, so we can observe how
    // many held streams the batch opens at once.
    mockSse.mockImplementation((_url, opts) => {
      return new Promise<void>((resolve) => {
        streams.push({ seed: (opts.body as { seed: number }).seed, resolve });
      });
    });
  });

  afterEach(() => {
    const store = useGenerationStore();
    store.resetJobs();
  });

  it("holds at most two sibling streams open at once", async () => {
    const store = useGenerationStore();
    store.submitBatch(req, 4);
    await flushPromises();
    // Seeds 100 and 101 open; 102 and 103 wait their turn.
    expect(streams.map((s) => s.seed).sort()).toEqual([100, 101]);

    resolveStream(100);
    await flushPromises();
    expect(streams.map((s) => s.seed).sort()).toEqual([101, 102]);
  });

  it("never opens a stream for a sibling cancelled before its turn", async () => {
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(req, 4);
    await flushPromises();

    // Cancel the last sibling while the first two hold the pool.
    jobs[3]!.status = "error";
    jobs[3]!.error = "Cancelled";

    resolveStream(100);
    await flushPromises();
    resolveStream(101);
    await flushPromises();
    resolveStream(102);
    await flushPromises();
    await settled;

    const openedSeeds = mockSse.mock.calls.map((c) => (c[1].body as { seed: number }).seed);
    expect(openedSeeds).toContain(102);
    expect(openedSeeds).not.toContain(103);
  });
});
