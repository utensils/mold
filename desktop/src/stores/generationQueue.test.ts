import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

// Streams stay open until the test resolves them — models jobs sitting in
// the server queue while more are submitted.
const openStreams: Array<{
  onEvent: (event: string, data: string) => void;
  resolve: () => void;
}> = [];

vi.mock("../lib/api/sse", () => ({
  sseStream: vi.fn(
    (_path: string, opts: { onEvent: (event: string, data: string) => void }) =>
      new Promise<void>((resolve) => {
        openStreams.push({ onEvent: opts.onEvent, resolve });
      }),
  ),
}));
vi.mock("../lib/api/client", () => ({
  apiFetch: vi.fn(() => Promise.resolve(new Response(null, { status: 204 }))),
  apiFetchTo: vi.fn(() => Promise.resolve(new Response(null, { status: 204 }))),
  currentTarget: () => ({ baseUrl: "http://primary:7680", apiKey: null }),
  ApiError: class ApiError extends Error {
    constructor(
      message: string,
      public readonly status: number,
    ) {
      super(message);
    }
  },
}));
vi.mock("../lib/notify", () => ({
  notifyGenerated: vi.fn(),
  notifyGenerationFailed: vi.fn(),
}));

import { useGenerationStore } from "./generation";

const req = {
  prompt: "a lighthouse",
  model: "flux2-klein:q4",
  width: 1024,
  height: 1024,
  steps: 4,
};

describe("generation queueing", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    openStreams.length = 0;
  });

  it("submitting while a job runs queues a second concurrent job", () => {
    const store = useGenerationStore();
    store.submitBatch({ ...req }, 1);
    // First job starts denoising…
    openStreams[0]!.onEvent(
      "progress",
      JSON.stringify({ type: "denoise_step", step: 1, total: 4 }),
    );
    // …and a second submission with a DIFFERENT model queues behind it.
    store.submitBatch({ ...req, model: "z-image:q8" }, 1);
    openStreams[1]!.onEvent("progress", JSON.stringify({ type: "queued", position: 1, id: "b" }));

    expect(store.jobs).toHaveLength(2);
    expect(store.pending).toHaveLength(2);
    // Each job snapshots its own request.
    expect(store.jobs[0]!.model).toBe("flux2-klein:q4");
    expect(store.jobs[1]!.model).toBe("z-image:q8");
    // The canvas tracks the developing job, not the queued one.
    expect(store.active!.clientId).toBe(store.jobs[0]!.clientId);
    expect(store.jobs[1]!.queuePosition).toBe(1);
  });

  it("creates every batch sibling but holds at most two streams open", () => {
    const store = useGenerationStore();
    const { jobs } = store.submitBatch({ ...req, seed: 100 }, 3);
    expect(jobs).toHaveLength(3);
    // All three jobs exist immediately, but the connection cap keeps only two
    // SSE streams open at once so the browser's per-host budget isn't drained.
    expect(openStreams).toHaveLength(2);
    expect(jobs.map((j) => j.batchId)).toEqual([
      jobs[0]!.batchId,
      jobs[0]!.batchId,
      jobs[0]!.batchId,
    ]);
    // The active job's siblings drive the batch dots.
    expect(store.siblings).toHaveLength(3);
  });

  it("cancel marks the job cancelled and asks the server to drop it", async () => {
    const store = useGenerationStore();
    store.submitBatch({ ...req }, 1);
    openStreams[0]!.onEvent(
      "progress",
      JSON.stringify({ type: "queued", position: 1, id: "job-1" }),
    );
    await store.cancel(store.jobs[0]!.clientId);
    const { apiFetchTo } = await import("../lib/api/client");
    expect(vi.mocked(apiFetchTo)).toHaveBeenCalledWith(
      { baseUrl: "http://primary:7680", apiKey: null },
      "/api/queue/job-1",
      { method: "DELETE" },
    );
    expect(store.jobs[0]!.status).toBe("error");
    expect(store.jobs[0]!.error).toBe("Cancelled");
  });

  it("prune drops the oldest finished jobs and keeps live ones", () => {
    const store = useGenerationStore();
    store.submitBatch({ ...req }, 1);
    store.submitBatch({ ...req }, 1);
    store.submitBatch({ ...req }, 1);
    store.jobs[0]!.status = "complete";
    store.jobs[1]!.status = "error";
    store.prune(1);
    expect(store.jobs).toHaveLength(2);
    expect(store.jobs.some((j) => j.status === "queued")).toBe(true);
  });
});
