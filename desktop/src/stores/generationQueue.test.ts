import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises } from "@vue/test-utils";

// Streams stay open until the test resolves them — models jobs sitting in
// the server queue while more are submitted.
const openStreams: Array<{
  onOpen: ((response: Response) => void) | undefined;
  onEvent: (event: string, data: string) => void;
  signal: AbortSignal;
  onClose: ((error: Error | null) => void) | undefined;
  resolve: () => void;
}> = [];

vi.mock("../lib/api/sse", () => ({
  sseStream: vi.fn(
    (
      _path: string,
      opts: {
        onOpen?: (response: Response) => void;
        onEvent: (event: string, data: string) => void;
        signal: AbortSignal;
        onClose?: (error: Error | null) => void;
      },
    ) =>
      new Promise<void>((resolve) => {
        openStreams.push({
          onOpen: opts.onOpen,
          onEvent: opts.onEvent,
          signal: opts.signal,
          onClose: opts.onClose,
          resolve,
        });
      }),
  ),
}));
vi.mock("../lib/api/client", () => ({
  apiFetch: vi.fn(() => Promise.resolve(new Response(null, { status: 204 }))),
  // A sequence is CREATED through `POST /api/chain-jobs` before its event
  // stream opens; every other call keeps the plain 204.
  apiFetchTo: vi.fn((_target: unknown, path: string) =>
    Promise.resolve(
      path === "/api/chain-jobs"
        ? new Response(JSON.stringify({ job_id: "chain-job-1" }))
        : new Response(null, { status: 204 }),
    ),
  ),
  // Reconciliation asks the host what really happened to a dead stream: an
  // empty queue and an empty gallery mean the print is genuinely gone.
  apiJsonTo: vi.fn((_target: unknown, path: string) =>
    Promise.resolve(
      path === "/api/queue"
        ? { entries: [] }
        : // Sequence recovery asks for the ephemeral chain record; an empty
          // listing is what proves the submission never landed, and a detail
          // that finalized nothing is what proves it produced no print.
          path.startsWith("/api/chain-jobs?")
          ? { jobs: [] }
          : path.startsWith("/api/chain-jobs/")
            ? { state: "complete", finalizes: [] }
            : [],
    ),
  ),
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

// A print is admitted through the durable queue and never opens a held
// stream, so every stream-lifecycle case below rides an auto-chained SEQUENCE
// — the one submission shape `streamJob` still serves.
const req = {
  prompt: "a lighthouse",
  model: "ltx-2-19b-distilled:fp8",
  width: 1024,
  height: 1024,
  steps: 4,
  frames: 241,
};

const chainDecision = {
  kind: "chain" as const,
  clipFrames: 97,
  motionTail: 17,
  stageCount: 3,
};

/** A chain completion carries the stitched clip, not a still. */
/** A sequence finishes with a saved FILENAME on `finalized` — never bytes. */
function chainComplete(overrides: { seed?: number; video?: string; format?: string } = {}): string {
  return JSON.stringify({
    type: "finalized",
    output: "final/output-1.mp4",
    gallery_filename: `sequence-${overrides.seed ?? 1}.${overrides.format ?? "mp4"}`,
  });
}

describe("generation queueing", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    openStreams.length = 0;
    vi.clearAllMocks();
  });

  afterEach(async () => {
    useGenerationStore().resetJobs();
    // A prune test can intentionally remove a job whose mocked SSE promise is
    // still held open, so it is no longer reachable through resetJobs(). Drain
    // every transport to release the module-level per-host stream permits.
    for (const stream of openStreams) stream.resolve();
    await flushPromises();
  });

  it("submitting while a job runs queues a second concurrent job", async () => {
    const store = useGenerationStore();
    store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();
    // First job starts denoising…
    openStreams[0]!.onEvent(
      "chain_job",
      JSON.stringify({ type: "denoise_step", stage_idx: 0, step: 1, total: 4 }),
    );
    // …and a second submission with a DIFFERENT model queues behind it.
    store.submitBatch({ ...req, model: "z-image:q8" }, 1, null, chainDecision);
    await flushPromises();

    expect(store.jobs).toHaveLength(2);
    expect(store.pending).toHaveLength(2);
    // Each job snapshots its own request.
    expect(store.jobs[0]!.model).toBe(req.model);
    expect(store.jobs[1]!.model).toBe("z-image:q8");
    // The canvas tracks the developing job, not the queued one.
    expect(store.active!.clientId).toBe(store.jobs[0]!.clientId);
    // A sequence's server identity is the chain job the create minted.
    expect(store.jobs[1]!.id).toBe("chain-job-1");
  });

  it("reports admission only after the machine accepts the sequence", async () => {
    // A sequence is accepted by `POST /api/chain-jobs`, not by a stream
    // opening: that POST is the moment the machine owns the work.
    const { apiFetchTo } = await import("../lib/api/client");
    let releaseCreate!: (value: Response) => void;
    vi.mocked(apiFetchTo).mockImplementationOnce(
      () => new Promise<Response>((resolve) => (releaseCreate = resolve)),
    );
    const batch = useGenerationStore().submitBatch({ ...req }, 1, null, chainDecision);
    let admitted = false;
    void batch.admitted!.then(() => {
      admitted = true;
    });
    await flushPromises();

    expect(admitted).toBe(false);
    releaseCreate(new Response(JSON.stringify({ job_id: "chain-job-1" })));
    await flushPromises();

    expect(admitted).toBe(true);
  });

  it("keeps an explicitly selected older job on the canvas", () => {
    const store = useGenerationStore();
    const older = store.startJob({ ...req, prompt: "older prompt" });
    const newer = store.startJob({ ...req, prompt: "newer prompt" });
    expect(store.active?.clientId).toBe(newer.clientId);

    store.select(older.clientId);
    expect(store.active?.clientId).toBe(older.clientId);
    expect(store.active?.request?.prompt).toBe("older prompt");

    store.select(null);
    expect(store.active?.clientId).toBe(newer.clientId);
  });

  it("returns the canvas to automatic active work when a new batch is submitted", async () => {
    const store = useGenerationStore();
    const inspected = store.startJob({ ...req, prompt: "inspected" });
    store.select(inspected.clientId);
    const { jobs } = store.submitBatch({ ...req, prompt: "new work" }, 1, null, chainDecision);
    await flushPromises();

    expect(store.selectedClientId).toBeNull();
    expect(store.active?.clientId).toBe(jobs[0]!.clientId);
  });

  it("creates every batch sibling and opens enough streams to fill a four-GPU host", async () => {
    const store = useGenerationStore();
    const { jobs } = store.submitBatch({ ...req, seed: 100 }, 5, null, chainDecision);
    await flushPromises();
    expect(jobs).toHaveLength(5);
    expect(openStreams).toHaveLength(4);
    expect(jobs.map((j) => j.batchId)).toEqual(Array(5).fill(jobs[0]!.batchId));
    // The active job's siblings drive the batch dots.
    expect(store.siblings).toHaveLength(5);
  });

  it("cancel marks the job cancelled and asks the server to drop it", async () => {
    const store = useGenerationStore();
    store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();
    await store.cancel(store.jobs[0]!.clientId);
    const { apiFetchTo } = await import("../lib/api/client");
    // The id comes from the create response, not from a progress frame.
    expect(vi.mocked(apiFetchTo)).toHaveBeenCalledWith(
      { baseUrl: "http://primary:7680", apiKey: null },
      "/api/chain-jobs/chain-job-1/cancel",
      { method: "POST" },
    );
    expect(store.jobs[0]!.status).toBe("error");
    expect(store.jobs[0]!.error).toBe("Cancelled");
  });

  it("removes exactly one settled row and refuses live or interrupted work", () => {
    const store = useGenerationStore();
    const cancelled = store.startJob({ ...req, prompt: "cancelled" });
    const live = store.startJob({ ...req, prompt: "live" });
    cancelled.status = "error";
    cancelled.error = "Cancelled";
    store.select(cancelled.clientId);

    expect(store.removeSettled(live.clientId)).toBe(false);
    cancelled.interrupted = true;
    expect(store.removeSettled(cancelled.clientId)).toBe(false);
    cancelled.interrupted = false;
    expect(store.removeSettled(cancelled.clientId)).toBe(true);
    expect(store.jobs.map((job) => job.clientId)).toEqual([live.clientId]);
    expect(store.selectedClientId).toBeNull();
  });

  it("repaints as cancelling before a running Wan request is acknowledged", async () => {
    const store = useGenerationStore();
    const { jobs } = store.submitBatch(
      { ...req, model: "wan22-i2v-a14b:q4" },
      1,
      null,
      chainDecision,
    );
    await flushPromises();
    openStreams[0]!.onEvent(
      "chain_job",
      JSON.stringify({
        type: "denoise_step",
        stage_idx: 0,
        step: 1,
        total: 28,
        job_id: "wan-running",
      }),
    );
    jobs[0]!.id = "wan-running";
    const { apiFetchTo } = await import("../lib/api/client");
    let acknowledge!: () => void;
    vi.mocked(apiFetchTo).mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          acknowledge = () => resolve(new Response(null, { status: 204 }));
        }),
    );

    const cancellation = store.cancel(jobs[0]!.clientId);
    expect(jobs[0]!.cancelling).toBe(true);
    expect(jobs[0]!.status).toBe("denoising");

    acknowledge();
    await cancellation;
    expect(jobs[0]).toMatchObject({ cancelling: false, status: "error", error: "Cancelled" });
  });

  it("keeps a running job and its stream alive when the server refuses cancellation", async () => {
    const store = useGenerationStore();
    const { jobs } = store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();
    openStreams[0]!.onEvent(
      "chain_job",
      JSON.stringify({
        type: "denoise_step",
        stage_idx: 0,
        step: 1,
        total: 4,
        job_id: "running-job",
      }),
    );
    jobs[0]!.id = "running-job";
    const { apiFetchTo, ApiError } = await import("../lib/api/client");
    vi.mocked(apiFetchTo).mockRejectedValueOnce(
      new ApiError("queue job running-job is already running", 409),
    );

    await expect(store.cancel(jobs[0]!.clientId)).rejects.toThrow("already running");

    expect(openStreams[0]!.signal.aborted).toBe(false);
    expect(jobs[0]).toMatchObject({ status: "denoising", error: null, cancelling: false });
  });

  it("keeps a server cancellation frame classified as cancellation during DELETE", async () => {
    const store = useGenerationStore();
    const { settled } = store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();
    openStreams[0]!.onEvent(
      "chain_job",
      JSON.stringify({ type: "chain_start", stage_count: 3, job_id: "job-race" }),
    );
    const { apiFetchTo } = await import("../lib/api/client");
    vi.mocked(apiFetchTo).mockImplementationOnce(async () => {
      openStreams[0]!.onEvent(
        "chain_job",
        JSON.stringify({ type: "state_changed", state: "cancelled" }),
      );
      return new Response(null, { status: 204 });
    });

    await store.cancel(store.jobs[0]!.clientId);
    openStreams[0]!.resolve();
    await settled;

    expect(store.jobs[0]).toMatchObject({ status: "error", error: "Cancelled" });
    const { notifyGenerationFailed } = await import("../lib/notify");
    expect(vi.mocked(notifyGenerationFailed)).not.toHaveBeenCalled();
  });

  it("ignores buffered progress and completion frames after local cancellation", async () => {
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();
    openStreams[0]!.onEvent(
      "chain_job",
      JSON.stringify({ type: "chain_start", stage_count: 3, job_id: "job-buffered" }),
    );

    await store.cancel(jobs[0]!.clientId);
    expect(openStreams[0]!.signal.aborted).toBe(true);
    openStreams[0]!.onEvent(
      "chain_job",
      JSON.stringify({ type: "denoise_step", stage_idx: 0, step: 4, total: 4 }),
    );
    openStreams[0]!.onEvent("chain_job", chainComplete({ seed: 5 }));
    openStreams[0]!.resolve();
    await settled;

    expect(jobs[0]).toMatchObject({ status: "error", error: "Cancelled", result: null });
    expect(jobs[0]!.resultUrl).toBeNull();
  });

  it("ignores buffered completion frames after the store resets", async () => {
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();

    store.resetJobs();
    expect(openStreams[0]!.signal.aborted).toBe(true);
    openStreams[0]!.onEvent("chain_job", chainComplete({ seed: 6 }));
    openStreams[0]!.resolve();
    await settled;

    expect(jobs[0]).toMatchObject({ status: "error", error: "Cancelled", result: null });
    expect(store.jobs).toHaveLength(0);
  });

  it("preserves a valid completion when DELETE subsequently fails", async () => {
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();
    const { apiFetchTo } = await import("../lib/api/client");
    vi.mocked(apiFetchTo).mockImplementationOnce(async () => {
      openStreams[0]!.onEvent("chain_job", chainComplete({ seed: 12 }));
      openStreams[0]!.resolve();
      throw new Error("DELETE connection reset");
    });

    await expect(store.cancel(jobs[0]!.clientId)).resolves.toBe(false);
    await settled;
    expect(jobs[0]).toMatchObject({
      status: "complete",
      error: null,
      result: { filename: "sequence-12.mp4" },
    });
  });

  it("keeps the local stream when remote cancellation cannot be confirmed", async () => {
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();
    openStreams[0]!.onEvent(
      "chain_job",
      JSON.stringify({ type: "chain_start", stage_count: 3, job_id: "job-offline" }),
    );
    const { apiFetchTo } = await import("../lib/api/client");
    vi.mocked(apiFetchTo).mockRejectedValueOnce(new Error("host went offline"));

    await expect(store.cancel(jobs[0]!.clientId)).rejects.toThrow("host went offline");
    expect(openStreams[0]!.signal.aborted).toBe(false);
    expect(jobs[0]).toMatchObject({ status: "queued", error: null });
    openStreams[0]!.resolve();
    await settled;
  });

  it("preserves a terminal server failure that races the cancellation request", async () => {
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();
    const { apiFetchTo } = await import("../lib/api/client");
    vi.mocked(apiFetchTo).mockImplementationOnce(async () => {
      openStreams[0]!.onEvent(
        "chain_job",
        JSON.stringify({
          type: "state_changed",
          state: "failed",
          error: "GPU ran out of memory while loading the model",
        }),
      );
      return new Response(null, { status: 204 });
    });

    await store.cancel(jobs[0]!.clientId);
    openStreams[0]!.resolve();
    await settled;

    expect(jobs[0]).toMatchObject({
      status: "error",
      error: "GPU ran out of memory while loading the model",
    });
    expect(jobs[0]!.error).not.toBe("Cancelled");
  });

  it("ignores a malformed frame and settles on the stream's own close", async () => {
    // A malformed frame carries no authority — the chain-job stream retries
    // and its terminal frame settles. A frame that never parses must not be
    // read as a failure of the render itself.
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();

    openStreams[0]!.onEvent("chain_job", "{not-json");
    expect(jobs[0]!.status).not.toBe("error");
    openStreams[0]!.resolve();
    await settled;

    // It ends unfinished rather than claiming a result it never received.
    expect(jobs[0]!.status).toBe("error");
    expect(jobs[0]!.result).toBeNull();
  });

  it("never completes a sequence the machine finalized without a saved file", async () => {
    // `finalized` carries the filename the print was saved under. Without one
    // there is nothing to show, so the row must not read as complete.
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();

    openStreams[0]!.onEvent("chain_job", JSON.stringify({ type: "finalized" }));
    openStreams[0]!.resolve();
    await settled;

    expect(jobs[0]).toMatchObject({ status: "error", result: null });
    expect(store.lastSeedUsed).toBeNull();
  });

  it("fails a clean stream close that arrives without a terminal event", async () => {
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();

    openStreams[0]!.resolve();
    await settled;

    // The bare close is no longer the outcome — the host is asked. It has no
    // such job queued and no such print, so this one really is gone, and the
    // directed copy (plus the notification) says so with authority instead of
    // reporting a raw socket event the user cannot act on.
    expect(jobs[0]).toMatchObject({
      status: "error",
      error: "The connection to primary was interrupted and this print didn’t finish.",
    });
    const { notifyGenerationFailed } = await import("../lib/notify");
    expect(vi.mocked(notifyGenerationFailed)).toHaveBeenCalled();
  });

  it("does not classify an HTTP rejection as a resumable transport interruption", async () => {
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch({ ...req }, 1, null, chainDecision);
    await flushPromises();
    openStreams[0]!.onClose?.(new Error("SSE request failed with HTTP 401"));
    openStreams[0]!.resolve();
    await settled;

    expect(jobs[0]).toMatchObject({
      status: "error",
      error: "SSE request failed with HTTP 401",
      interrupted: false,
    });
  });

  it("prune drops the oldest finished jobs and keeps live ones", () => {
    const store = useGenerationStore();
    store.submitBatch({ ...req }, 1, null, chainDecision);
    store.submitBatch({ ...req }, 1, null, chainDecision);
    store.submitBatch({ ...req }, 1, null, chainDecision);
    store.jobs[0]!.status = "complete";
    store.jobs[1]!.status = "error";
    store.prune(1);
    expect(store.jobs).toHaveLength(2);
    expect(store.jobs.some((j) => j.status === "queued")).toBe(true);
  });

  it("prune preserves a just-completed older job selected as the latest result", () => {
    const store = useGenerationStore();
    const jobs = Array.from({ length: 5 }, (_, index) =>
      store.startJob({ ...req, prompt: `prompt ${index}` }),
    );
    for (const job of jobs) job.status = "complete";

    store.prune(4, jobs[0]!.clientId);

    expect(store.jobs).toHaveLength(4);
    expect(store.jobs).toContain(jobs[0]);
    expect(store.jobs).not.toContain(jobs[1]);
  });

  it("keeps long rail history lightweight beyond the freshest media window", () => {
    const store = useGenerationStore();
    for (let index = 0; index < 13; index += 1) {
      const job = store.startJob({ ...req, prompt: `history ${index}` });
      job.status = "complete";
      job.result = {
        image: "large-encoded-output",
        format: "png",
        width: 1024,
        height: 1024,
        seed_used: index,
        generation_time_ms: 1,
        model: req.model,
      } as never;
      job.resultUrl = `blob:history-${index}`;
      job.resultUrlIsObjectUrl = true;
    }

    store.prune(50);

    expect(store.jobs).toHaveLength(13);
    expect(store.jobs[0]?.result?.image).toBe("");
    expect(store.jobs[0]?.resultUrl).toBeNull();
    expect(store.jobs[1]?.result?.image).toBe("large-encoded-output");
  });

  it("prunes only terminal jobs whose consumer callbacks have run", () => {
    const store = useGenerationStore();
    const jobs = Array.from({ length: 3 }, (_, index) =>
      store.startJob({ ...req, prompt: `prompt ${index}` }),
    );
    for (const job of jobs) job.status = "complete";

    store.prune(1, jobs[0]!.clientId, new Set([jobs[0]!.clientId]));
    expect(store.jobs).toEqual(jobs);

    store.prune(1, jobs[1]!.clientId, new Set([jobs[0]!.clientId, jobs[1]!.clientId]));
    expect(store.jobs).toEqual([jobs[1], jobs[2]]);
  });

  it("defers automatic pruning until consumers see a slow older completion", async () => {
    const store = useGenerationStore();
    const { jobs, settled } = store.submitBatch(
      { ...req, prompt: "slow oldest" },
      1,
      null,
      chainDecision,
    );
    await flushPromises();
    for (let index = 0; index < 50; index += 1) {
      const newer = store.startJob({ ...req, prompt: `newer ${index}` });
      newer.status = "complete";
    }
    openStreams[0]!.onEvent("chain_job", chainComplete({ seed: 99 }));
    openStreams[0]!.resolve();

    const returned = await settled;
    expect(returned[0]).toBe(jobs[0]);
    expect(store.jobs).toContain(jobs[0]);
    await new Promise((resolve) => setTimeout(resolve, 1));
    expect(store.jobs).toContain(jobs[0]);
    expect(store.jobs).toHaveLength(50);
  });

  it("does not auto-prune a terminal job from another batch whose consumer is pending", async () => {
    const store = useGenerationStore();
    const pendingBatch = store.submitBatch(
      { ...req, prompt: "pending consumer" },
      1,
      null,
      chainDecision,
    );
    await flushPromises();
    const finishingBatch = store.submitBatch(
      { ...req, prompt: "finishing consumer" },
      1,
      null,
      chainDecision,
    );
    await flushPromises();

    let pendingConsumerObserved = false;
    void pendingBatch.settled.then(() => {
      pendingConsumerObserved = true;
    });

    // The first job has a terminal result, but its held-open transport keeps
    // that batch's settled consumer pending while the other batch completes.
    openStreams[0]!.onEvent("chain_job", chainComplete({ seed: 201 }));
    openStreams[1]!.onEvent("chain_job", chainComplete({ seed: 202 }));
    openStreams[1]!.resolve();

    // Force the finishing batch's automatic housekeeping over the retention
    // threshold. The older terminal job is first in store order, so it would
    // be the first casualty without the cross-batch consumer guard.
    for (let index = 0; index < 50; index += 1) {
      const filler = store.startJob({ ...req, prompt: `finished filler ${index}` });
      filler.status = "complete";
    }

    await finishingBatch.settled;
    await new Promise((resolve) => setTimeout(resolve, 1));

    expect(pendingConsumerObserved).toBe(false);
    expect(store.pendingConsumerBatchIds).toEqual([pendingBatch.jobs[0]!.batchId]);
    expect(store.jobs).toContain(pendingBatch.jobs[0]);
    expect(store.jobs).toHaveLength(51);

    openStreams[0]!.resolve();
    await pendingBatch.settled;
    await flushPromises();
    expect(pendingConsumerObserved).toBe(true);
  });
});
