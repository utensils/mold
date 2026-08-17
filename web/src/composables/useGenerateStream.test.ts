import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __testing__,
  activeCanvasJob,
  isPrebuiltChainRequest,
  latestUnresolvedError,
  resolveChainRequest,
  useGenerateStream,
  type Job,
} from "./useGenerateStream";
import type {
  ChainRequestWire,
  GenerateRequestWire,
  SseCompleteEvent,
} from "../types";
import type { ChainRoutingDecision } from "../lib/chainRouting";
import type {
  ChainStreamHandlers,
  GenerateStreamHandlers,
  StreamTarget,
} from "../api";
import { cancelQueueJob } from "../api";
import type { HostRoute } from "../lib/hostRouting";

function persistedPayload(jobs: unknown[]): string {
  return JSON.stringify({ version: 1, jobs });
}

// Capture the most recent handlers passed into `generateStream` /
// `generateChainStream` so each test can drive the SSE lifecycle (complete /
// error) deterministically without spinning up a real EventSource. The mocks
// resolve immediately — no network.
let lastSingleHandlers: GenerateStreamHandlers | null = null;
let lastChainHandlers: ChainStreamHandlers | null = null;
// The dispatch target the singleton threaded through — `undefined` means the
// submission was never routed and lands on the serving origin.
let lastSingleTarget: StreamTarget | undefined;
let lastChainTarget: StreamTarget | undefined;

vi.mock("../api", () => ({
  cancelQueueJob: vi.fn().mockResolvedValue(undefined),
  generateStream: vi.fn(
    (
      _req: GenerateRequestWire,
      handlers: GenerateStreamHandlers,
      _signal?: AbortSignal,
      target?: StreamTarget,
    ) => {
      lastSingleHandlers = handlers;
      lastSingleTarget = target;
      return Promise.resolve();
    },
  ),
  generateChainStream: vi.fn(
    (
      _req: ChainRequestWire,
      handlers: ChainStreamHandlers,
      _signal?: AbortSignal,
      target?: StreamTarget,
    ) => {
      lastChainHandlers = handlers;
      lastChainTarget = target;
      return Promise.resolve();
    },
  ),
}));

vi.mock("@studio/api/referenceUploads", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/referenceUploads")>()),
  // Keep the real `requestNeedsReferenceUpload` predicate; only the network
  // side of the lease is stubbed so a submission can reach the stream.
  prepareReferenceUploads: vi.fn(
    ({ request }: { request: GenerateRequestWire }) =>
      Promise.resolve({
        request,
        expiresAtMs: Date.now() + 60_000,
        requestScopeSha256: "a".repeat(64),
        cancel: () => Promise.resolve(),
      }),
  ),
}));

function fakeCompleteEvent(
  overrides: Partial<SseCompleteEvent> = {},
): SseCompleteEvent {
  return {
    image: "AAAA",
    format: "png",
    width: 512,
    height: 512,
    seed_used: 42,
    generation_time_ms: 1234,
    model: "flux-dev:fp16",
    ...overrides,
  } as SseCompleteEvent;
}

function chainDecision(
  overrides: Partial<Extract<ChainRoutingDecision, { kind: "chain" }>> = {},
): Extract<ChainRoutingDecision, { kind: "chain" }> {
  return {
    kind: "chain",
    clipFrames: 97,
    motionTail: 4,
    stageCount: 3,
    ...overrides,
  };
}

function singleGen(
  overrides: Partial<GenerateRequestWire> = {},
): GenerateRequestWire {
  return {
    prompt: "a cat walking through autumn leaves",
    model: "ltx-2-19b-distilled:fp8",
    width: 1216,
    height: 704,
    steps: 8,
    guidance: 3.0,
    strength: 1.0,
    fps: 24,
    output_format: "mp4",
    frames: 241,
    ...overrides,
  };
}

function scriptChain(
  overrides: Partial<ChainRequestWire> = {},
): ChainRequestWire {
  return {
    model: "ltx-2-19b-distilled:fp8",
    stages: [
      { prompt: "cat in a garden", frames: 97, transition: "smooth" },
      { prompt: "cat on a rooftop", frames: 97, transition: "cut" },
      { prompt: "cat on the moon", frames: 97, transition: "fade" },
    ],
    motion_tail_frames: 17,
    width: 1216,
    height: 704,
    fps: 24,
    steps: 8,
    guidance: 3.0,
    strength: 1.0,
    output_format: "mp4",
    ...overrides,
  };
}

// ── Load-time dead-letter ───────────────────────────────────────────────────
//
// At module-import time we have no way to reconnect to whatever SSE
// streams were live in the prior session. A persisted `running` row is,
// by construction, a zombie — its underlying connection is gone. The
// load path flips it to `error` with a load-bearing reason so the user
// has a card they can dismiss / retry instead of one that pretends to
// be running indefinitely.
describe("loadPersistedJobs dead-letters running rows on rehydrate", () => {
  function persisted(overrides: Record<string, unknown> = {}) {
    return {
      id: "abc",
      request: singleGen(),
      startedAt: 1_000_000,
      progress: {
        stage: "Denoising",
        step: 5,
        totalSteps: 30,
        weightBytesLoaded: null,
        weightBytesTotal: null,
        queuePosition: null,
        gpu: null,
        elapsedMs: 1000,
      },
      result: null,
      error: null,
      state: "running",
      chain: null,
      lastProgressAt: 1_010_000,
      workStarted: true,
      hostId: null,
      hostLabel: null,
      serverId: null,
      ...overrides,
    };
  }

  it("flips a persisted running job to error with a server-progress-lost message", () => {
    const raw = persistedPayload([persisted({ id: "zombie-1" })]);
    const jobs = __testing__.loadPersistedJobs(raw);
    expect(jobs).toHaveLength(1);
    expect(jobs[0].state).toBe("error");
    expect(jobs[0].error).toMatch(/server progress lost/i);
    // Progress is preserved so the user sees where the zombie stalled
    // instead of an empty card.
    expect(jobs[0].progress.stage).toBe("Denoising");
    expect(jobs[0].progress.step).toBe(5);
  });

  it("gives a failure discovered from a running row current canvas authority", () => {
    const raw = persistedPayload([
      persisted({ id: "older-zombie", startedAt: 100 }),
      persisted({ id: "newer-zombie", startedAt: 200 }),
      persisted({ id: "old-error", state: "error", error: "old failure" }),
    ]);

    const loaded = __testing__.loadPersistedState(raw);

    expect(loaded.canvasErrorJobId).toBe("newer-zombie");
    expect(loaded.jobs.map((job) => job.state)).toEqual([
      "error",
      "error",
      "error",
    ]);
    expect(loaded.jobs[0].settledAt).toBeGreaterThan(1_010_000);
    expect(loaded.jobs[1].settledAt).toBe(loaded.jobs[0].settledAt);
  });

  it("writes a boot-discovered failure through so the next boot sees history", () => {
    const raw = persistedPayload([persisted({ id: "one-time-zombie" })]);
    localStorage.setItem(__testing__.STORAGE_KEY, raw);
    const firstBoot = __testing__.initializePersistedState(raw);
    expect(firstBoot.canvasErrorJobId).toBe("one-time-zombie");

    const stored = localStorage.getItem(__testing__.STORAGE_KEY);
    const secondBoot = __testing__.loadPersistedState(stored);

    expect(secondBoot.canvasErrorJobId).toBeNull();
    expect(secondBoot.jobs[0]).toMatchObject({
      id: "one-time-zombie",
      state: "error",
      error: "page reloaded — server progress lost",
    });
    localStorage.removeItem(__testing__.STORAGE_KEY);
  });

  it("keeps a reloaded job with a known server id running for the reconciler", () => {
    // A reload must not announce "Generation failed" while the server may
    // still be rendering: the queue reconciler proves the outcome instead.
    const raw = persistedPayload([
      persisted({ id: "reloaded", serverId: "srv-1" }),
    ]);
    const loaded = __testing__.loadPersistedState(raw);
    expect(loaded.canvasErrorJobId).toBeNull();
    expect(loaded.jobs[0]).toMatchObject({
      id: "reloaded",
      state: "running",
      error: null,
      detached: true,
      settledAt: null,
    });
    // Progress carried through so the card shows where it left off.
    expect(loaded.jobs[0].progress.stage).toBe("Denoising");
  });

  it("does not give settled persisted failures canvas authority", () => {
    const raw = persistedPayload([
      persisted({ id: "old-error", state: "error", error: "old failure" }),
    ]);
    expect(__testing__.loadPersistedState(raw).canvasErrorJobId).toBeNull();
  });

  it("passes through done jobs unchanged", () => {
    const raw = persistedPayload([persisted({ id: "x", state: "done" })]);
    const jobs = __testing__.loadPersistedJobs(raw);
    expect(jobs).toHaveLength(1);
    expect(jobs[0].state).toBe("done");
    expect(jobs[0].error).toBeNull();
  });

  it("passes through error jobs unchanged (preserves the original error text)", () => {
    const raw = persistedPayload([
      persisted({ id: "x", state: "error", error: "OOM" }),
    ]);
    const jobs = __testing__.loadPersistedJobs(raw);
    expect(jobs[0].state).toBe("error");
    expect(jobs[0].error).toBe("OOM");
  });

  it("returns [] for null / empty / malformed payloads (no throw)", () => {
    expect(__testing__.loadPersistedJobs(null)).toEqual([]);
    expect(__testing__.loadPersistedJobs("")).toEqual([]);
    expect(__testing__.loadPersistedJobs("not-json")).toEqual([]);
    expect(__testing__.loadPersistedJobs('{"not":"an array"}')).toEqual([]);
    expect(
      __testing__.loadPersistedJobs(JSON.stringify([persisted()])),
    ).toEqual([]);
  });

  it("rejects a payload from a different storage version", () => {
    const raw = JSON.stringify({ version: 0, jobs: [persisted()] });
    expect(__testing__.loadPersistedJobs(raw)).toEqual([]);
  });
});

describe("isPrebuiltChainRequest", () => {
  it("returns true for a ChainRequestWire with populated stages", () => {
    expect(isPrebuiltChainRequest(scriptChain())).toBe(true);
  });

  it("returns false for a single-clip GenerateRequestWire", () => {
    expect(isPrebuiltChainRequest(singleGen())).toBe(false);
  });

  it("returns false when `stages` is an empty array", () => {
    // An empty stages[] is ambiguous — treat it as 'not pre-built' so the
    // auto-expand helper takes over. (Server would 422 either way, but this
    // keeps the router predictable.)
    const req = scriptChain({ stages: [] });
    expect(isPrebuiltChainRequest(req)).toBe(false);
  });
});

describe("resolveChainRequest", () => {
  it("passes a script payload through verbatim (regression: prior code nuked stages)", () => {
    // Repro for the HTTP 422
    //   "chain request needs either stages[] or prompt + total_frames"
    // that fired whenever Script mode submitted a ChainRequestWire: submit()
    // used to unconditionally re-project through buildChainRequest, which
    // reads GenerateRequestWire fields that don't exist on a script payload
    // and dropped `stages` entirely. The outgoing body ended up with no
    // stages and no auto-expand form → server 422.
    const req = scriptChain();
    const resolved = resolveChainRequest(req, chainDecision());
    expect(resolved).toBe(req);
    expect(resolved.stages).toHaveLength(3);
    expect(resolved.stages?.[0]?.prompt).toBe("cat in a garden");
    expect(resolved.stages?.[1]?.transition).toBe("cut");
    expect(resolved.stages?.[2]?.transition).toBe("fade");
    // Script mode must not smuggle in auto-expand fields either; those are
    // mutually exclusive with stages[] and the server's normalise() would
    // prefer stages[] regardless, but having them unset keeps the wire body
    // unambiguous.
    expect(resolved.prompt).toBeUndefined();
    expect(resolved.total_frames).toBeUndefined();
  });

  it("projects a single-prompt request into the auto-expand form", () => {
    const req = singleGen({
      prompt: "a single prompt",
      frames: 241,
      original_prompt: "source prompt",
      batch_id: "prepared-batch-1",
      batch_index: 2,
      batch_count: 3,
    });
    const resolved = resolveChainRequest(req, chainDecision());
    expect(resolved.stages).toBeUndefined();
    expect(resolved.prompt).toBe("a single prompt");
    expect(resolved.total_frames).toBe(241);
    expect(resolved.clip_frames).toBe(97);
    expect(resolved.motion_tail_frames).toBe(4);
    expect(resolved).toMatchObject({
      original_prompt: "source prompt",
      batch_id: "prepared-batch-1",
      batch_index: 2,
      batch_count: 3,
    });
  });

  it("falls back to auto-expand when stages[] is empty", () => {
    // Defensive: a caller shouldn't send empty stages, but if they do we
    // don't want to pass that through (the server would 422 on empty
    // stages). The resolver treats this as 'not pre-built' and re-projects.
    const req = scriptChain({ stages: [] });
    const resolved = resolveChainRequest(
      req as unknown as GenerateRequestWire,
      chainDecision(),
    );
    // buildChainRequest reads `prompt`/`frames` off the input — those are
    // absent on an empty-stages chain request, so they come through as
    // undefined. That's the expected downstream failure mode (422 from the
    // server), not a silent success. The assertion here only verifies that
    // we took the non-passthrough branch.
    expect(resolved.stages).toBeUndefined();
  });
});

// ── Work-start tracking ─────────────────────────────────────────────────────
//
// The stale-stream warning is useful only after server-side work has actually
// begun. A job can sit in the queue without progress for minutes; that should
// not look like a dropped stream. These tests drive the SSE callbacks directly
// so the component can rely on `job.workStarted` instead of inferring from
// display text.
describe("workStarted tracking", () => {
  beforeEach(() => {
    lastSingleHandlers = null;
    lastChainHandlers = null;
    try {
      localStorage.removeItem("mold.generate.jobs");
    } catch {
      /* ignore — happy-dom should have it */
    }
    const stream = useGenerateStream();
    stream.canvasErrorJobId.value = null;
    for (const j of stream.jobs.value) {
      if (j.state === "running") {
        j.controller.abort();
        j.state = "canceled";
      }
    }
    stream.clearDone();
  });

  it("stays false while the job is queued, then flips on real progress", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    const job = stream.jobs.value.find((j) => j.id === id)!;

    expect(job.workStarted).toBe(false);
    expect(lastSingleHandlers).not.toBeNull();
    lastSingleHandlers!.onProgress({
      type: "queued",
      position: 3,
      id: "server-job",
    });
    expect(job.workStarted).toBe(false);
    expect(job.progress.queuePosition).toBe(3);
    expect(job.serverId).toBe("server-job");

    lastSingleHandlers!.onProgress({ type: "stage_start", name: "Loading" });
    expect(job.workStarted).toBe(true);
    expect(job.progress.queuePosition).toBeNull();
  });

  it("keeps a chain queued until a stage starts and between durable stages", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 241 }), chainDecision());
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    expect(lastChainHandlers).not.toBeNull();

    lastChainHandlers!.onProgress({
      type: "chain_start",
      stage_count: 3,
      estimated_total_frames: 241,
    });
    expect(job.workStarted).toBe(false);
    expect(job.progress.stage).toBe("Queued · 3 clips · ~241 frames");

    lastChainHandlers!.onProgress({ type: "stage_start", stage_idx: 0 });
    expect(job.workStarted).toBe(true);
    expect(job.progress.stage).toBe("Preparing clip 1/3");

    lastChainHandlers!.onProgress({
      type: "stage_done",
      stage_idx: 0,
      frames_emitted: 97,
    });
    expect(job.workStarted).toBe(false);
    expect(job.progress.stage).toBe("Clip 1/3 done · next clip queued");

    lastChainHandlers!.onProgress({
      type: "stage_done",
      stage_idx: 2,
      frames_emitted: 97,
    });
    expect(job.workStarted).toBe(true);
    expect(job.progress.stage).toBe("Clip 3/3 done · preparing final output");

    lastChainHandlers!.onProgress({ type: "stitching", total_frames: 241 });
    expect(job.workStarted).toBe(true);
    expect(job.progress.stage).toBe("Stitching 241 frames…");
  });

  it("returns to automatic canvas selection when new work is submitted", () => {
    const stream = useGenerateStream();
    const inspected = stream.submit(singleGen({ prompt: "inspected" }), {
      kind: "single",
    });
    stream.select(inspected);
    stream.canvasErrorJobId.value = "historical-failure";
    expect(stream.selectedJob.value?.id).toBe(inspected);

    stream.submit(singleGen({ prompt: "new work" }), { kind: "single" });
    expect(stream.selectedJob.value).toBeNull();
    expect(stream.canvasErrorJobId.value).toBeNull();
  });

  it("settles a reconciled missing job through the canvas error authority", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ prompt: "lost stream" }), {
      kind: "single",
    });
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    job.previewUrl = "data:image/png;base64,preview";

    stream.failRunning(id, "job not found on server — connection lost");

    expect(job.state).toBe("error");
    expect(job.error).toBe("job not found on server — connection lost");
    expect(job.settledAt).not.toBeNull();
    expect(job.previewUrl).toBeNull();
    expect(stream.canvasErrorJobId.value).toBe(id);
  });

  it("does not let a late reconciliation response overwrite terminal state", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ prompt: "already complete" }), {
      kind: "single",
    });
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    job.state = "done";
    job.settledAt = Date.now();

    stream.failRunning(id, "job not found on server — connection lost");

    expect(job.state).toBe("done");
    expect(job.error).toBeNull();
    expect(stream.canvasErrorJobId.value).toBeNull();
  });

  it("settles a retained terminal frame softly instead of as a hard failure", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ prompt: "kept in the queue" }), {
      kind: "single",
    });
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    job.previewUrl = "data:image/png;base64,preview";

    lastSingleHandlers!.onError({
      kind: "http",
      status: 0,
      body: JSON.stringify({
        message: "mold is restarting; this generation was kept in the queue",
        retained: true,
        code: "server_restarting",
      }),
    });

    expect(job.state).toBe("error");
    expect(job.error).toBe(
      "mold is restarting; this generation was kept in the queue",
    );
    expect(job.settledAt).not.toBeNull();
    expect(job.previewUrl).toBeNull();
    // The host kept the work: it may well finish. The canvas must not claim
    // this print failed.
    expect(stream.canvasErrorJobId.value).toBeNull();
  });

  it("keeps a non-retained terminal frame the canvas failure authority", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ prompt: "really failed" }), {
      kind: "single",
    });
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;

    lastSingleHandlers!.onError({
      kind: "http",
      status: 0,
      body: JSON.stringify({ message: "host ran out of memory" }),
    });

    expect(job.error).toBe("host ran out of memory");
    expect(stream.canvasErrorJobId.value).toBe(id);
  });

  it("does not treat pre-queue info as work, but does treat post-queue info as work", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    const job = stream.jobs.value.find((j) => j.id === id)!;
    expect(lastSingleHandlers).not.toBeNull();

    lastSingleHandlers!.onProgress({
      type: "info",
      message: "dimension warning",
    });
    expect(job.workStarted).toBe(false);

    lastSingleHandlers!.onProgress({
      type: "queued",
      position: 1,
      id: "server-job",
    });
    lastSingleHandlers!.onProgress({
      type: "info",
      message: "Downloading weights",
    });
    expect(job.workStarted).toBe(true);
    expect(job.progress.queuePosition).toBeNull();
  });
});

describe("insecure-context compatibility", () => {
  it("submits when crypto.randomUUID is unavailable", () => {
    vi.stubGlobal("crypto", {
      getRandomValues(bytes: Uint8Array) {
        bytes.fill(7);
        return bytes;
      },
    });
    try {
      const id = useGenerateStream().submit(singleGen(), { kind: "single" });
      expect(id).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
      );
      expect(lastSingleHandlers).not.toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

// ── Auto-remove on completion ───────────────────────────────────────────────
//
// The running-strip card is supposed to vanish ~1.5 s after a successful
// generation lands so the freshly-arrived gallery thumbnail underneath
// becomes the focal point instead of duplicating it. These tests pin the
// timing contract: success auto-dismisses, failure modes don't, and a
// manual dismiss before the timer fires is harmless.
//
// We also clean up `localStorage` between cases — the singleton's `jobs`
// ref is module-scoped and persists across tests in the same file, so
// without isolation a leftover "running" job from one case would skew
// the next case's job-count assertions.
describe("auto-remove completed jobs", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    lastSingleHandlers = null;
    lastChainHandlers = null;
    // Reset module-level singleton state between tests. We can't import
    // the underlying ref directly, so the cleanest path is wiping the
    // persisted snapshot and clearing whatever jobs the previous test
    // left in the live ref via clearDone()/cancel()+clearDone().
    try {
      localStorage.removeItem("mold.generate.jobs");
    } catch {
      /* ignore — happy-dom should have it */
    }
    const stream = useGenerateStream();
    stream.canvasErrorJobId.value = null;
    // Cancel anything still "running" then drop everything settled.
    for (const j of stream.jobs.value) {
      if (j.state === "running") {
        j.controller.abort();
        j.state = "canceled";
      }
    }
    stream.clearDone();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("auto-removes a job ~1500ms after it transitions to done", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("running");
    stream.canvasErrorJobId.value = "prior-failure";

    // Fire the SSE complete callback the singleton registered.
    expect(lastSingleHandlers).not.toBeNull();
    lastSingleHandlers!.onComplete(fakeCompleteEvent({ seed_used: 7 }));

    // Job is "done" but still on screen during the grace period.
    const job = stream.jobs.value.find((j) => j.id === id);
    expect(job?.state).toBe("done");
    expect(stream.canvasErrorJobId.value).toBeNull();

    // Just before the timer — still present.
    vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS - 1);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeDefined();

    // Tick past the timer — gone.
    vi.advanceTimersByTime(2);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeUndefined();
  });

  it("does not restore an older failure after a later success auto-removes", () => {
    const stream = useGenerateStream();
    const failedId = stream.submit(singleGen({ prompt: "first attempt" }), {
      kind: "single",
    });
    lastSingleHandlers!.onError({
      kind: "http",
      status: 500,
      body: "owner thread lost",
    });
    const failed = stream.jobs.value.find((job) => job.id === failedId)!;
    expect(stream.canvasErrorJobId.value).toBe(failedId);

    const successfulId = stream.submit(
      singleGen({ prompt: "second attempt" }),
      { kind: "single" },
    );
    lastSingleHandlers!.onComplete(fakeCompleteEvent());
    expect(stream.canvasErrorJobId.value).toBeNull();

    vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS + 1);

    expect(
      stream.jobs.value.find((job) => job.id === successfulId),
    ).toBeUndefined();
    expect(stream.jobs.value.find((job) => job.id === failedId)).toBe(failed);
    expect(
      latestUnresolvedError(stream.jobs.value, stream.canvasErrorJobId.value),
    ).toBeUndefined();
  });

  it("does NOT auto-remove a job that errors out", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    expect(lastSingleHandlers).not.toBeNull();
    lastSingleHandlers!.onError({
      kind: "http",
      status: 500,
      body: "boom",
    });
    const job = stream.jobs.value.find((j) => j.id === id);
    expect(job?.state).toBe("error");
    expect(stream.canvasErrorJobId.value).toBe(id);

    // Even well past the would-be auto-remove window, an errored card
    // sticks around for the user to read.
    vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS * 5);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeDefined();
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("error");
  });

  it("shows the server's JSON error message without transport noise", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 25 }), { kind: "single" });
    lastSingleHandlers!.onError({
      kind: "http",
      status: 0,
      body: JSON.stringify({
        message:
          "generation error: LTX-2 audio output is unavailable. Set enable_audio=false and retry.",
      }),
    });

    expect(stream.jobs.value.find((j) => j.id === id)?.error).toBe(
      "generation error: LTX-2 audio output is unavailable. Set enable_audio=false and retry.",
    );
  });

  it("does NOT auto-remove a locally waiting canceled job", async () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    stream.select(id);
    expect(stream.selectedJob.value?.id).toBe(id);
    const submitted = stream.jobs.value.find((j) => j.id === id)!;
    submitted.streamStarted = false;
    await stream.cancel(id);
    const job = stream.jobs.value.find((j) => j.id === id);
    expect(job?.state).toBe("canceled");
    expect(stream.selectedJob.value).toBeNull();

    vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS * 5);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeDefined();
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("canceled");
  });

  it("manual remove() before the auto-remove timer is harmless", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    expect(lastSingleHandlers).not.toBeNull();
    lastSingleHandlers!.onComplete(fakeCompleteEvent());

    // User dismisses early.
    stream.remove(id);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeUndefined();

    // The pending setTimeout still fires; removeJob filters by id so a
    // missing id is a no-op. No throw, no resurrection, no duplicate
    // removal effects on other jobs.
    expect(() =>
      vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS + 100),
    ).not.toThrow();
    expect(stream.jobs.value.find((j) => j.id === id)).toBeUndefined();
  });

  it("keeps a completed result authoritative if cancel is clicked during its grace period", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    expect(lastSingleHandlers).not.toBeNull();
    lastSingleHandlers!.onComplete(fakeCompleteEvent());
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("done");

    // A late Cancel cannot rewrite the server's completed terminal result.
    vi.advanceTimersByTime(500);
    stream.cancel(id);
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("done");

    // It follows the normal completed-row removal path.
    vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS + 100);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeUndefined();
  });

  it("auto-removes a chain job ~1500ms after chain complete", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 241 }), chainDecision());
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("running");
    expect(lastChainHandlers).not.toBeNull();

    // Chain complete events carry a `video` field instead of `image`.
    lastChainHandlers!.onComplete({
      video: "AAAA",
      format: "mp4",
      width: 1216,
      height: 704,
      frames: 241,
      fps: 24,
      generation_time_ms: 9876,
      // The fields below are optional on the wire but the singleton
      // shape-shifts them into a SseCompleteEvent with sensible defaults.
      thumbnail: null,
      gif_preview: null,
      has_audio: false,
      duration_ms: null,
      audio_sample_rate: null,
      audio_channels: null,
      gpu: 0,
    } as Parameters<ChainStreamHandlers["onComplete"]>[0]);

    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("done");
    vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS + 1);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeUndefined();
  });
});

// ── Live latent preview ─────────────────────────────────────────────────────
//
// The server emits `preview` progress events (small base64 PNG at latent
// resolution) during denoising. The reducer folds them into the job as a
// data URI — deliberately NOT a blob URL, so the persisted module-singleton
// job list needs no revoke lifecycle — and drops them once the job settles.
describe("live latent preview", () => {
  beforeEach(() => {
    lastSingleHandlers = null;
    try {
      localStorage.removeItem("mold.generate.jobs");
    } catch {
      /* ignore — happy-dom should have it */
    }
    const stream = useGenerateStream();
    for (const j of stream.jobs.value) {
      if (j.state === "running") {
        j.controller.abort();
        j.state = "canceled";
      }
    }
    stream.clearDone();
  });

  it("reduces a preview event into a data-URI preview with step/total", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    const job = stream.jobs.value.find((j) => j.id === id)!;
    expect(job.previewUrl).toBeNull();

    expect(lastSingleHandlers).not.toBeNull();
    lastSingleHandlers!.onProgress({
      type: "preview",
      image: "UFJFVklFVw==",
      step: 3,
      total: 8,
    });

    expect(job.previewUrl).toBe("data:image/png;base64,UFJFVklFVw==");
    expect(job.progress.step).toBe(3);
    expect(job.progress.totalSteps).toBe(8);
    // A preview only exists once denoising is underway — it proves work.
    expect(job.workStarted).toBe(true);
  });

  it("clears the preview when the job completes", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    const job = stream.jobs.value.find((j) => j.id === id)!;
    lastSingleHandlers!.onProgress({
      type: "preview",
      image: "AAAA",
      step: 1,
      total: 4,
    });
    expect(job.previewUrl).not.toBeNull();

    lastSingleHandlers!.onComplete(fakeCompleteEvent());
    expect(job.state).toBe("done");
    expect(job.previewUrl).toBeNull();
  });

  it("clears the preview when the job errors", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    const job = stream.jobs.value.find((j) => j.id === id)!;
    lastSingleHandlers!.onProgress({
      type: "preview",
      image: "AAAA",
      step: 1,
      total: 4,
    });
    expect(job.previewUrl).not.toBeNull();

    lastSingleHandlers!.onError({ kind: "http", status: 500, body: "boom" });
    expect(job.state).toBe("error");
    expect(job.previewUrl).toBeNull();
  });

  it("derives seedVisual from an explicit seed", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1, seed: 42 }), {
      kind: "single",
    });
    const job = stream.jobs.value.find((j) => j.id === id)!;
    expect(job.seedVisual).toBe("42");
  });

  it("derives a stable model·prompt seedVisual when the seed is random", () => {
    const stream = useGenerateStream();
    const req = singleGen({ frames: 1 });
    const id = stream.submit(req, { kind: "single" });
    const job = stream.jobs.value.find((j) => j.id === id)!;
    expect(job.seedVisual).toBe(`${req.model}·${req.prompt}`);
  });

  it("omits previewUrl from persistence and rehydrates it as null", async () => {
    vi.useFakeTimers();
    try {
      const stream = useGenerateStream();
      const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
      lastSingleHandlers!.onProgress({
        type: "preview",
        image: "UFJFVklFVw==",
        step: 3,
        total: 8,
      });
      expect(stream.jobs.value.find((j) => j.id === id)?.previewUrl).not.toBe(
        null,
      );

      // Let the deep watcher flush, then the 200 ms persist debounce fire.
      await vi.advanceTimersByTimeAsync(300);
      const raw = localStorage.getItem(__testing__.STORAGE_KEY);
      expect(raw).not.toBeNull();
      expect(raw!).not.toContain("previewUrl");
      expect(raw!).not.toContain("UFJFVklFVw==");

      const restored = __testing__.loadPersistedJobs(raw);
      const back = restored.find((j) => j.id === id)!;
      expect(back.previewUrl).toBeNull();
      // seedVisual is recomputed from the persisted request.
      expect(back.seedVisual).toBe(
        `${back.request.model}·${(back.request as GenerateRequestWire).prompt}`,
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it("persists only reference descriptors, never inline bytes or one-use secrets", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    job.request = singleGen({
      model: "minimax-h3-ref2va",
      frames: 124,
      references: [
        {
          kind: "image",
          media: { authority: "inline", data: "PRIVATE-IMAGE-BYTES" },
          provenance: { name: "identity.png", sha256: "a".repeat(64) },
          mime_type: "image/png",
          width: 32,
          height: 24,
        },
        {
          kind: "audio",
          media: { authority: "upload", handle: "ONE-USE-HANDLE" },
          provenance: { name: "timing.wav", sha256: "b".repeat(64) },
          mime_type: "audio/wav",
          duration_ms: 2_000,
          sample_rate: 24_000,
          channels: 1,
        },
      ],
    });

    __testing__.persistJobs([job]);
    const raw = localStorage.getItem(__testing__.STORAGE_KEY)!;
    expect(raw).not.toContain("PRIVATE-IMAGE-BYTES");
    expect(raw).not.toContain("ONE-USE-HANDLE");
    const restored = __testing__.loadPersistedJobs(raw)[0]!;
    expect((restored.request as GenerateRequestWire).references).toEqual([
      expect.objectContaining({ media: { authority: "descriptor" } }),
      expect.objectContaining({ media: { authority: "descriptor" } }),
    ]);
  });
});

// ── Active-canvas job selection ─────────────────────────────────────────────
//
// With several jobs queued (prepared batch variations), the rail is
// newest-first but the server denoises the EARLIEST submission — a naive
// "first running" pick binds the canvas to a job that will sit previewless
// while another one is actively developing.
describe("activeCanvasJob", () => {
  function runningJob(overrides: Partial<Job> = {}): Job {
    return {
      id: crypto.randomUUID(),
      request: singleGen({ frames: 1 }),
      startedAt: 0,
      controller: new AbortController(),
      progress: {
        stage: "Starting",
        step: null,
        totalSteps: null,
        weightBytesLoaded: null,
        weightBytesTotal: null,
        queuePosition: null,
        gpu: null,
        elapsedMs: null,
      },
      result: null,
      error: null,
      state: "running",
      settledAt: null,
      chain: null,
      lastProgressAt: 0,
      workStarted: false,
      hostId: null,
      hostLabel: null,
      target: null,
      serverId: null,
      streamStarted: false,
      previewUrl: null,
      seedVisual: "seed",
      ...overrides,
    };
  }

  it("returns undefined when nothing is running", () => {
    expect(activeCanvasJob([])).toBeUndefined();
    expect(activeCanvasJob([runningJob({ state: "done" })])).toBeUndefined();
  });

  it("prefers the running job that holds a live preview over a newer one", () => {
    const newerIdle = runningJob({ id: "newer", startedAt: 200 });
    const olderDeveloping = runningJob({
      id: "older",
      startedAt: 100,
      previewUrl: "data:image/png;base64,AAAA",
    });
    // Rail order is newest-first.
    expect(activeCanvasJob([newerIdle, olderDeveloping])?.id).toBe("older");
  });

  it("falls back to the earliest-submitted running job when none has a preview", () => {
    const newer = runningJob({ id: "newer", startedAt: 200 });
    const older = runningJob({ id: "older", startedAt: 100 });
    const settled = runningJob({ id: "done", startedAt: 50, state: "done" });
    expect(activeCanvasJob([newer, older, settled])?.id).toBe("older");
  });
});

describe("latestUnresolvedError", () => {
  function failedJob(
    id: string,
    settledAt: number,
    overrides: Partial<Job> = {},
  ): Job {
    return {
      id,
      request: singleGen({ frames: 1 }),
      startedAt: settledAt - 100,
      controller: new AbortController(),
      progress: {
        stage: "Failed",
        step: null,
        totalSteps: null,
        weightBytesLoaded: null,
        weightBytesTotal: null,
        queuePosition: null,
        gpu: null,
        elapsedMs: null,
      },
      result: null,
      error: "model load error",
      state: "error",
      settledAt,
      chain: null,
      lastProgressAt: settledAt,
      workStarted: true,
      hostId: null,
      hostLabel: null,
      target: null,
      serverId: null,
      streamStarted: false,
      previewUrl: null,
      seedVisual: "seed",
      ...overrides,
    };
  }

  it("does not resurrect a historical failure without live canvas authority", () => {
    const stale = failedJob("stale", 100);
    expect(latestUnresolvedError([stale], null)).toBeUndefined();
  });

  it("shows the exact live failure that owns the canvas", () => {
    const current = failedJob("current", 300);
    expect(latestUnresolvedError([current], "current")?.id).toBe("current");
  });

  it("fails closed when the authority id is missing or no longer failed", () => {
    const current = failedJob("current", 300);
    expect(latestUnresolvedError([current], "missing")).toBeUndefined();
    expect(
      latestUnresolvedError(
        [failedJob("done", 300, { state: "done", error: null })],
        "done",
      ),
    ).toBeUndefined();
  });

  it("shows an explicitly opened historical failure", () => {
    const stale = failedJob("stale", 100);
    expect(latestUnresolvedError([stale], null, stale)?.id).toBe("stale");
  });
});

describe("useGenerateStream host routing", () => {
  beforeEach(() => {
    localStorage.clear();
    lastSingleTarget = undefined;
    lastChainTarget = undefined;
  });

  const studioRoute: HostRoute = {
    hostId: "studio",
    label: "Studio",
    target: { baseUrl: "http://studio:7680", apiKey: "sk-studio" },
  };

  it("dispatches a single-clip submission to the routed host with its key", () => {
    const stream = useGenerateStream();
    stream.submit(singleGen({ frames: 1 }), { kind: "single" }, studioRoute);
    expect(lastSingleTarget).toEqual({
      baseUrl: "http://studio:7680",
      apiKey: "sk-studio",
    });
  });

  it("dispatches an auto-promoted chain submission to the routed host", () => {
    const stream = useGenerateStream();
    stream.submit(singleGen({ frames: 241 }), chainDecision(), studioRoute);
    expect(lastChainTarget).toEqual({
      baseUrl: "http://studio:7680",
      apiKey: "sk-studio",
    });
  });

  it("leaves the target undefined when no route is supplied (origin)", () => {
    const stream = useGenerateStream();
    stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    expect(lastSingleTarget).toBeUndefined();
  });

  it("softly detaches a network close from a host that retains its queue", () => {
    const stream = useGenerateStream();
    const id = stream.submit(
      singleGen({ frames: 1 }),
      { kind: "single" },
      {
        ...studioRoute,
        durableQueue: true,
      },
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;

    lastSingleHandlers!.onError({
      kind: "network",
      message: "connection lost",
    });

    expect(job.state).toBe("error");
    expect(stream.canvasErrorJobId.value).toBeNull();
  });

  it("keeps a reference-upload network close a hard failure even on a durable host", async () => {
    // Reference-upload media is excluded from the journal at admission, so
    // this job reports `durable: false` on a host that advertises the durable
    // queue. Promising it will finish would be a lie.
    const stream = useGenerateStream();
    const id = stream.submit(
      singleGen({
        model: "minimax-h3-ref2va",
        frames: 124,
        references: [
          {
            kind: "image",
            media: { authority: "inline", data: "PRIVATE-IMAGE-BYTES" },
            provenance: { name: "identity.png", sha256: "a".repeat(64) },
            mime_type: "image/png",
            width: 32,
            height: 24,
          },
        ],
      } as Partial<GenerateRequestWire>),
      { kind: "single" },
      { ...studioRoute, durableQueue: true },
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    // The lease is awaited before the stream opens.
    await vi.waitFor(() => expect(job.streamStarted).toBe(true));

    lastSingleHandlers!.onError({
      kind: "network",
      message: "connection lost",
    });

    expect(stream.canvasErrorJobId.value).toBe(id);
    expect(job.error).toBe("connection lost");
  });

  it("keeps a network close a hard failure against a host without a durable queue", () => {
    const stream = useGenerateStream();
    const id = stream.submit(
      singleGen({ frames: 1 }),
      { kind: "single" },
      studioRoute,
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;

    lastSingleHandlers!.onError({
      kind: "network",
      message: "connection lost",
    });

    expect(job.error).toBe("connection lost");
    expect(stream.canvasErrorJobId.value).toBe(id);
  });

  it("attributes the job to the host it was routed to", () => {
    const stream = useGenerateStream();
    const id = stream.submit(
      singleGen({ frames: 1 }),
      { kind: "single" },
      studioRoute,
    );
    const job = stream.jobs.value.find((j) => j.id === id);
    expect(job?.hostId).toBe("studio");
    expect(job?.hostLabel).toBe("Studio");
  });

  it("cancels the server job on the exact routed host", async () => {
    const stream = useGenerateStream();
    const id = stream.submit(
      singleGen({ frames: 1 }),
      { kind: "single" },
      studioRoute,
    );
    lastSingleHandlers?.onProgress({
      type: "queued",
      id: "server-job-7",
      position: 0,
    });

    await stream.cancel(id);

    expect(cancelQueueJob).toHaveBeenCalledWith(
      "server-job-7",
      studioRoute.target,
    );
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("canceled");
  });

  it("keeps a server-owned job running when cancellation is refused", async () => {
    vi.mocked(cancelQueueJob).mockRejectedValueOnce(
      new Error("already running"),
    );
    const stream = useGenerateStream();
    const id = stream.submit(
      singleGen({ frames: 1 }),
      { kind: "single" },
      studioRoute,
    );
    lastSingleHandlers?.onProgress({
      type: "queued",
      id: "server-job-running",
      position: 0,
    });

    await expect(stream.cancel(id)).rejects.toThrow("already running");

    const job = stream.jobs.value.find((candidate) => candidate.id === id);
    expect(job?.state).toBe("running");
    expect(job?.controller.signal.aborted).toBe(false);
  });

  it("keeps an opened stream live until its queue id can be cancelled", async () => {
    const stream = useGenerateStream();
    const id = stream.submit(
      singleGen({ frames: 1 }),
      { kind: "single" },
      studioRoute,
    );

    await expect(stream.cancel(id)).rejects.toThrow(
      "Remote cancellation was not confirmed before the queue ID arrived.",
    );

    const job = stream.jobs.value.find((candidate) => candidate.id === id);
    expect(job?.state).toBe("running");
    expect(job?.controller.signal.aborted).toBe(false);
  });

  it("leaves an unrouted job unattributed rather than guessing", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    const job = stream.jobs.value.find((j) => j.id === id);
    expect(job?.hostId).toBeNull();
    expect(job?.hostLabel).toBeNull();
  });

  it("round-trips the host attribution through persistence", () => {
    const restored = __testing__.loadPersistedJobs(
      persistedPayload([
        {
          id: "j1",
          request: singleGen({ frames: 1 }),
          startedAt: 1,
          progress: null,
          result: null,
          error: null,
          state: "done",
          chain: null,
          lastProgressAt: 1,
          workStarted: true,
          hostId: "studio",
          hostLabel: "Studio",
          serverId: null,
        },
      ]),
    );
    expect(restored[0]?.hostId).toBe("studio");
    expect(restored[0]?.hostLabel).toBe("Studio");
  });

  it("rejects the retired pre-routing job-array schema", () => {
    const restored = __testing__.loadPersistedJobs(JSON.stringify([]));
    expect(restored).toEqual([]);
  });
});
