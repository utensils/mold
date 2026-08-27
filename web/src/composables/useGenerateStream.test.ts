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
  GalleryImage,
  GenerateRequestWire,
} from "../types";
import type {
  GenerationBatchStatus,
  GenerationBatchStatusResponse,
} from "@studio/api/generationAdmission";
import type { ChainRoutingDecision } from "../lib/chainRouting";
import type { ChainStreamHandlers, StreamTarget } from "../api";
import { cancelQueueJob } from "../api";
import type { HostRoute } from "../lib/hostRouting";

function persistedPayload(jobs: unknown[]): string {
  return JSON.stringify({ version: 1, jobs });
}

// A SEQUENCE is the only submission that still opens an attached SSE stream,
// so `generateChainStream`'s handlers are captured to drive that lifecycle.
// Every PRINT is admitted through `POST /api/generation-batches` and settles
// through the durable authority — `admitGenerationBatch` plus
// `reconcileGenerationBatches` are its equivalent seam.
let lastChainHandlers: ChainStreamHandlers | null = null;
// The dispatch target the singleton threaded through — `undefined` means the
// submission was never routed and lands on the serving origin.
let lastChainTarget: StreamTarget | undefined;

const admitGenerationBatch = vi.hoisted(() => vi.fn());
const lookupGenerationBatchByClientId = vi.hoisted(() => vi.fn());
const reconcileGenerationBatches = vi.hoisted(() => vi.fn());
const fetchEventSource = vi.hoisted(() => vi.fn(() => new Promise(() => {})));
const mutateQueueJobOnExpectedInstance = vi.hoisted(() =>
  vi.fn().mockResolvedValue(undefined),
);
const listGalleryFrom = vi.hoisted(() => vi.fn().mockResolvedValue([]));
const fetchGalleryBlob = vi.hoisted(() => vi.fn());
const fetchGalleryThumbnailBlob = vi.hoisted(() => vi.fn());

vi.mock("@studio/api/generationAdmission", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/generationAdmission")>()),
  admitGenerationBatch,
  lookupGenerationBatchByClientId,
  reconcileGenerationBatches,
}));

vi.mock("@microsoft/fetch-event-source", () => ({ fetchEventSource }));

vi.mock("@studio/api/queuePlan", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/queuePlan")>()),
  mutateQueueJobOnExpectedInstance,
}));

vi.mock("../lib/galleryMedia", () => ({
  fetchGalleryBlob,
  fetchGalleryThumbnailBlob,
}));

vi.mock("../api", () => ({
  cancelQueueJob: vi.fn().mockResolvedValue(undefined),
  fetchQueue: vi.fn().mockResolvedValue({ entries: [] }),
  listGalleryFrom,
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

// ── Durable-print harness ───────────────────────────────────────────────────
//
// A print has no attached stream any more: it is admitted through
// `POST /api/generation-batches` and settles from the host's own authority.
// These helpers are the print equivalent of `lastChainHandlers`.

const durableRoute: HostRoute = {
  hostId: "render-box",
  label: "Render box",
  target: { baseUrl: "http://render-box:7680", apiKey: "secret" },
  instanceId: "instance-1",
  durableGeneration: { heterogeneous_batch_max_outputs: 64 },
  eventsAvailable: true,
};

function durableBatch(
  clientBatchId: string,
  states: Array<
    | "accepted"
    | "queued"
    | "running"
    | "held"
    | "complete"
    | "failed"
    | "cancelled"
  > = ["queued"],
  childOverrides: Record<string, unknown> = {},
): GenerationBatchStatus {
  return {
    id: `server-${clientBatchId}`,
    client_batch_id: clientBatchId,
    instance_id: "instance-1",
    durable: true,
    children: states.map((state, offset) => ({
      index: offset + 1,
      job_id: `job-${clientBatchId}-${offset + 1}`,
      state,
      created_at_ms: 10,
      updated_at_ms: 20 + offset,
      ...(state === "complete"
        ? {
            completed_at_ms: 30,
            result: { filename: `print-${offset + 1}.png` },
          }
        : {}),
      ...childOverrides,
    })),
  };
}

function durableStatusResponse(
  batches: GenerationBatchStatus[],
): GenerationBatchStatusResponse {
  return {
    instance_id: "instance-1",
    batches,
    missing: { client_batch_ids: [], batch_ids: [] },
  };
}

/** Drop every row from the module-scoped singleton between tests. */
function clearJobs(): void {
  const stream = useGenerateStream();
  for (const job of [...stream.jobs.value]) stream.remove(job.id);
  stream.canvasErrorJobId.value = null;
}

/** Drain the durable admission/hydration microtasks without moving the clock,
 * so a test holding fake timers keeps the auto-remove window to itself. */
async function flushDurable(): Promise<void> {
  for (let tick = 0; tick < 40; tick += 1) await Promise.resolve();
}

/** Admit one print durably and wait for its server identity to land. */
async function submitDurable(
  stream: ReturnType<typeof useGenerateStream>,
  request: GenerateRequestWire,
  route: HostRoute = durableRoute,
): Promise<string> {
  serveDurableArtifacts();
  admitGenerationBatch.mockImplementation(
    (_target: unknown, body: { client_batch_id: string }) =>
      Promise.resolve(durableBatch(body.client_batch_id)),
  );
  const id = stream.submit(request, { kind: "single" }, route);
  await flushDurable();
  expect(stream.jobs.value.find((job) => job.id === id)?.serverId).toBeTruthy();
  return id;
}

function galleryRow(filename: string): GalleryImage {
  return {
    filename,
    timestamp: 1,
    format: "png",
    metadata: {
      prompt: "a cat walking through autumn leaves",
      model: "ltx-2-19b-distilled:fp8",
      seed: 42,
      steps: 8,
      guidance: 3,
      width: 1216,
      height: 704,
      version: "test",
    },
  };
}

/** Serve the artifact a completed durable print hydrates its result from. */
function serveDurableArtifacts(): void {
  listGalleryFrom.mockResolvedValue([galleryRow("print-1.png")]);
  fetchGalleryBlob.mockResolvedValue(new Blob(["media"]));
  fetchGalleryThumbnailBlob.mockResolvedValue(new Blob(["thumbnail"]));
}

/** Move one admitted print to a terminal state through the host authority. */
async function settleDurable(
  stream: ReturnType<typeof useGenerateStream>,
  id: string,
  state: "complete" | "failed" | "cancelled" | "running",
  childOverrides: Record<string, unknown> = {},
  route: HostRoute = durableRoute,
): Promise<void> {
  const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
  reconcileGenerationBatches.mockResolvedValue(
    durableStatusResponse([
      durableBatch(job.durableBatch!.clientBatchId, [state], childOverrides),
    ]),
  );
  await __testing__.reconcileDurableHost(route.hostId);
  await flushDurable();
  if (state === "complete") {
    expect(stream.jobs.value.find((c) => c.id === id)?.result).toBeTruthy();
  }
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

  it("keeps a detached settle detached across a reload", () => {
    // Without this the row comes back as a plain error and the strip labels a
    // print the host finished "Failed" — the same lie, one refresh later.
    const raw = persistedPayload([
      persisted({
        id: "retained-1",
        state: "error",
        error: "mold is restarting; this generation was kept in the queue",
        serverId: "srv-1",
        detached: true,
      }),
    ]);

    const jobs = __testing__.loadPersistedJobs(raw);

    expect(jobs[0].detached).toBe(true);
  });

  it("does not invent a detached flag for an ordinary persisted failure", () => {
    const raw = persistedPayload([
      persisted({ id: "failed-1", state: "error", error: "out of memory" }),
    ]);

    expect(__testing__.loadPersistedJobs(raw)[0].detached).not.toBe(true);
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
    lastChainHandlers = null;
    __testing__.resetDurableLifecycleForTests();
    admitGenerationBatch.mockReset();
    reconcileGenerationBatches.mockReset();
    lookupGenerationBatchByClientId.mockReset();
    lookupGenerationBatchByClientId.mockResolvedValue({ kind: "missing" });
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

  it("returns to automatic canvas selection when new work is submitted", async () => {
    const stream = useGenerateStream();
    const inspected = await submitDurable(
      stream,
      singleGen({ prompt: "inspected" }),
    );
    stream.select(inspected);
    stream.canvasErrorJobId.value = "historical-failure";
    expect(stream.selectedJob.value?.id).toBe(inspected);

    await submitDurable(stream, singleGen({ prompt: "new work" }));
    expect(stream.selectedJob.value).toBeNull();
    expect(stream.canvasErrorJobId.value).toBeNull();
  });

  it("settles a reconciled missing job through the canvas error authority", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(
      stream,
      singleGen({ prompt: "lost stream" }),
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    job.previewUrl = "data:image/png;base64,preview";

    stream.failRunning(id, "job not found on server — connection lost");

    expect(job.state).toBe("error");
    expect(job.error).toBe("job not found on server — connection lost");
    expect(job.settledAt).not.toBeNull();
    expect(job.previewUrl).toBeNull();
    expect(stream.canvasErrorJobId.value).toBe(id);
  });

  it("does not let a late reconciliation response overwrite terminal state", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(
      stream,
      singleGen({ prompt: "already complete" }),
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    job.state = "done";
    job.settledAt = Date.now();

    stream.failRunning(id, "job not found on server — connection lost");

    expect(job.state).toBe("done");
    expect(job.error).toBeNull();
    expect(stream.canvasErrorJobId.value).toBeNull();
  });

  it("marks a reconciler-detached settle the same way", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(
      stream,
      singleGen({ prompt: "away while it ran" }),
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;

    stream.settleDetached(id, "check the Library for the result");

    expect(job.state).toBe("error");
    expect(job.detached).toBe(true);
  });

  it("gives a durable failure the canvas failure authority", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(
      stream,
      singleGen({ prompt: "really failed" }),
    );
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;

    await settleDurable(stream, id, "failed", {
      error: "host ran out of memory",
    });

    expect(job.error).toBe("host ran out of memory");
    expect(stream.canvasErrorJobId.value).toBe(id);
    expect(job.detached).not.toBe(true);
  });
});

describe("insecure-context compatibility", () => {
  beforeEach(() => {
    __testing__.resetDurableLifecycleForTests();
    clearJobs();
    admitGenerationBatch.mockReset();
  });

  it("submits when crypto.randomUUID is unavailable", () => {
    vi.stubGlobal("crypto", {
      getRandomValues(bytes: Uint8Array) {
        bytes.fill(7);
        return bytes;
      },
    });
    try {
      admitGenerationBatch.mockImplementation(() => new Promise(() => {}));
      const id = useGenerateStream().submit(
        singleGen(),
        { kind: "single" },
        durableRoute,
      );
      expect(id).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
      );
      expect(admitGenerationBatch).toHaveBeenCalledTimes(1);
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
    lastChainHandlers = null;
    __testing__.resetDurableLifecycleForTests();
    admitGenerationBatch.mockReset();
    reconcileGenerationBatches.mockReset();
    lookupGenerationBatchByClientId.mockReset();
    lookupGenerationBatchByClientId.mockResolvedValue({ kind: "missing" });
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

  it("auto-removes a job ~1500ms after it transitions to done", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(stream, singleGen({ frames: 1 }));
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("running");
    stream.canvasErrorJobId.value = "prior-failure";

    await settleDurable(stream, id, "complete");

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

  it("does not restore an older failure after a later success auto-removes", async () => {
    const stream = useGenerateStream();
    const failedId = await submitDurable(
      stream,
      singleGen({ prompt: "first attempt" }),
    );
    await settleDurable(stream, failedId, "failed", {
      error: "owner thread lost",
    });
    const failed = stream.jobs.value.find((job) => job.id === failedId)!;
    expect(stream.canvasErrorJobId.value).toBe(failedId);

    const successfulId = await submitDurable(
      stream,
      singleGen({ prompt: "second attempt" }),
    );
    await settleDurable(stream, successfulId, "complete");
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

  it("does NOT auto-remove a job that errors out", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(stream, singleGen({ frames: 1 }));
    await settleDurable(stream, id, "failed", { error: "boom" });
    const job = stream.jobs.value.find((j) => j.id === id);
    expect(job?.state).toBe("error");
    expect(stream.canvasErrorJobId.value).toBe(id);

    // Even well past the would-be auto-remove window, an errored card
    // sticks around for the user to read.
    vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS * 5);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeDefined();
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("error");
  });

  it("shows the host's failure message without transport noise", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(stream, singleGen({ frames: 25 }));
    await settleDurable(stream, id, "failed", {
      error:
        "generation error: LTX-2 audio output is unavailable. Set enable_audio=false and retry.",
    });

    expect(stream.jobs.value.find((j) => j.id === id)?.error).toBe(
      "generation error: LTX-2 audio output is unavailable. Set enable_audio=false and retry.",
    );
  });

  it("does NOT auto-remove a locally waiting canceled job", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(stream, singleGen({ frames: 1 }));
    stream.select(id);
    expect(stream.selectedJob.value?.id).toBe(id);
    await stream.cancel(id);
    await settleDurable(stream, id, "cancelled");
    const job = stream.jobs.value.find((j) => j.id === id);
    expect(job?.state).toBe("canceled");
    expect(stream.selectedJob.value).toBeNull();

    vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS * 5);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeDefined();
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("canceled");
  });

  it("manual remove() before the auto-remove timer is harmless", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(stream, singleGen({ frames: 1 }));
    await settleDurable(stream, id, "complete");

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

  it("keeps a completed result authoritative if cancel is clicked during its grace period", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(stream, singleGen({ frames: 1 }));
    await settleDurable(stream, id, "complete");
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

// ── Seed visuals and persistence redaction ──────────────────────────────────
//
// The live latent preview a print used to receive over its own SSE stream is
// gone with that stream; `previewUrl` now only ever arrives from the queue
// preview endpoint, which this composable does not own. What remains here is
// the persistence contract: a preview is never written to localStorage, and
// neither is media authority or biometric metadata.
describe("seed visuals and persistence redaction", () => {
  beforeEach(() => {
    __testing__.resetDurableLifecycleForTests();
    admitGenerationBatch.mockReset();
    reconcileGenerationBatches.mockReset();
    lookupGenerationBatchByClientId.mockReset();
    lookupGenerationBatchByClientId.mockResolvedValue({ kind: "missing" });
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

  it("derives seedVisual from an explicit seed", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(stream, singleGen({ frames: 1, seed: 42 }));
    const job = stream.jobs.value.find((j) => j.id === id)!;
    expect(job.seedVisual).toBe("42");
  });

  it("derives a stable model·prompt seedVisual when the seed is random", async () => {
    const stream = useGenerateStream();
    const req = singleGen({ frames: 1 });
    const id = await submitDurable(stream, req);
    const job = stream.jobs.value.find((j) => j.id === id)!;
    expect(job.seedVisual).toBe(`${req.model}·${req.prompt}`);
  });

  it("omits previewUrl from persistence and rehydrates it as null", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(stream, singleGen({ frames: 1 }));
    // A RUNNING durable row lives in its own recovery record, never the
    // shared rail, so settle it first — this is the presentation history.
    await settleDurable(stream, id, "complete");
    const job = stream.jobs.value.find((j) => j.id === id)!;
    job.previewUrl = "data:image/png;base64,UFJFVklFVw==";

    __testing__.persistJobs([job]);
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
  });

  it("keeps media authority and biometric metadata out of localStorage", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(stream, singleGen({ frames: 1 }));
    await settleDurable(stream, id, "complete");
    const job = stream.jobs.value.find((candidate) => candidate.id === id)!;
    job.request = singleGen({
      model: "minimax-h3-ref2va",
      frames: 124,
      source_image: "PRIVATE-SOURCE-BYTES",
      id_image: "PRIVATE-FACE-BYTES",
      id_image_name: "identity.png",
      id_weight: 1.5,
      id_start_step: 2,
      audio_file_path: "/private/audio.wav",
      source_video_path: "/private/source.mp4",
      references: [
        {
          kind: "image",
          media: { authority: "inline", data: "PRIVATE-IMAGE-BYTES" },
          provenance: { name: "identity.png", sha256: "BIOMETRIC-DIGEST" },
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
    expect(raw).not.toContain("PRIVATE-SOURCE-BYTES");
    expect(raw).not.toContain("PRIVATE-FACE-BYTES");
    expect(raw).not.toContain("PRIVATE-IMAGE-BYTES");
    expect(raw).not.toContain("ONE-USE-HANDLE");
    expect(raw).not.toContain("/private/audio.wav");
    expect(raw).not.toContain("/private/source.mp4");
    expect(raw).not.toContain("identity.png");
    expect(raw).not.toContain("BIOMETRIC-DIGEST");
    const restored = __testing__.loadPersistedJobs(raw)[0]!;
    expect(
      (restored.request as GenerateRequestWire).references,
    ).toBeUndefined();
    expect(
      (restored.request as GenerateRequestWire).id_image_name,
    ).toBeUndefined();
    expect((restored.request as GenerateRequestWire).id_weight).toBeUndefined();
    expect(
      (restored.request as GenerateRequestWire).id_start_step,
    ).toBeUndefined();
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
    lastChainTarget = undefined;
    __testing__.resetDurableLifecycleForTests();
    clearJobs();
    admitGenerationBatch.mockReset();
    reconcileGenerationBatches.mockReset();
    lookupGenerationBatchByClientId.mockReset();
    lookupGenerationBatchByClientId.mockResolvedValue({ kind: "missing" });
    vi.mocked(cancelQueueJob).mockReset();
    vi.mocked(cancelQueueJob).mockResolvedValue(undefined);
    mutateQueueJobOnExpectedInstance.mockReset();
    mutateQueueJobOnExpectedInstance.mockResolvedValue(undefined);
  });

  /** A machine that advertises nothing: the chain path still routes to it,
   *  and a print is refused against it by name. */
  const studioRoute: HostRoute = {
    hostId: "studio",
    label: "Studio",
    target: { baseUrl: "http://studio:7680", apiKey: "sk-studio" },
  };

  const studioDurableRoute: HostRoute = {
    ...studioRoute,
    instanceId: "instance-1",
    durableGeneration: { heterogeneous_batch_max_outputs: 64 },
    eventsAvailable: true,
  };

  it("admits a print on the routed host with its key", async () => {
    const stream = useGenerateStream();
    await submitDurable(stream, singleGen({ frames: 1 }), studioDurableRoute);
    expect(admitGenerationBatch.mock.calls[0]![0]).toEqual({
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

  it("refuses a print with no machine selected and queues nothing", () => {
    const stream = useGenerateStream();
    expect(() =>
      stream.submit(singleGen({ frames: 1 }), { kind: "single" }),
    ).toThrow("no machine is selected for this print.");
    expect(admitGenerationBatch).not.toHaveBeenCalled();
    expect(stream.jobs.value).toHaveLength(0);
  });

  it("refuses a print on a machine that has not reported its instance", () => {
    const stream = useGenerateStream();
    expect(() =>
      stream.submit(
        singleGen({ frames: 1 }),
        { kind: "single" },
        {
          ...studioRoute,
          durableGeneration: { heterogeneous_batch_max_outputs: 64 },
        },
      ),
    ).toThrow("Studio has not reported its server instance yet.");
    expect(admitGenerationBatch).not.toHaveBeenCalled();
    expect(stream.jobs.value).toHaveLength(0);
  });

  it("refuses a print on a machine with no durable queue, by name", () => {
    const stream = useGenerateStream();
    expect(() =>
      stream.submit(
        singleGen({ frames: 1 }),
        { kind: "single" },
        { ...studioRoute, instanceId: "instance-studio" },
      ),
    ).toThrow(
      "Studio cannot queue this print: this machine does not advertise the durable generation queue.",
    );
    expect(admitGenerationBatch).not.toHaveBeenCalled();
    expect(stream.jobs.value).toHaveLength(0);
  });

  it("attributes the job to the host it was routed to", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(
      stream,
      singleGen({ frames: 1 }),
      studioDurableRoute,
    );
    const job = stream.jobs.value.find((j) => j.id === id);
    expect(job?.hostId).toBe("studio");
    expect(job?.hostLabel).toBe("Studio");
  });

  it("cancels a print on the exact machine that admitted it", async () => {
    const stream = useGenerateStream();
    const id = await submitDurable(
      stream,
      singleGen({ frames: 1 }),
      studioDurableRoute,
    );

    await stream.cancel(id);

    expect(mutateQueueJobOnExpectedInstance).toHaveBeenCalledWith(
      { baseUrl: "http://studio:7680", apiKey: "sk-studio" },
      expect.objectContaining({ instanceId: "instance-1" }),
      "cancel",
    );
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("canceled");
  });

  it("repaints as cancelling before the host acknowledges a running job", async () => {
    let acknowledge!: () => void;
    mutateQueueJobOnExpectedInstance.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          acknowledge = resolve;
        }),
    );
    const stream = useGenerateStream();
    const id = await submitDurable(
      stream,
      singleGen({ model: "wan22-i2v-a14b:q4", frames: 81 }),
      studioDurableRoute,
    );

    const cancellation = stream.cancel(id);
    expect(stream.jobs.value.find((job) => job.id === id)?.cancelling).toBe(
      true,
    );

    acknowledge();
    await cancellation;
    expect(stream.jobs.value.find((job) => job.id === id)).toMatchObject({
      cancelling: false,
      state: "canceled",
    });
  });

  it("keeps a server-owned job running when cancellation is refused", async () => {
    mutateQueueJobOnExpectedInstance.mockRejectedValueOnce(
      new Error("already running"),
    );
    const stream = useGenerateStream();
    const id = await submitDurable(
      stream,
      singleGen({ frames: 1 }),
      studioDurableRoute,
    );

    await expect(stream.cancel(id)).rejects.toThrow("already running");

    const job = stream.jobs.value.find((candidate) => candidate.id === id);
    expect(job?.state).toBe("running");
    expect(job?.cancelling).toBe(false);
    expect(job?.controller.signal.aborted).toBe(false);
  });

  it("keeps an opened stream live until its queue id can be cancelled", async () => {
    const stream = useGenerateStream();
    const id = stream.submit(
      singleGen({ frames: 241 }),
      chainDecision(),
      studioRoute,
    );

    await expect(stream.cancel(id)).rejects.toThrow(
      "Remote cancellation was not confirmed before the queue ID arrived.",
    );

    const job = stream.jobs.value.find((candidate) => candidate.id === id);
    expect(job?.state).toBe("running");
    expect(job?.controller.signal.aborted).toBe(false);
  });

  it("leaves an unrouted sequence unattributed rather than guessing", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 241 }), chainDecision());
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
