import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __testing__,
  isPrebuiltChainRequest,
  resolveChainRequest,
  useGenerateStream,
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
    try {
      localStorage.removeItem("mold.generate.jobs");
    } catch {
      /* ignore — happy-dom should have it */
    }
    const stream = useGenerateStream();
    for (const j of stream.jobs.value) {
      if (j.state === "running") stream.cancel(j.id);
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
    // Cancel anything still "running" then drop everything settled.
    for (const j of stream.jobs.value) {
      if (j.state === "running") stream.cancel(j.id);
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

    // Fire the SSE complete callback the singleton registered.
    expect(lastSingleHandlers).not.toBeNull();
    lastSingleHandlers!.onComplete(fakeCompleteEvent({ seed_used: 7 }));

    // Job is "done" but still on screen during the grace period.
    const job = stream.jobs.value.find((j) => j.id === id);
    expect(job?.state).toBe("done");

    // Just before the timer — still present.
    vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS - 1);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeDefined();

    // Tick past the timer — gone.
    vi.advanceTimersByTime(2);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeUndefined();
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

    // Even well past the would-be auto-remove window, an errored card
    // sticks around for the user to read.
    vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS * 5);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeDefined();
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("error");
  });

  it("does NOT auto-remove a canceled job", () => {
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    stream.cancel(id);
    const job = stream.jobs.value.find((j) => j.id === id);
    expect(job?.state).toBe("canceled");

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

  it("does NOT auto-remove if user cancels during the grace period (no flash)", () => {
    // Regression: prior code unconditionally removed at +1500ms, so a
    // cancel landing between done-flip and timer-fire would briefly show
    // "canceled" on the card and then auto-dismiss anyway, losing the
    // user's signal. Timer must re-check `state === "done"` at fire time.
    const stream = useGenerateStream();
    const id = stream.submit(singleGen({ frames: 1 }), { kind: "single" });
    expect(lastSingleHandlers).not.toBeNull();
    lastSingleHandlers!.onComplete(fakeCompleteEvent());
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("done");

    // User clicks Cancel during the 1500ms grace window.
    vi.advanceTimersByTime(500);
    stream.cancel(id);
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("canceled");

    // Timer fires; job must still be present and still in `canceled`.
    vi.advanceTimersByTime(__testing__.AUTO_REMOVE_DONE_MS + 100);
    expect(stream.jobs.value.find((j) => j.id === id)).toBeDefined();
    expect(stream.jobs.value.find((j) => j.id === id)?.state).toBe("canceled");
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
