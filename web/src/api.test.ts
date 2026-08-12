import { afterEach, describe, expect, it, vi } from "vitest";
import {
  amendChainJob,
  ApiHttpError,
  cancelChainJob,
  chainJobEventsUrl,
  chainJobStagePreviewUrl,
  createChainJob,
  deleteChainJob,
  deleteModel,
  fetchQueue,
  gcChainJobs,
  generateStream,
  generateChainStream,
  fetchChainLimits,
  getChainJob,
  listChainJobs,
  resumeChainJob,
  retakeChainJob,
  upscaleStream,
  updateQueueJobTargetGpu,
  validateChain,
  type ChainStreamHandlers,
  type GenerateStreamHandlers,
  type UpscaleStreamHandlers,
} from "./api";
import type {
  ChainProgressEvent,
  ChainRequestWire,
  GenerateRequestWire,
  SseChainCompleteEvent,
  SseCompleteEvent,
  SseProgressEvent,
  UpscaleRequestWire,
} from "./types";
import type { SseEvent, StreamSseOptions } from "./lib/sse";

// streamSse is the I/O surface; mocking it lets us drive the SSE lifecycle
// (deliver-this-event, close-cleanly, close-without-terminal) deterministically
// without spinning up a real fetch. We reach into the module so each test can
// install its own driver implementation.
vi.mock("./lib/sse", () => ({
  streamSse: vi.fn(),
}));

import { streamSse } from "./lib/sse";

// `vi.fn<T>()` types the resulting Mock against the target signature,
// which lets both `generateStream(req, h)` (handlers must satisfy the
// union types) and `h.onError.mock.calls` (assertions need Mock's
// `.mock`) compile cleanly without `as unknown as` gymnastics.
function singleHandlers() {
  return {
    onProgress: vi.fn<GenerateStreamHandlers["onProgress"]>(),
    onComplete: vi.fn<GenerateStreamHandlers["onComplete"]>(),
    onError: vi.fn<GenerateStreamHandlers["onError"]>(),
  };
}

function chainHandlers() {
  return {
    onProgress: vi.fn<ChainStreamHandlers["onProgress"]>(),
    onComplete: vi.fn<ChainStreamHandlers["onComplete"]>(),
    onError: vi.fn<ChainStreamHandlers["onError"]>(),
  };
}

function upscaleHandlers() {
  return {
    onProgress: vi.fn<UpscaleStreamHandlers["onProgress"]>(),
    onComplete: vi.fn<UpscaleStreamHandlers["onComplete"]>(),
    onError: vi.fn<UpscaleStreamHandlers["onError"]>(),
  };
}

function singleRequest(): GenerateRequestWire {
  return {
    prompt: "a cat",
    model: "flux-dev:fp16",
    width: 512,
    height: 512,
    steps: 8,
    guidance: 3.0,
    strength: 1.0,
    fps: 24,
    output_format: "png",
    frames: 1,
  } as GenerateRequestWire;
}

function chainRequest(): ChainRequestWire {
  return {
    model: "ltx-2-19b-distilled:fp8",
    stages: [{ prompt: "a cat", frames: 97, transition: "smooth" }],
  } as ChainRequestWire;
}

function upscaleRequest(): UpscaleRequestWire {
  return {
    image: "AAAA",
    model: "real-esrgan-x4plus:fp16",
    scale: 4,
    output_format: "png",
  } as UpscaleRequestWire;
}

/** Helper: install a streamSse fake that runs `driver` with the caller's
 * onEvent / onHttpError, then resolves with a 200 Response. */
function installDriver(
  driver: (
    onEvent: (evt: SseEvent) => void,
    onHttpError?: (res: Response) => void,
  ) => void | Promise<void>,
) {
  vi.mocked(streamSse).mockImplementationOnce(
    async (opts: StreamSseOptions<unknown>) => {
      await driver(opts.onEvent, opts.onHttpError);
      return new Response("", { status: 200 });
    },
  );
}

afterEach(() => {
  vi.mocked(streamSse).mockReset();
  vi.unstubAllGlobals();
});

describe("queue api", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fetchQueue preserves queued target_gpu metadata", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          entries: [
            {
              id: "srv-1",
              model: "flux-dev:q4",
              state: "queued",
              started_at_unix_ms: 0,
              position: 0,
              target_gpu: 1,
            },
          ],
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const listing = await fetchQueue();

    expect(fetchMock).toHaveBeenCalledWith("/api/queue", { signal: undefined });
    expect(listing.entries[0].target_gpu).toBe(1);
  });

  it("PATCHes queued job target_gpu including null for Auto", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "srv-1",
          model: "flux-dev:q4",
          state: "queued",
          started_at_unix_ms: 0,
          position: 0,
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await updateQueueJobTargetGpu("srv-1", null);

    expect(fetchMock).toHaveBeenCalledWith("/api/queue/srv-1", {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ target_gpu: null }),
      signal: undefined,
    });
  });
});

describe("chain validation api", () => {
  it("requests chain limits for the selected fps", async () => {
    const limits = {
      model: "ltx-2-19b-distilled:fp8",
      frames_per_clip_cap: 241,
      frames_per_clip_recommended: 97,
      max_stages: 16,
      max_total_frames: 3856,
      fade_frames_max: 32,
      transition_modes: ["smooth"],
      quantization_family: "fp8",
      supports_audio: true,
      supports_sequence: true,
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify(limits), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      fetchChainLimits(limits.model, undefined, 12),
    ).resolves.toEqual(limits);
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/capabilities/chain-limits?model=${encodeURIComponent(limits.model)}&fps=12`,
      { headers: {} },
    );
  });

  it("validates on the exact authenticated host without creating a job", async () => {
    const response = {
      model: "ltx-2-19b-distilled:fp8",
      width: 1216,
      height: 704,
      fps: 24,
      motion_tail_frames: 17,
      stage_count: 1,
      estimated_total_frames: 97,
      estimated_duration_ms: 4042,
      stages: [
        {
          prompt: "a cat",
          frames: 97,
          output_frames: 97,
          transition: "smooth",
          fade_frames: null,
          has_source_image: false,
          has_negative_prompt: false,
        },
      ],
      warnings: [],
      vram_estimate: null,
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify(response), { status: 200 }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const result = await validateChain(chainRequest(), {
      baseUrl: "http://render-box:7680",
      apiKey: "secret",
    });

    expect(result).toEqual(response);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://render-box:7680/api/generate/chain/validate",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "content-type": "application/json",
          "x-api-key": "secret",
        }),
        body: JSON.stringify(chainRequest()),
      }),
    );
  });

  it("surfaces the server's structured validation detail without raw JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            error: "motion_tail_frames must be less than clip 2 frames",
            code: "VALIDATION_ERROR",
          }),
          { status: 422 },
        ),
      ),
    );

    const error = await validateChain(chainRequest()).catch(
      (candidate: unknown) => candidate,
    );
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).toContain(
      "motion_tail_frames must be less than clip 2 frames",
    );
    expect((error as Error).message).not.toContain("VALIDATION_ERROR");
  });
});

describe("generateStream", () => {
  it("flips a job to network error when the stream resolves without complete/error", async () => {
    // Server-side scenario: the worker crashed mid-job, or a proxy /
    // network blip closed the SSE pipe after a few progress events.
    // The reader exits cleanly with no `complete` or `error` event.
    // The browser used to see this as "still running" forever because
    // streamSse just resolves; this test locks in the new behavior.
    installDriver((onEvent) => {
      onEvent({
        event: null,
        data: JSON.stringify({
          type: "denoise_step",
          step: 3,
          total: 30,
          elapsed_ms: 100,
        } as SseProgressEvent),
      });
      // No terminal event — driver returns and streamSse resolves.
    });

    const h = singleHandlers();
    await generateStream(singleRequest(), h);

    expect(h.onProgress).toHaveBeenCalledOnce();
    expect(h.onComplete).not.toHaveBeenCalled();
    expect(h.onError).toHaveBeenCalledOnce();
    const err = h.onError.mock.calls[0][0];
    if (err.kind !== "network") throw new Error("expected network error");
    expect(String(err.message).toLowerCase()).toMatch(
      /close|disconnect|complet/,
    );
  });

  it("does NOT synthesize a network error when complete fires", async () => {
    // Happy path — the stream closes after a complete event. This must
    // not flip the job back into error.
    const completeEvt: SseCompleteEvent = {
      image: "AAAA",
      format: "png",
      width: 512,
      height: 512,
      seed_used: 42,
      generation_time_ms: 1234,
      model: "flux-dev:fp16",
    } as SseCompleteEvent;

    installDriver((onEvent) => {
      onEvent({ event: "complete", data: JSON.stringify(completeEvt) });
    });

    const h = singleHandlers();
    await generateStream(singleRequest(), h);

    expect(h.onComplete).toHaveBeenCalledOnce();
    expect(h.onError).not.toHaveBeenCalled();
  });

  it("does NOT synthesize a network error when an error event fires", async () => {
    // Server emitted an error event before closing — the explicit error
    // is the terminal signal; the stream-close that follows is expected.
    installDriver((onEvent) => {
      onEvent({
        event: "error",
        data: JSON.stringify({ message: "OOM" }),
      });
    });

    const h = singleHandlers();
    await generateStream(singleRequest(), h);

    expect(h.onError).toHaveBeenCalledOnce();
    // Single call only — the silent-close path must not double-fire.
  });

  it("does NOT synthesize a network error when the signal aborts", async () => {
    // User clicked Cancel — the AbortController fires and streamSse
    // throws an AbortError, which the wrapper swallows. We must not
    // turn that into a network error (the UI already shows "canceled").
    const controller = new AbortController();
    vi.mocked(streamSse).mockImplementationOnce(async () => {
      controller.abort();
      const err = new Error("aborted");
      err.name = "AbortError";
      throw err;
    });

    const h = singleHandlers();
    await generateStream(singleRequest(), h, controller.signal);
    expect(h.onError).not.toHaveBeenCalled();
  });

  it("fires a single error on HTTP 503 (queue full) and skips the silent-close synthesis", async () => {
    installDriver((_onEvent, onHttpError) => {
      const res = new Response("queue full", {
        status: 503,
        headers: { "Retry-After": "2" },
      });
      onHttpError?.(res);
    });

    const h = singleHandlers();
    await generateStream(singleRequest(), h);

    // onHttpError schedules a .then(); flush microtasks.
    await Promise.resolve();
    await Promise.resolve();

    expect(h.onError).toHaveBeenCalledOnce();
    const err = h.onError.mock.calls[0][0];
    if (err.kind !== "http") throw new Error("expected http error");
    expect(err.status).toBe(503);
  });
});

describe("generateChainStream", () => {
  it("flips a chain job to network error on silent close", async () => {
    installDriver((onEvent) => {
      onEvent({
        event: null,
        data: JSON.stringify({
          type: "denoise_step",
          stage_idx: 0,
          step: 1,
          total: 8,
        } as ChainProgressEvent),
      });
    });

    const h = chainHandlers();
    await generateChainStream(chainRequest(), h);

    expect(h.onError).toHaveBeenCalledOnce();
    expect(h.onError.mock.calls[0][0].kind).toBe("network");
  });

  it("does not synthesize a network error after a chain complete", async () => {
    const evt: SseChainCompleteEvent = {
      video: "AAAA",
      format: "mp4",
      width: 1216,
      height: 704,
      frames: 97,
      fps: 24,
      generation_time_ms: 8000,
    } as SseChainCompleteEvent;

    installDriver((onEvent) => {
      onEvent({ event: "complete", data: JSON.stringify(evt) });
    });

    const h = chainHandlers();
    await generateChainStream(chainRequest(), h);
    expect(h.onComplete).toHaveBeenCalledOnce();
    expect(h.onError).not.toHaveBeenCalled();
  });
});

describe("upscaleStream", () => {
  it("preserves Retry-After on HTTP errors", async () => {
    installDriver((_onEvent, onHttpError) => {
      const res = new Response("busy", {
        status: 503,
        headers: { "Retry-After": "1.5" },
      });
      onHttpError?.(res);
    });

    const h = upscaleHandlers();
    await upscaleStream(upscaleRequest(), h);

    await Promise.resolve();
    await Promise.resolve();

    expect(h.onError).toHaveBeenCalledOnce();
    const err = h.onError.mock.calls[0][0];
    if (err.kind !== "http") throw new Error("expected http error");
    expect(err.status).toBe(503);
    expect(err.retryAfter).toBe(1.5);
  });

  it("routes preprocessing to the selected host with its API key", async () => {
    installDriver(() => undefined);
    await upscaleStream(upscaleRequest(), upscaleHandlers(), undefined, {
      baseUrl: "http://studio:7680",
      apiKey: "sk-studio",
    });
    expect(streamSse).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "http://studio:7680/api/upscale/stream",
        headers: { "x-api-key": "sk-studio" },
      }),
    );
  });
});

describe("model lifecycle api helpers", () => {
  it("turns MODEL_LOADED into an actionable delete error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            code: "MODEL_LOADED",
            error: "model is currently loaded; unload it first",
          }),
          { status: 409, headers: { "content-type": "application/json" } },
        ),
      ),
    );

    await expect(deleteModel("flux-dev:q8")).rejects.toThrow(
      "Unload flux-dev:q8 before deleting it.",
    );
  });
});

describe("chain job api helpers", () => {
  function installFetch(body: unknown = {}, status = 200) {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    return fetchMock;
  }

  it("creates, lists, gets, resumes, retakes, cancels, deletes, and GC's jobs", async () => {
    const summary = {
      id: "job/1",
      state: "queued",
      model: "ltx-2",
      stage_count: 2,
      current_stage: 0,
      created_at_unix_ms: 1,
      updated_at_unix_ms: 2,
      error: null,
      ephemeral: false,
    };
    const detail = {
      ...summary,
      stages: [],
      finalizes: [],
      retakes: [],
      script: { schema: "mold.chain.v1", chain: {}, stage: [] },
    };
    const fetchMock = installFetch({ job_id: "job/1" });
    await createChainJob(chainRequest());
    expect(fetchMock).toHaveBeenLastCalledWith("/api/chain-jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(chainRequest()),
    });

    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ jobs: [summary] })),
    );
    await expect(listChainJobs()).resolves.toEqual({ jobs: [summary] });
    expect(fetchMock).toHaveBeenLastCalledWith("/api/chain-jobs", {
      headers: {},
    });

    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(detail)));
    await expect(getChainJob("job/1")).resolves.toEqual(detail);
    expect(fetchMock).toHaveBeenLastCalledWith("/api/chain-jobs/job%2F1", {
      headers: {},
    });

    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(summary)));
    await resumeChainJob("job/1");
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/chain-jobs/job%2F1/resume",
      {
        method: "POST",
        headers: {},
      },
    );

    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(summary)));
    await retakeChainJob("job/1", { stage_idx: 1, mode: "splice" });
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/chain-jobs/job%2F1/retake",
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ stage_idx: 1, mode: "splice" }),
      },
    );

    const amendRequest = { stages: chainRequest().stages ?? [], steps: 12 };
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ ...summary, preserved_stages: 1 })),
    );
    await amendChainJob("job/1", amendRequest);
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/chain-jobs/job%2F1/amend",
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(amendRequest),
      },
    );

    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify(summary)));
    await cancelChainJob("job/1");
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/chain-jobs/job%2F1/cancel",
      {
        method: "POST",
        headers: {},
      },
    );

    fetchMock.mockResolvedValueOnce(new Response("", { status: 204 }));
    await deleteChainJob("job/1");
    expect(fetchMock).toHaveBeenLastCalledWith("/api/chain-jobs/job%2F1", {
      method: "DELETE",
      headers: {},
    });

    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify({ swept_ephemeral_jobs: 1, pruned_artifact_dirs: 2 }),
      ),
    );
    await expect(gcChainJobs()).resolves.toEqual({
      swept_ephemeral_jobs: 1,
      pruned_artifact_dirs: 2,
    });
    expect(fetchMock).toHaveBeenLastCalledWith("/api/chain-jobs/gc", {
      method: "POST",
      headers: {},
    });
  });

  it("preserves typed HTTP errors from durable sequence requests", async () => {
    installFetch({ error: "job moved on" }, 409);

    const error = await amendChainJob("job/1", {
      stages: chainRequest().stages ?? [],
    }).catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(ApiHttpError);
    expect(error).toMatchObject({
      message:
        'POST /api/chain-jobs/job/1/amend failed: 409 {"error":"job moved on"}',
      status: 409,
    });
  });

  it("encodes chain job event and preview URLs", () => {
    expect(chainJobEventsUrl("job/1")).toBe("/api/chain-jobs/job%2F1/events");
    expect(chainJobStagePreviewUrl("job/1", 12)).toBe(
      "/api/chain-jobs/job%2F1/stages/12/preview",
    );
  });

  it("keeps every durable sequence request on an authenticated remote host", async () => {
    const fetchMock = installFetch({ job_id: "remote-1" });
    const target = { baseUrl: "http://plato:7680", apiKey: "secret" };

    await createChainJob(chainRequest(), target);
    expect(fetchMock).toHaveBeenLastCalledWith(
      "http://plato:7680/api/chain-jobs",
      expect.objectContaining({
        headers: {
          "content-type": "application/json",
          "x-api-key": "secret",
        },
      }),
    );
    expect(chainJobEventsUrl("remote-1", target)).toBe(
      "http://plato:7680/api/chain-jobs/remote-1/events",
    );
  });
});
