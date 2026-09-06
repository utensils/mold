import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiHttpError,
  cancelChainJob,
  chainJobEventsUrl,
  createChainJob,
  deleteModel,
  fetchQueue,
  upscaleStream,
  updateQueueJobTargetGpu,
  type UpscaleStreamHandlers,
} from "./api";
import type { ChainRequestWire, UpscaleRequestWire } from "./types";
import type { SseEvent, StreamSseOptions } from "./lib/sse";

// streamSse is the I/O surface; mocking it lets us drive the SSE lifecycle
// (deliver-this-event, close-cleanly, close-without-terminal) deterministically
// without spinning up a real fetch. We reach into the module so each test can
// install its own driver implementation.
vi.mock("./lib/sse", () => ({
  streamSse: vi.fn(),
}));

import { streamSse } from "./lib/sse";

function upscaleHandlers() {
  return {
    onProgress: vi.fn<UpscaleStreamHandlers["onProgress"]>(),
    onComplete: vi.fn<UpscaleStreamHandlers["onComplete"]>(),
    onError: vi.fn<UpscaleStreamHandlers["onError"]>(),
  };
}

/** What `useGenerateStream` sends when one render is longer than the
 *  checkpoint's single-pass clip size: ONE prompt, `ephemeral: true`, and no
 *  authored stages — web has no other chain body left. */
function chainRequest(): ChainRequestWire {
  return {
    model: "ltx-2-19b-distilled:fp8",
    ephemeral: true,
    output_mode: "one-shot",
    prompt: "a cat",
    total_frames: 200,
    clip_frames: 97,
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
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ queue_capacity: 2 }))
      .mockResolvedValueOnce(
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

    expect(fetchMock).toHaveBeenCalledWith("/api/queue?limit=2", {
      signal: undefined,
    });
    expect(listing.entries[0].target_gpu).toBe(1);
    expect(listing.page).toBeUndefined();
  });

  it("fetches an encoded page from the exact authenticated target with its signal", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        entries: [],
        live_only_entries: [],
        page: {
          limit: 11,
          offset: 0,
          returned: 0,
          next_cursor: "next-page",
        },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    const listing = await fetchQueue(
      { baseUrl: "https://render.example", apiKey: "secret" },
      controller.signal,
      { limit: 11, cursor: "opaque/+ token=" },
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "https://render.example/api/queue?limit=11&cursor=opaque%2F%2B+token%3D",
      {
        headers: { "x-api-key": "secret" },
        signal: controller.signal,
      },
    );
    expect(listing.page?.next_cursor).toBe("next-page");
  });

  it("rejects an invalid page limit without making a request", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      fetchQueue(undefined, undefined, { limit: 0 }),
    ).rejects.toThrow("positive integer");
    expect(fetchMock).not.toHaveBeenCalled();
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

describe("auto-chained long video", () => {
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

  it("creates the ephemeral job, cancels it, and names its event stream", async () => {
    const fetchMock = installFetch({ job_id: "job/1" });
    await createChainJob(chainRequest(), undefined, "create-op");
    expect(fetchMock).toHaveBeenLastCalledWith("/api/chain-jobs", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-mold-operation-id": "create-op",
      },
      body: JSON.stringify(chainRequest()),
    });

    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          id: "job/1",
          state: "cancelled",
          model: "ltx-2",
          stage_count: 3,
          current_stage: 1,
          created_at_unix_ms: 1,
          updated_at_unix_ms: 2,
          error: null,
          ephemeral: true,
        }),
      ),
    );
    await cancelChainJob("job/1");
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/chain-jobs/job%2F1/cancel",
      { method: "POST", headers: {} },
    );

    expect(chainJobEventsUrl("job/1")).toBe("/api/chain-jobs/job%2F1/events");
  });

  it("surfaces request warnings the host attached to the accepted job", async () => {
    const fetchMock = installFetch({ job_id: "job/2" });
    const onRequestWarnings = vi.fn();
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ job_id: "job/2" }), {
        headers: {
          "x-mold-request-warning": "retimed clip; output was still created",
        },
      }),
    );
    await createChainJob(
      chainRequest(),
      undefined,
      undefined,
      onRequestWarnings,
    );
    expect(onRequestWarnings).toHaveBeenCalledWith([
      "retimed clip; output was still created",
    ]);
  });

  it("preserves typed HTTP errors from a refused cancel", async () => {
    installFetch({ error: "job already settled" }, 409);

    const error = await cancelChainJob("job/1").catch(
      (cause: unknown) => cause,
    );

    expect(error).toBeInstanceOf(ApiHttpError);
    expect(error).toMatchObject({
      message:
        'POST /api/chain-jobs/job/1/cancel failed: 409 {"error":"job already settled"}',
      status: 409,
    });
  });

  it("keeps creation and its event stream on an authenticated remote host", async () => {
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

  it("exposes no authoring endpoint any more", async () => {
    const api = await import("./api");
    for (const gone of [
      "fetchChainLimits",
      "validateChain",
      "listChainJobs",
      "getChainJob",
      "resumeChainJob",
      "retakeChainJob",
      "amendChainJob",
      "cancelChainJobMutation",
      "deleteChainJob",
      "gcChainJobs",
      "chainJobStagePreviewUrl",
    ]) {
      expect(api).not.toHaveProperty(gone);
    }
  });
});
