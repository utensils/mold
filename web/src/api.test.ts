import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchQueue,
  generateStream,
  generateChainStream,
  updateQueueJobTargetGpu,
  type ChainStreamHandlers,
  type GenerateStreamHandlers,
} from "./api";
import type {
  ChainProgressEvent,
  ChainRequestWire,
  GenerateRequestWire,
  SseChainCompleteEvent,
  SseCompleteEvent,
  SseProgressEvent,
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
