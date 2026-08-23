import { afterEach, describe, expect, it, vi } from "vitest";
import {
  selectedQueueGeneration,
  settingsRestoreMetadata,
  watchSelectedQueuePreview,
} from "./generationSelection";

describe("generation settings selection", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("uses one metadata restore path for Library and queue selection", () => {
    const metadata = { prompt: "red dunes", seed: 42, steps: 18 };
    expect(settingsRestoreMetadata(metadata)).toEqual(metadata);
    expect(
      selectedQueueGeneration<typeof metadata>(
        [
          {
            id: "job-1",
            model: "flux-dev:q4",
            state: "running",
            started_at_unix_ms: 1,
            position: 0,
            seed_pinned: false,
            metadata,
          },
        ],
        "job-1",
      ),
    ).toEqual({
      metadata: { ...metadata, seed: null },
      jobId: "job-1",
      running: true,
    });
    expect(metadata.seed).toBe(42);
  });

  it("publishes selected-job previews and stops polling", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(
      async () =>
        new Response(
          JSON.stringify({ image: "UFJFVklFVw==", step: 4, total: 20 }),
          {
            status: 200,
            headers: { "content-type": "application/json" },
          },
        ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const onPreview = vi.fn();
    const stop = watchSelectedQueuePreview(
      { baseUrl: "https://gpu.example", apiKey: "secret" },
      "job/1",
      onPreview,
      500,
    );
    await vi.runOnlyPendingTimersAsync();
    expect(fetchMock).toHaveBeenCalledWith(
      "https://gpu.example/api/queue/job%2F1/preview",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(onPreview).toHaveBeenCalledWith({
      image: "UFJFVklFVw==",
      step: 4,
      total: 20,
    });
    stop();
    const calls = fetchMock.mock.calls.length;
    await vi.advanceTimersByTimeAsync(1_000);
    expect(fetchMock).toHaveBeenCalledTimes(calls);
  });

  it("releases the selected canvas when the live queue row disappears", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify({ error: "gone" }), {
            status: 404,
            headers: { "content-type": "application/json" },
          }),
      ),
    );
    const onEnded = vi.fn();
    watchSelectedQueuePreview(
      { baseUrl: "https://gpu.example", apiKey: null },
      "finished-job",
      vi.fn(),
      500,
      onEnded,
    );

    await vi.runOnlyPendingTimersAsync();
    expect(onEnded).toHaveBeenCalledOnce();
  });

  it("keeps polling while a live queue row has not emitted a preview", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(
      async () =>
        new Response("null", {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const onEnded = vi.fn();
    const stop = watchSelectedQueuePreview(
      { baseUrl: "https://gpu.example", apiKey: null },
      "loading-job",
      vi.fn(),
      500,
      onEnded,
    );

    await vi.advanceTimersByTimeAsync(1_100);
    expect(fetchMock.mock.calls.length).toBeGreaterThan(1);
    expect(onEnded).not.toHaveBeenCalled();
    stop();
  });
});
