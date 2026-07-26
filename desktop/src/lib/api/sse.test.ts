import { beforeEach, describe, expect, it, vi } from "vitest";

const fetchEventSource = vi.hoisted(() => vi.fn());

vi.mock("@microsoft/fetch-event-source", () => ({ fetchEventSource }));

import { sseStream } from "./sse";

beforeEach(() => {
  fetchEventSource.mockReset();
});

describe("sseStream", () => {
  it("merges caller headers with SSE defaults and host authentication", async () => {
    fetchEventSource.mockResolvedValue(undefined);

    await sseStream("/api/generate/stream", {
      method: "POST",
      body: { prompt: "queued print" },
      signal: new AbortController().signal,
      onEvent: vi.fn(),
      headers: { "X-Mold-SSE-Payload": "metadata-only" },
      target: { baseUrl: "http://studio:7680", apiKey: "secret" },
    });

    expect(fetchEventSource).toHaveBeenCalledWith(
      "http://studio:7680/api/generate/stream",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          Accept: "text/event-stream",
          "Content-Type": "application/json",
          "X-Api-Key": "secret",
          "X-Mold-SSE-Payload": "metadata-only",
        }),
      }),
    );
  });

  it("reports every successful connection so consumers can refetch after reconnects", async () => {
    const onOpen = vi.fn();
    fetchEventSource.mockImplementation(async (_url: string, options: { onopen: Function }) => {
      const response = new Response(null, { status: 200 });
      await options.onopen(response);
      await options.onopen(response);
    });

    await sseStream("/api/downloads/stream", {
      signal: new AbortController().signal,
      onEvent: vi.fn(),
      onOpen,
      target: { baseUrl: "http://127.0.0.1:7680", apiKey: null },
    });

    expect(onOpen).toHaveBeenCalledTimes(2);
  });
});
