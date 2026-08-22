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
    expect(onOpen).toHaveBeenLastCalledWith(expect.any(Response));
  });

  for (const status of [401, 403, 404]) {
    it(`makes HTTP ${status} terminal when the consumer marks it non-retryable`, async () => {
      fetchEventSource.mockImplementation(
        async (
          _url: string,
          options: {
            onopen: (response: Response) => Promise<void>;
            onerror: (error: Error) => void;
          },
        ) => {
          let failure: Error;
          try {
            await options.onopen(new Response(null, { status }));
            throw new Error("expected onopen to reject");
          } catch (error) {
            failure = error as Error;
          }
          expect(() => options.onerror(failure)).toThrow(`HTTP ${status}`);
        },
      );

      await sseStream("/api/events", {
        signal: new AbortController().signal,
        onEvent: vi.fn(),
        retry: true,
        terminalHttpStatuses: [401, 403, 404],
        target: { baseUrl: "http://host", apiKey: "stale" },
      });

      expect(fetchEventSource).toHaveBeenCalledTimes(1);
    });
  }

  it("preserves a structured server validation message on an SSE HTTP error", async () => {
    fetchEventSource.mockImplementation(
      async (_url: string, options: { onopen: (response: Response) => Promise<void> }) => {
        await options.onopen(
          new Response(
            JSON.stringify({
              code: "VALIDATION_ERROR",
              error: "Qwen Image Edit needs at least one image. Add a Target image and try again.",
            }),
            { status: 422, headers: { "content-type": "application/json" } },
          ),
        );
      },
    );
    const onClose = vi.fn();

    await sseStream("/api/generate/stream", {
      method: "POST",
      body: {},
      signal: new AbortController().signal,
      onEvent: vi.fn(),
      onClose,
      retry: false,
      target: { baseUrl: "http://plato", apiKey: null },
    });

    expect(onClose).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Qwen Image Edit needs at least one image. Add a Target image and try again.",
      }),
    );
  });
});
