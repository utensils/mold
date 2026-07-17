import { beforeEach, describe, expect, it, vi } from "vitest";

const fetchEventSource = vi.hoisted(() => vi.fn());

vi.mock("@microsoft/fetch-event-source", () => ({ fetchEventSource }));

import { sseStream } from "./sse";

beforeEach(() => {
  fetchEventSource.mockReset();
});

describe("sseStream", () => {
  it("reports the first successful connection only once across reconnects", async () => {
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

    expect(onOpen).toHaveBeenCalledTimes(1);
  });
});
