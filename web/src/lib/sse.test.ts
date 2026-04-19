import { describe, expect, it, vi } from "vitest";
import { streamSse, type SseEvent } from "./sse";

/** Build a Response whose body streams the given string in the given chunk sizes. */
function mockFetchStreaming(
  body: string,
  opts: { status?: number; chunkSize?: number } = {},
) {
  const { status = 200, chunkSize = body.length } = opts;
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (let i = 0; i < body.length; i += chunkSize) {
        controller.enqueue(encoder.encode(body.slice(i, i + chunkSize)));
      }
      controller.close();
    },
  });
  return vi.fn(
    async () =>
      new Response(stream, {
        status,
        headers: { "Content-Type": "text/event-stream" },
      }),
  );
}

describe("streamSse", () => {
  it("parses named events and routes them to onEvent in order", async () => {
    const body =
      "event: progress\n" +
      'data: {"step":1}\n' +
      "\n" +
      "event: complete\n" +
      'data: {"ok":true}\n' +
      "\n";
    globalThis.fetch = mockFetchStreaming(body) as typeof fetch;

    const events: SseEvent[] = [];
    await streamSse({
      url: "/api/generate/stream",
      body: {},
      onEvent: (e) => events.push(e),
    });

    expect(events).toEqual([
      { event: "progress", data: '{"step":1}' },
      { event: "complete", data: '{"ok":true}' },
    ]);
  });

  it("concatenates multi-line data fields with '\\n'", async () => {
    const body =
      "event: info\n" +
      "data: line one\n" +
      "data: line two\n" +
      "data: line three\n" +
      "\n";
    globalThis.fetch = mockFetchStreaming(body) as typeof fetch;

    const events: SseEvent[] = [];
    await streamSse({
      url: "/x",
      body: {},
      onEvent: (e) => events.push(e),
    });

    expect(events).toHaveLength(1);
    expect(events[0].data).toBe("line one\nline two\nline three");
  });

  it("handles CRLF line endings and comment lines", async () => {
    // Server may emit `\r\n` (per spec) and `:keepalive` heartbeat lines.
    const body =
      ":keepalive\r\n" + "event: progress\r\n" + "data: hi\r\n" + "\r\n";
    globalThis.fetch = mockFetchStreaming(body) as typeof fetch;

    const events: SseEvent[] = [];
    await streamSse({
      url: "/x",
      body: {},
      onEvent: (e) => events.push(e),
    });

    expect(events).toEqual([{ event: "progress", data: "hi" }]);
  });

  it("reassembles events split across arbitrary chunk boundaries", async () => {
    // Stream the same payload one byte at a time; the parser must buffer
    // partial lines until a newline arrives.
    const body =
      'event: progress\ndata: {"step":42}\n\nevent: done\ndata: end\n\n';
    globalThis.fetch = mockFetchStreaming(body, {
      chunkSize: 1,
    }) as typeof fetch;

    const events: SseEvent[] = [];
    await streamSse({
      url: "/x",
      body: {},
      onEvent: (e) => events.push(e),
    });

    expect(events).toEqual([
      { event: "progress", data: '{"step":42}' },
      { event: "done", data: "end" },
    ]);
  });

  it("flushes a trailing event that lacks a final blank line", async () => {
    // Some servers close the stream without the terminating \n\n.
    // The parser should still deliver the event on reader close.
    const body = "event: complete\ndata: tail\n";
    globalThis.fetch = mockFetchStreaming(body) as typeof fetch;

    const events: SseEvent[] = [];
    await streamSse({
      url: "/x",
      body: {},
      onEvent: (e) => events.push(e),
    });

    expect(events).toEqual([{ event: "complete", data: "tail" }]);
  });

  it("invokes onHttpError and skips parsing on non-2xx responses", async () => {
    globalThis.fetch = mockFetchStreaming("event: x\ndata: y\n\n", {
      status: 503,
    }) as typeof fetch;

    const events: SseEvent[] = [];
    const onHttpError = vi.fn();
    const res = await streamSse({
      url: "/x",
      body: {},
      onEvent: (e) => events.push(e),
      onHttpError,
    });

    expect(res.status).toBe(503);
    expect(onHttpError).toHaveBeenCalledOnce();
    expect(events).toHaveLength(0);
  });

  it("POSTs JSON with the SSE Accept header", async () => {
    const fetchMock = mockFetchStreaming("");
    globalThis.fetch = fetchMock as typeof fetch;

    await streamSse({
      url: "/api/generate/stream",
      body: { prompt: "a cat" },
      onEvent: () => {},
    });

    const call = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    const [url, init] = call;
    expect(url).toBe("/api/generate/stream");
    expect(init.method).toBe("POST");
    expect(init.body).toBe('{"prompt":"a cat"}');
    const headers = init.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
    expect(headers["Accept"]).toBe("text/event-stream");
  });
});
