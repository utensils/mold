import { afterEach, expect, it, vi } from "vitest";
import { createOriginEventSource } from "./originEventSource";
import { setOriginApiKey } from "./originAuth";
const stream = vi.hoisted(() => vi.fn(() => new Promise<void>(() => {})));
vi.mock("@microsoft/fetch-event-source", () => ({ fetchEventSource: stream }));
afterEach(() => {
  sessionStorage.clear();
  stream.mockClear();
});
it("authenticates event streams with headers and aborts on close", async () => {
  setOriginApiKey("stream-key");
  const source = createOriginEventSource("/api/resources/stream");
  const listener = vi.fn();
  source.addEventListener("snapshot", listener);
  const [url, options] = stream.mock.calls[0] as unknown as [
    string,
    {
      fetch: typeof fetch;
      signal: AbortSignal;
      onmessage: (e: { event: string; data: string }) => void;
    },
  ];
  expect(url).toBe("/api/resources/stream");
  expect(options.fetch).toBeTypeOf("function");
  options.onmessage({ event: "snapshot", data: '{"gpus":[]}' });
  expect(listener.mock.calls[0][0].data).toBe('{"gpus":[]}');
  source.close();
  expect(options.signal.aborted).toBe(true);
});
