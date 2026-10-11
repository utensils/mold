import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
const source = readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), "secureMediaWorker.js"),
  "utf8",
);
class Port {
  peer!: Port;
  onmessage?: (event: { data: unknown }) => void;
  closed = false;
  postMessage(data: unknown) {
    if (!this.closed) this.peer.onmessage?.({ data });
  }
  close() {
    this.closed = true;
  }
}
async function start(signal?: AbortSignal, headers?: HeadersInit) {
  const listeners = new Map<string, (event: unknown) => void>();
  let page!: Port;
  let message: Record<string, unknown> | undefined;
  const client = {
    url: "https://studio.test/app/",
    postMessage(data: Record<string, unknown>, ports: Port[]) {
      message = data;
      page = ports[0]!;
    },
  };
  const self = {
    location: { origin: "https://studio.test" },
    registration: { scope: client.url },
    clients: { get: async () => client },
    addEventListener(name: string, callback: (event: unknown) => void) {
      listeners.set(name, callback);
    },
  };
  class Channel {
    port1 = new Port();
    port2 = new Port();
    constructor() {
      this.port1.peer = this.port2;
      this.port2.peer = this.port1;
    }
  }
  runInNewContext(source, {
    self,
    URL,
    Response,
    ReadableStream,
    Uint8Array,
    ArrayBuffer,
    MessageChannel: Channel,
    setTimeout,
    clearTimeout,
  });
  let response!: Promise<Response>;
  let lifetime!: Promise<void>;
  listeners.get("fetch")!({
    request: new Request(
      "https://studio.test/app/__mold_media/12345678-1234-1234-1234-123456789012",
      { ...(signal ? { signal } : {}), ...(headers ? { headers } : {}) },
    ),
    clientId: "page",
    waitUntil(value: Promise<void>) {
      lifetime = value;
    },
    respondWith(value: Promise<Response>) {
      response = value;
    },
  });
  await vi.advanceTimersByTimeAsync(0);
  return { response, lifetime, page, message };
}
beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());
it("times out a pending read if its owning page disappears", async () => {
  const task = await start();
  task.page.postMessage({ type: "headers", status: 200, headers: [] });
  const reader = (await task.response).body!.getReader();
  const read = reader.read();
  const rejected = expect(read).rejects.toThrow(/timed out/);
  await vi.advanceTimersByTimeAsync(20001);
  await rejected;
  await task.lifetime;
});
it("does not time out a paused consumer, and preserves Range conditions", async () => {
  const task = await start(undefined, {
    range: "bytes=10-20",
    "if-range": '"etag"',
  });
  expect(task.message).toMatchObject({
    range: "bytes=10-20",
    ifRange: '"etag"',
  });
  task.page.postMessage({ type: "headers", status: 206, headers: [] });
  const reader = (await task.response).body!.getReader();
  await vi.advanceTimersByTimeAsync(60000);
  const read = reader.read();
  await Promise.resolve();
  task.page.postMessage({ type: "data", bytes: new Uint8Array([42]).buffer });
  expect(await read).toEqual({ done: false, value: new Uint8Array([42]) });
  await reader.cancel();
  await task.lifetime;
});
it("cancels an already-aborted request before asking the page for media", async () => {
  const controller = new AbortController();
  controller.abort();
  const task = await start(controller.signal);
  expect((await task.response).status).toBe(499);
  expect(task.message).toBeUndefined();
  await task.lifetime;
});
