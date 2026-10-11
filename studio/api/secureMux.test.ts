import { describe, expect, it } from "vitest";
import { SecureMux, type SecureRecord } from "./secureMux";

const bytes = (value: unknown) =>
  new TextEncoder().encode(JSON.stringify(value));
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
function harness() {
  const sent: SecureRecord[] = [];
  const mux = new SecureMux(async (record) => {
    sent.push(record);
  });
  return {
    mux,
    sent,
    receive: (kind: number, payload = new Uint8Array(), streamId = 1) =>
      mux.receive({ kind, streamId, payload }),
  };
}

describe("encrypted HTTP stream multiplexing", () => {
  it("fails a truncated or oversized declared response instead of reporting successful EOF", async () => {
    for (const length of [2, 4]) {
      const h = harness();
      const pending = h.mux.fetch(new Request("https://host.example/api/file"));
      h.receive(
        2,
        bytes({ status: 200, headers: [["content-length", String(length)]] }),
      );
      const response = await pending;
      if (length === 2)
        expect(() => h.receive(3, new Uint8Array([1, 2, 3]))).toThrow(/length/);
      else {
        h.receive(3, new Uint8Array([1, 2, 3]));
        expect(() => h.receive(4)).toThrow(/length/);
      }
      await expect(response.arrayBuffer()).rejects.toThrow(/length/);
    }
  });
  it("preserves status, headers and Range while granting credit only after consumption", async () => {
    const h = harness();
    const pending = h.mux.fetch(
      new Request("https://host.example/api/file", {
        headers: { Range: "bytes=9-11" },
      }),
    );
    await tick();
    expect(
      JSON.parse(new TextDecoder().decode(h.sent[0]!.payload)),
    ).toMatchObject({
      method: "GET",
      path: "/api/file",
      headers: [["range", "bytes=9-11"]],
      length: 0,
    });
    h.receive(
      2,
      bytes({ status: 206, headers: [["content-range", "bytes 9-11/100"]] }),
    );
    h.receive(3, new Uint8Array([9, 10, 11]));
    const response = await pending;
    expect(response.status).toBe(206);
    expect(response.headers.get("content-range")).toBe("bytes 9-11/100");
    expect(h.sent.filter((r) => r.kind === 6)).toHaveLength(0);
    const reader = response.body!.getReader();
    expect((await reader.read()).value).toEqual(new Uint8Array([9, 10, 11]));
    await tick();
    expect(h.sent.filter((r) => r.kind === 6)).toHaveLength(0);
    h.receive(4);
    expect((await reader.read()).done).toBe(true);
    expect(h.mux.activeStreams).toBe(0);
  });

  it("batches consumed byte credit without granting credit for buffered bytes", async () => {
    const h = harness();
    const pending = h.mux.fetch(new Request("https://host.example/api/file"));
    h.receive(2, bytes({ status: 200, headers: [] }));
    const reader = (await pending).body!.getReader();
    for (let i = 0; i < 32; i++) h.receive(3, new Uint8Array(16384));
    expect(h.sent.filter((r) => r.kind === 6)).toHaveLength(0);
    for (let i = 0; i < 31; i++) await reader.read();
    expect(h.sent.filter((r) => r.kind === 6)).toHaveLength(0);
    await reader.read();
    const credit = h.sent.filter((r) => r.kind === 6);
    expect(credit).toHaveLength(1);
    expect(new DataView(credit[0]!.payload.buffer).getUint32(0)).toBe(524288);
    await reader.cancel();
  });

  it("bounds an unread response to 1 MiB and fails the session on credit violation", async () => {
    const h = harness();
    const pending = h.mux.fetch(new Request("https://host.example/api/file"));
    h.receive(2, bytes({ status: 200, headers: [] }));
    const response = await pending;
    for (let i = 0; i < 64; i++) h.receive(3, new Uint8Array(16384));
    expect(() => h.receive(3, new Uint8Array([1]))).toThrow(/credit/);
    await expect(response.arrayBuffer()).rejects.toThrow(/credit/);
    expect(h.mux.activeStreams).toBe(0);
  });

  it("coalesces paused one-byte records into bounded chunks without delaying live events", async () => {
    const h = harness();
    const pending = h.mux.fetch(new Request("https://host.example/api/events"));
    h.receive(2, bytes({ status: 200, headers: [] }));
    const reader = (await pending).body!.getReader();
    const live = reader.read();
    await tick();
    h.receive(3, new Uint8Array([7]));
    expect((await live).value).toEqual(new Uint8Array([7]));
    for (let i = 0; i < 65536; i++) h.receive(3, new Uint8Array([i % 256]));
    h.receive(4);
    let total = 0,
      chunks = 0;
    for (;;) {
      const next = await reader.read();
      if (next.done) break;
      chunks++;
      for (const value of next.value!) expect(value).toBe(total++ % 256);
    }
    expect(total).toBe(65536);
    expect(chunks).toBeLessThanOrEqual(4);
  });

  it("sends uploads in bounded chunks and pauses at the window without replay", async () => {
    const h = harness();
    const pending = h.mux.fetch(
      new Request("https://host.example/api/upload", {
        method: "POST",
        body: new Uint8Array(1200000),
      }),
    );
    const failure = pending.catch((error: unknown) => error);
    await tick();
    expect(
      h.sent
        .filter((r) => r.kind === 3)
        .reduce((n, r) => n + r.payload.length, 0),
    ).toBe(1048576);
    const credit = new Uint8Array(4);
    new DataView(credit.buffer).setUint32(0, 16384);
    h.receive(6, credit);
    await tick();
    expect(
      h.sent
        .filter((r) => r.kind === 3)
        .reduce((n, r) => n + r.payload.length, 0),
    ).toBe(1064960);
    h.mux.close(new Error("disconnected; request may have completed"));
    expect(await failure).toMatchObject({
      message: "disconnected; request may have completed",
    });
    expect(h.sent.filter((r) => r.kind === 1)).toHaveLength(1);
  });

  it("cancels only the selected stream, never reuses IDs, and ignores late cancelled data", async () => {
    const h = harness();
    const controller = new AbortController();
    const pending = h.mux.fetch(
      new Request("https://host.example/api/events", {
        signal: controller.signal,
      }),
    );
    h.receive(2, bytes({ status: 200, headers: [] }));
    const response = await pending;
    controller.abort();
    await expect(response.text()).rejects.toThrow();
    expect(h.sent.some((r) => r.kind === 5 && r.streamId === 1)).toBe(true);
    h.receive(3, new Uint8Array([1]));
    const next = h.mux.fetch(new Request("https://host.example/api/status"));
    h.receive(2, bytes({ status: 204, headers: [] }), 3);
    h.receive(4, new Uint8Array(), 3);
    expect((await next).status).toBe(204);
    expect(h.sent.filter((r) => r.kind === 1).map((r) => r.streamId)).toEqual([
      1, 3,
    ]);
  });

  it("refuses peer-created streams and body data before headers", async () => {
    const h = harness();
    const pending = h.mux.fetch(new Request("https://host.example/api/status"));
    const failure = pending.catch((error: unknown) => error);
    expect(() => h.receive(3, new Uint8Array([1]))).toThrow(/headers/);
    expect(await failure).toBeInstanceOf(Error);
    const other = harness();
    expect(() =>
      other.receive(2, bytes({ status: 200, headers: [] }), 99),
    ).toThrow(/stream/);
  });
});
