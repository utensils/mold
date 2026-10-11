import { webcrypto } from "node:crypto";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  CipherObjectCarrier,
  objectPolicy,
  packObject,
  unpackObject,
  validateObjectUrl,
  type ObjectContext,
} from "./secureObjects";
const origin =
  "https://mold-relay-042506291754-us-west-2.s3.us-west-2.amazonaws.com";
const context: ObjectContext = {
  namespace: "a".repeat(32),
  sid: "session+1",
  rid: "device=2",
  origin,
};
const policy = {
  version: 1,
  origin,
  maxBytes: 262144,
  maxCount: 16,
  maxInFlightBytes: 1048576,
  maxInFlightObjects: 4,
};
const id = "b".repeat(32);
const url = (direction = "host") =>
  `${origin}/ciphertext/v3/${context.namespace}/session%2B1/device%3D2/${direction}/${id}?X-Amz-Signature=fixture`;
beforeEach(() => vi.stubGlobal("crypto", webcrypto));
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
it("pins object storage independently of relay advertisements and every URL session segment", () => {
  expect(objectPolicy(policy, origin)).toEqual(policy);
  expect(
    objectPolicy({ ...policy, origin: "https://attacker.test" }, origin),
  ).toBeNull();
  expect(objectPolicy({ ...policy, maxBytes: 999999 }, origin)).toBeNull();
  expect(validateObjectUrl(url(), context, "host", id)).toBe(url());
  for (const invalid of [
    url("device"),
    url().replace("session%2B1", "other"),
    url().replace(origin, "https://attacker.test"),
    url() + "#fragment",
    url().replace("https://", "https://user:pass@"),
  ]) {
    expect(() => validateObjectUrl(invalid, context, "host", id)).toThrow();
  }
});
it("rejects a conflicting ciphertext retry while its batch is still being staged", () => {
  const carrier = new CipherObjectCarrier(
    context,
    () => {},
    () => {},
    () => 0,
    () => {},
  );
  const frame = {
    a: "data",
    v: 3,
    sid: context.sid,
    rid: context.rid,
    seq: 0,
    d: "AQ==",
  };
  carrier.send(frame);
  expect(() => carrier.send({ ...frame, d: "Ag==" })).toThrow(/Conflicting/);
  carrier.close();
});
it("verifies whole ciphertext objects before yielding any original chunk", async () => {
  const chunks = [new Uint8Array([1, 2]), new Uint8Array([3, 4, 5])];
  const packed = await packObject(chunks);
  expect(
    await unpackObject(packed.bytes, {
      bytes: 13,
      count: 2,
      sha256: packed.sha256,
    }),
  ).toEqual(chunks);
  const corrupt = packed.bytes.slice();
  corrupt[corrupt.length - 1] = corrupt[corrupt.length - 1]! ^ 1;
  await expect(
    unpackObject(corrupt, { bytes: 13, count: 2, sha256: packed.sha256 }),
  ).rejects.toThrow();
  await expect(
    unpackObject(packed.bytes, { bytes: 13, count: 1, sha256: packed.sha256 }),
  ).rejects.toThrow();
  await expect(
    packObject(Array.from({ length: 16 }, () => new Uint8Array(16384))),
  ).rejects.toThrow();
});
it("fetches and verifies once, coalesces ACKs, and acknowledges already-consumed refs without fetching deleted objects", async () => {
  vi.useFakeTimers();
  const packed = await packObject([new Uint8Array([1]), new Uint8Array([2])]);
  const fetcher = vi.fn(
    async (_url: string, _init?: RequestInit) =>
      new Response(packed.bytes.slice().buffer),
  );
  vi.stubGlobal("fetch", fetcher);
  let next = 0;
  const sent: Record<string, unknown>[] = [];
  const received: unknown[] = [];
  const carrier = new CipherObjectCarrier(
    context,
    (frame) => sent.push(frame),
    (frame) => {
      received.push(frame);
      next++;
      carrier.send({
        a: "ack",
        v: 3,
        sid: context.sid,
        rid: context.rid,
        next,
        credit: 16,
        ack_seq: next,
      });
    },
    () => next,
    () => {},
  );
  const frame = {
    a: "data",
    v: 3,
    sid: context.sid,
    rid: context.rid,
    seq: 0,
    o: {
      id,
      bytes: packed.bytes.length,
      count: 2,
      sha256: packed.sha256,
      url: url(),
    },
  };
  await carrier.receive(frame);
  expect(received).toHaveLength(2);
  await vi.advanceTimersByTimeAsync(32);
  expect(sent).toHaveLength(1);
  expect(sent[0]).toMatchObject({ a: "ack", next: 2 });
  await carrier.receive(frame);
  expect(fetcher).toHaveBeenCalledTimes(1);
  expect(sent[1]).toEqual(sent[0]);
  expect(fetcher.mock.calls[0]?.[1]).toMatchObject({
    credentials: "omit",
    redirect: "error",
  });
  carrier.close();
});
it("rejects corrupt, oversized, cross-session and out-of-window objects before delivering bytes", async () => {
  const packed = await packObject([new Uint8Array([1])]);
  const fetcher = vi.fn(async () => new Response(new Uint8Array(99)));
  vi.stubGlobal("fetch", fetcher);
  const delivered = vi.fn();
  const carrier = new CipherObjectCarrier(
    context,
    () => {},
    delivered,
    () => 0,
    () => {},
  );
  const frame = {
    a: "data",
    v: 3,
    sid: context.sid,
    rid: context.rid,
    seq: 0,
    o: { id, bytes: 5, count: 1, sha256: packed.sha256, url: url() },
  };
  await expect(carrier.receive({ ...frame, sid: "other" })).rejects.toThrow();
  await expect(carrier.receive({ ...frame, seq: 64 })).rejects.toThrow();
  expect(fetcher).not.toHaveBeenCalled();
  await expect(carrier.receive(frame)).rejects.toThrow();
  expect(delivered).not.toHaveBeenCalled();
  carrier.close();
});
it("replays the ACK when a late GET sees a tombstone only after every referenced frame was consumed", async () => {
  const packed = await packObject([new Uint8Array([1])]);
  let finish!: (response: Response) => void;
  vi.stubGlobal(
    "fetch",
    vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          finish = resolve;
        }),
    ),
  );
  let next = 0;
  const emitted = vi.fn();
  const delivered = vi.fn();
  const carrier = new CipherObjectCarrier(
    context,
    emitted,
    delivered,
    () => next,
    () => {},
  );
  const frame = {
    a: "data",
    v: 3,
    sid: context.sid,
    rid: context.rid,
    seq: 0,
    o: {
      id,
      bytes: packed.bytes.length,
      count: 1,
      sha256: packed.sha256,
      url: url(),
    },
  };
  const pending = carrier.receive(frame);
  next = 1;
  const ack = {
    a: "ack",
    v: 3,
    sid: context.sid,
    rid: context.rid,
    next,
    credit: 16,
    ack_seq: 7,
  };
  carrier.send(ack);
  finish(
    new Response(new Uint8Array(0), { headers: { "content-length": "0" } }),
  );
  await expect(pending).resolves.toBeUndefined();
  expect(delivered).not.toHaveBeenCalled();
  expect(emitted).toHaveBeenLastCalledWith(ack);
  carrier.close();
});

it("uploads only retained ciphertext, binds broker replies, and retransmits the same object reference", async () => {
  vi.useFakeTimers();
  const sent: Record<string, unknown>[] = [];
  const fetcher = vi.fn(
    async (_url: string, _init?: RequestInit) =>
      new Response(null, { status: 200 }),
  );
  vi.stubGlobal("fetch", fetcher);
  const carrier = new CipherObjectCarrier(
    context,
    (frame) => sent.push(frame),
    () => {},
    () => 0,
    () => {},
  );
  const frames = Array.from({ length: 9 }, (_, seq) => ({
    a: "data",
    v: 3,
    sid: context.sid,
    rid: context.rid,
    seq,
    d: btoa("x".repeat(16320)),
  }));
  frames.forEach((frame) => carrier.send(frame));
  await vi.waitFor(() =>
    expect(sent.some((frame) => frame.op === "create")).toBe(true),
  );
  const request = sent.find((frame) => frame.op === "create")!;
  const checksum = btoa(
    String.fromCharCode(
      ...Uint8Array.from((request.sha256 as string).match(/../g)!, (value) =>
        parseInt(value, 16),
      ),
    ),
  );
  await carrier.receive({
    ...request,
    op: "created",
    id,
    putUrl: url("device"),
    getUrl: url("device"),
    putHeaders: {
      "content-type": "application/octet-stream",
      "x-amz-checksum-sha256": checksum,
      "if-none-match": "*",
    },
    expiresAt: Math.floor(Date.now() / 1000) + 300,
  });
  await vi.waitFor(() => expect(sent.some((frame) => frame.o)).toBe(true));
  const ref = sent.find((frame) => frame.o)!;
  carrier.send(frames[0]!);
  await vi.advanceTimersByTimeAsync(32);
  expect(sent.filter((frame) => frame.o).at(-1)).toEqual(ref);
  expect(fetcher).toHaveBeenCalledTimes(1);
  expect(fetcher.mock.calls[0]?.[1]).toMatchObject({
    method: "PUT",
    credentials: "omit",
    redirect: "error",
  });
  carrier.close();
});
