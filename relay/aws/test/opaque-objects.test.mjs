import test from "node:test";
import assert from "node:assert/strict";
import { createObjectBroker } from "../opaque-objects.mjs";
function fixture() {
  const rows = new Map(),
    cleared = [],
    heads = [];
  let clock = 1000,
    ready = true,
    clearFails = false;
  const store = {
    async get(k) {
      return structuredClone(rows.get(k));
    },
    async cas(k, r, v) {
      if ((rows.get(k)?.revision ?? 0) !== r) return false;
      rows.set(k, structuredClone({ ...v, revision: r + 1 }));
      return true;
    },
  };
  const storage = {
    origin: "https://test.s3.us-west-2.amazonaws.com",
    async grant(key, value) {
      return {
        putUrl: `${this.origin}/${key}?put`,
        getUrl: `${this.origin}/${key}?get`,
        putHeaders: {
          "content-type": "application/octet-stream",
          "x-amz-checksum-sha256": Buffer.from(value.sha256, "hex").toString(
            "base64",
          ),
          "if-none-match": "*",
        },
      };
    },
    async head(key) {
      heads.push(key);
      return ready ? { bytes: 100, sha256: "a".repeat(64) } : undefined;
    },
    async clear(key) {
      if (clearFails) throw new Error("Unavailable");
      cleared.push(key);
    },
  };
  const broker = createObjectBroker({ store, storage, now: () => clock });
  const ctx = {
    namespace: "b".repeat(32),
    sid: "session",
    rid: "peer==",
    sender: "host",
    direction: "host",
  };
  const create = (seq = 0, requestId = String(seq).padStart(32, "0")) => ({
    a: "object",
    v: 3,
    op: "create",
    requestId,
    sid: ctx.sid,
    rid: ctx.rid,
    seq,
    count: 2,
    bytes: 100,
    sha256: "a".repeat(64),
  });
  return {
    broker,
    ctx,
    create,
    rows,
    cleared,
    heads,
    setTime: (v) => (clock = v),
    setReady: (v) => (ready = v),
    setClearFails: (v) => (clearFails = v),
  };
}
test("allocation retries are idempotent and changes conflict; parallel allocations respect quota", async () => {
  const f = fixture();
  const a = await f.broker.command(f.ctx, f.create());
  assert.equal(a.op, "created");
  assert.deepEqual(await f.broker.command(f.ctx, f.create()), a);
  assert.equal(
    (await f.broker.command(f.ctx, { ...f.create(), bytes: 101 })).code,
    "conflict",
  );
  const results = await Promise.all(
    [1, 2, 3, 4, 5].map((n) => f.broker.command(f.ctx, f.create(n * 2))),
  );
  assert.equal(results.filter((x) => x.op === "created").length, 3);
  assert.equal(results.filter((x) => x.code === "capacity").length, 2);
});
test("object forwarding verifies stored bytes, canonical URL and session-bound fields", async () => {
  const f = fixture(),
    a = await f.broker.command(f.ctx, f.create());
  const frame = {
    a: "data",
    v: 3,
    sid: f.ctx.sid,
    rid: f.ctx.rid,
    seq: a.seq,
    o: {
      id: a.id,
      bytes: a.bytes,
      sha256: a.sha256,
      url: a.getUrl,
      count: a.count,
    },
  };
  assert.equal(
    (
      await f.broker.forward(f.ctx, {
        ...frame,
        o: { ...frame.o, url: "https://attacker.invalid/" },
      })
    ).code,
    "conflict",
  );
  f.setReady(false);
  assert.equal((await f.broker.forward(f.ctx, frame)).code, "not_ready");
  f.setReady(true);
  assert.deepEqual((await f.broker.forward(f.ctx, frame)).frame, frame);
  assert.equal(
    (await f.broker.forward({ ...f.ctx, sender: "different" }, frame)).code,
    "conflict",
  );
});
test("ACK cannot clear partially consumed bundles; whole ACK is idempotent and prevents stale allocation", async () => {
  const f = fixture(),
    a = await f.broker.command(f.ctx, f.create());
  const frame = {
    a: "data",
    v: 3,
    sid: f.ctx.sid,
    rid: f.ctx.rid,
    seq: 0,
    o: {
      id: a.id,
      bytes: a.bytes,
      sha256: a.sha256,
      url: a.getUrl,
      count: a.count,
    },
  };
  await f.broker.forward(f.ctx, frame);
  await f.broker.ack(f.ctx, { a: "ack", next: 1, ack_seq: 1 });
  assert.equal(f.cleared.length, 0);
  const release = {
    a: "object",
    v: 3,
    op: "release",
    requestId: "f".repeat(32),
    sid: f.ctx.sid,
    rid: f.ctx.rid,
    id: a.id,
  };
  assert.equal((await f.broker.command(f.ctx, release)).code, "conflict");
  const ack = { a: "ack", next: 2, ack_seq: 2 };
  await f.broker.ack(f.ctx, ack);
  await f.broker.ack(f.ctx, ack);
  assert.equal(f.cleared.length, 1);
  assert.deepEqual((await f.broker.forward(f.ctx, frame)).ack, ack);
  assert.equal((await f.broker.command(f.ctx, f.create())).code, "conflict");
});
test("expired allocation is cleaned before capacity returns; cancellation is confined to exact direction", async () => {
  const f = fixture();
  await f.broker.command(f.ctx, f.create());
  await f.broker.command(
    { ...f.ctx, direction: "device", sender: "peer" },
    f.create(),
  );
  await f.broker.cancel(f.ctx);
  assert.equal(f.cleared.length, 1);
  f.setTime(1301);
  assert.equal(
    (
      await f.broker.command(
        { ...f.ctx, direction: "device", sender: "peer" },
        f.create(2),
      )
    ).op,
    "created",
  );
  assert.equal(f.cleared.length, 2);
});

test("failed payload clearing retains quota until a successful tombstone retry", async () => {
  const f = fixture();
  for (let i = 0; i < 4; i++) await f.broker.command(f.ctx, f.create(i * 2));
  f.setClearFails(true);
  await assert.rejects(f.broker.cancel(f.ctx), /Unavailable/);
  assert.equal(
    [...f.rows.values()][0].members &&
      Object.keys([...f.rows.values()][0].members).length,
    4,
  );
  assert.equal(f.cleared.length, 0);
  f.setClearFails(false);
  await f.broker.cancel(f.ctx);
  assert.equal(Object.keys([...f.rows.values()][0].members).length, 0);
  assert.equal(f.cleared.length, 4);
});
