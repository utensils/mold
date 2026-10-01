import test from "node:test";
import assert from "node:assert/strict";
import { createRouter } from "../router-core.mjs";
function fixture() {
  const rows = new Map();
  const delivered = [];
  const closed = [];
  let now = 100;
  const store = {
    async get(id) {
      return structuredClone(rows.get(id));
    },
    async cas(id, revision, value) {
      if ((rows.get(id)?.revision ?? 0) !== revision) return false;
      rows.set(id, structuredClone({ ...value, revision: revision + 1 }));
      return true;
    },
    async put(id, value) {
      rows.set(id, structuredClone({ ...value, revision: 1 }));
    },
    async remove(id) {
      rows.delete(id);
    },
  };
  return {
    rows,
    delivered,
    closed,
    advance(n) {
      now += n;
    },
    router: createRouter({
      store,
      now: () => now,
      tokens: async () => ({ host: "host-test", frontend: "frontend-test" }),
      post: async (id, frame) => {
        delivered.push({ id, ...frame });
      },
      close: async (id) => {
        closed.push(id);
      },
    }),
  };
}
const connect = (router, id, role, token) =>
  router({
    requestContext: { routeKey: "$connect", connectionId: id },
    headers: { authorization: `Bearer ${token}`, "x-mold-relay-role": role },
  });
const send = (router, id, frame) =>
  router({
    requestContext: { routeKey: "$default", connectionId: id },
    body: JSON.stringify(frame),
  });
test("role-specific admission, singleton lease, expired host and stale frames fail closed", async () => {
  const f = fixture();
  assert.equal(
    (await connect(f.router, "bad", "host", "frontend-test")).statusCode,
    403,
  );
  assert.equal(
    (await connect(f.router, "h", "host", "host-test")).statusCode,
    200,
  );
  assert.equal(
    (await connect(f.router, "other", "host", "host-test")).statusCode,
    409,
  );
  await send(f.router, "h", { a: "hello", v: 2 });
  const sid = f.rows.get("host").sid;
  assert.equal(
    (await connect(f.router, "g", "frontend", "frontend-test")).statusCode,
    200,
  );
  await send(f.router, "g", { a: "hello", v: 2 });
  assert.equal(f.delivered.at(-1).a, "open");
  await send(f.router, "g", {
    a: "data",
    v: 2,
    sid,
    rid: "g",
    seq: 0,
    d: "YQ==",
    from: "spoof",
  });
  assert.equal(f.delivered.at(-1).id, "h");
  assert.equal(f.delivered.at(-1).from, "g");
  f.advance(91);
  const before = f.delivered.length;
  await send(f.router, "g", {
    a: "data",
    v: 2,
    sid,
    rid: "g",
    seq: 1,
    d: "Yg==",
  });
  assert.equal(f.delivered.length, before);
});
test("concurrent admission is bounded and expired guest capacity is reclaimed", async () => {
  const f = fixture();
  await connect(f.router, "h", "host", "host-test");
  const result = await Promise.all(
    Array.from({ length: 40 }, (_, i) =>
      connect(f.router, `g${i}`, "frontend", "frontend-test"),
    ),
  );
  assert.ok(result.filter((x) => x.statusCode === 200).length <= 32);
  assert.ok(Object.keys(f.rows.get("host").guests).length <= 32);
  f.advance(80);
  await send(f.router, "h", { a: "heartbeat", v: 2 });
  f.advance(11);
  assert.equal(
    (await connect(f.router, "renew", "frontend", "frontend-test")).statusCode,
    200,
  );
});
test("host cannot address a foreign guest, oversized and malformed frames are rejected", async () => {
  const f = fixture();
  await connect(f.router, "h", "host", "host-test");
  const sid = f.rows.get("host").sid;
  await send(f.router, "h", {
    a: "data",
    v: 2,
    sid,
    rid: "foreign",
    seq: 0,
    d: "YQ==",
  });
  assert.equal(f.delivered.length, 0);
  await connect(f.router, "g", "frontend", "frontend-test");
  assert.equal(
    (
      await send(f.router, "g", {
        a: "data",
        v: 2,
        sid,
        rid: "g",
        seq: 0,
        d: "a".repeat(25000),
      })
    ).statusCode,
    400,
  );
  assert.equal(
    (
      await send(f.router, "g", {
        a: "data",
        v: 2,
        sid,
        rid: "g",
        seq: -1,
        d: "YQ==",
      })
    ).statusCode,
    400,
  );
});

test("concurrent guest hello opens the host stream once", async () => {
  const f = fixture();
  await connect(f.router, "h", "host", "host-test");
  await connect(f.router, "g", "frontend", "frontend-test");
  await Promise.all([
    send(f.router, "g", { a: "hello", v: 2 }),
    send(f.router, "g", { a: "hello", v: 2 }),
  ]);
  assert.equal(f.delivered.filter((frame) => frame.a === "open").length, 1);
});

test("failure diagnostics expose only bounded error category, never details", async () => {
  const { failureCategory } = await import("../router-core.mjs");
  assert.equal(failureCategory(new Error("secret URL and token")), "Error");
  assert.equal(
    failureCategory({ name: "https://secret.invalid/?token=secret" }),
    "Error",
  );
  assert.equal(
    failureCategory({ name: "AccessDeniedException" }),
    "AccessDeniedException",
  );
});

test("late host frames after guest disconnect do not evict the enrolled host", async () => {
  const f = fixture();
  await connect(f.router, "h", "host", "host-test");
  await connect(f.router, "g", "frontend", "frontend-test");
  const sid = f.rows.get("host").sid;
  await f.router({
    requestContext: { routeKey: "$disconnect", connectionId: "g" },
  });
  const reply = await send(f.router, "h", {
    a: "ack",
    v: 2,
    sid,
    rid: "g",
    next: 1,
    credit: 4,
    ack_seq: 1,
  });
  assert.equal(reply.statusCode, 200);
  assert.deepEqual(f.closed, []);
  assert.equal(f.rows.get("host").connectionId, "h");
});
