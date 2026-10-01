import test from "node:test";
import assert from "node:assert/strict";
import { createRouter } from "../router-core.mjs";
function fixture(options = {}) {
  const rows = new Map();
  const delivered = [];
  const closed = [];
  let now = 100;
  const store = {
    async get(id) {
      const snapshot = structuredClone(rows.get(id));
      await options.afterRead?.(id);
      return snapshot;
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
        await options.post?.(id, frame);
        delivered.push({ id, ...frame });
      },
      checkConnection: options.checkConnection,
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
  await send(f.router, "h", { a: "hello", v: 2 });
  f.delivered.length = 0;
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
  await send(f.router, "h", { a: "hello", v: 2 });
  f.delivered.length = 0;
  const sid = f.rows.get("host").sid;
  await send(f.router, "h", {
    a: "data",
    v: 2,
    sid,
    rid: "foreign",
    seq: 0,
    d: "YQ==",
  });
  assert.deepEqual(f.delivered, [
    { id: "h", a: "cancel", v: 2, sid, rid: "foreign" },
  ]);
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
  await send(f.router, "h", { a: "hello", v: 2 });
  f.delivered.length = 0;
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
  await send(f.router, "h", { a: "hello", v: 2 });
  f.delivered.length = 0;
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
test("vanished guest during delivery drops harmlessly without host error packet", async () => {
  let vanished = false;
  const f = fixture({
    post: async (id) => {
      if (vanished && id === "g") throw { $metadata: { httpStatusCode: 410 } };
    },
  });
  await connect(f.router, "h", "host", "host-test");
  await send(f.router, "h", { a: "hello", v: 2 });
  f.delivered.length = 0;
  await connect(f.router, "g", "frontend", "frontend-test");
  await send(f.router, "g", { a: "hello", v: 2 });
  const sid = f.rows.get("host").sid;
  vanished = true;
  const reply = await send(f.router, "h", {
    a: "data",
    v: 2,
    sid,
    rid: "g",
    seq: 0,
    d: "YQ==",
  });
  assert.equal(reply.statusCode, 200);
  assert.equal(f.rows.get("host").connectionId, "h");
  assert.equal(f.rows.has("connection#g"), false);
  assert.equal(f.closed.includes("h"), false);
});
test("stale host requests receive cancellation without forwarding or host eviction", async () => {
  for (const stale of [
    "missing-row",
    "expired-row",
    "expired-lease",
    "missing-lease",
  ]) {
    const f = fixture();
    await connect(f.router, "h", "host", "host-test");
    await send(f.router, "h", { a: "hello", v: 2 });
    f.delivered.length = 0;
    await connect(f.router, "g", "frontend", "frontend-test");
    await send(f.router, "g", { a: "hello", v: 2 });
    const host = f.rows.get("host"),
      sid = host.sid;
    if (stale === "missing-row") f.rows.delete("connection#g");
    if (stale === "expired-row") f.rows.get("connection#g").expiresAt = 99;
    if (stale === "expired-lease") host.guests.g = 99;
    if (stale === "missing-lease") delete host.guests.g;
    f.delivered.length = 0;
    const reply = await send(f.router, "h", {
      a: "eof",
      v: 2,
      sid,
      rid: "g",
      seq: 1,
    });
    assert.equal(reply.statusCode, 200);
    assert.deepEqual(f.delivered, [
      { id: "h", a: "cancel", v: 2, sid, rid: "g" },
    ]);
    assert.deepEqual(f.closed, []);
    assert.equal(f.rows.get("host").connectionId, "h");
  }
});
test("frontend admission waits for authenticated host hello readiness", async () => {
  const f = fixture();
  await connect(f.router, "h", "host", "host-test");
  assert.equal(
    (await connect(f.router, "early", "frontend", "frontend-test")).statusCode,
    503,
  );
  assert.equal(f.rows.get("host").connectionId, "h");
  assert.equal(f.rows.has("connection#early"), false);
  await send(f.router, "h", { a: "hello", v: 2 });
  assert.equal(
    (await connect(f.router, "g", "frontend", "frontend-test")).statusCode,
    200,
  );
});

test("lease renewed after initial read does not falsely close host", async () => {
  let delay = false;
  let f;
  f = fixture({
    afterRead: async (id) => {
      if (delay && id === "host") {
        delay = false;
        f.advance(6);
        f.rows.get("connection#h").expiresAt = 196;
      }
    },
  });
  await connect(f.router, "h", "host", "host-test");
  await send(f.router, "h", { a: "hello", v: 2 });
  f.rows.get("connection#h").expiresAt = 105;
  delay = true;
  const sid = f.rows.get("host").sid;
  const reply = await send(f.router, "h", {
    a: "eof",
    v: 2,
    sid,
    rid: "completed",
    seq: 0,
  });
  assert.equal(reply.statusCode, 200);
  assert.deepEqual(f.closed, []);
});
test("host hello records readiness before delivery and retries failed delivery", async () => {
  let reject = true;
  let f;
  f = fixture({
    post: async (_id, frame) => {
      if (frame.a === "ready") {
        assert.equal(f.rows.get("host").ready, true);
        if (reject) {
          reject = false;
          throw new Error("delivery failed");
        }
      }
    },
  });
  await connect(f.router, "h", "host", "host-test");
  await send(f.router, "h", { a: "hello", v: 2 });
  assert.equal(
    (await send(f.router, "h", { a: "hello", v: 2 })).statusCode,
    200,
  );
  assert.equal(
    (await connect(f.router, "g", "frontend", "frontend-test")).statusCode,
    200,
  );
  assert.equal(
    (await send(f.router, "g", { a: "hello", v: 2 })).statusCode,
    200,
  );
  assert.deepEqual(
    f.delivered.map(({ id, a }) => ({ id, a })),
    [
      { id: "h", a: "ready" },
      { id: "g", a: "ready" },
      { id: "h", a: "open" },
    ],
  );
});
test("legacy proven hello migrates but explicit unready host does not", async () => {
  const f = fixture();
  await connect(f.router, "h", "host", "host-test");
  f.rows.get("connection#h").hello = true;
  assert.equal(
    (await connect(f.router, "g", "frontend", "frontend-test")).statusCode,
    503,
  );
  delete f.rows.get("host").ready;
  assert.equal(
    (await connect(f.router, "g", "frontend", "frontend-test")).statusCode,
    200,
  );
});
test("replacement host resets readiness and stale hello cannot publish it", async () => {
  const f = fixture();
  await connect(f.router, "h", "host", "host-test");
  await send(f.router, "h", { a: "hello", v: 2 });
  f.advance(91);
  await connect(f.router, "new", "host", "host-test");
  assert.equal(f.rows.get("host").ready, false);
  assert.equal(
    (await send(f.router, "h", { a: "hello", v: 2 })).statusCode,
    403,
  );
  assert.equal(f.rows.get("host").ready, false);
});

for (const outcome of ["alive", "gone", "transient"]) {
  test(`host Post410 confirms ${outcome} before eviction`, async () => {
    let fail = false;
    const f = fixture({
      post: async (id) => {
        if (fail && id === "h") throw { $metadata: { httpStatusCode: 410 } };
      },
      checkConnection: async () => {
        if (outcome !== "alive")
          throw {
            $metadata: { httpStatusCode: outcome === "gone" ? 410 : 503 },
          };
      },
    });
    await connect(f.router, "h", "host", "host-test");
    await send(f.router, "h", { a: "hello", v: 2 });
    await connect(f.router, "g", "frontend", "frontend-test");
    await send(f.router, "g", { a: "hello", v: 2 });
    fail = true;
    const result = await send(f.router, "g", {
      a: "cancel",
      v: 2,
      sid: f.rows.get("host").sid,
      rid: "g",
    });
    assert.equal(result.statusCode, 503);
    assert.equal(f.rows.has("connection#h"), outcome !== "gone");
    assert.equal(f.rows.get("host").expiresAt > 100, outcome !== "gone");
    assert.deepEqual(f.closed, []);
  });
}
test("confirmed host gone cannot evict a replacement epoch", async () => {
  let fail = false;
  let f;
  f = fixture({
    post: async (id) => {
      if (fail && id === "h") throw { $metadata: { httpStatusCode: 410 } };
    },
    checkConnection: async () => {
      const current = f.rows.get("host");
      f.rows.set("host", {
        ...current,
        sid: "replacement",
        connectionId: "new",
      });
      f.rows.set("connection#new", {
        role: "host",
        sid: "replacement",
        expiresAt: 190,
      });
      throw { $metadata: { httpStatusCode: 410 } };
    },
  });
  await connect(f.router, "h", "host", "host-test");
  await send(f.router, "h", { a: "hello", v: 2 });
  await connect(f.router, "g", "frontend", "frontend-test");
  await send(f.router, "g", { a: "hello", v: 2 });
  const sid = f.rows.get("host").sid;
  fail = true;
  assert.equal(
    (await send(f.router, "g", { a: "cancel", v: 2, sid, rid: "g" }))
      .statusCode,
    503,
  );
  assert.equal(f.rows.get("host").sid, "replacement");
  assert.equal(f.rows.get("host").expiresAt, 190);
  assert.equal(f.rows.has("connection#new"), true);
});
