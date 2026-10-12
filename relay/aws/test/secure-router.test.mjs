import test from "node:test";
import assert from "node:assert/strict";
import { sign } from "node:crypto";
import { identity } from "./secure-admission.test.mjs";
import {
  challengeBytes,
  grantBytes,
  inviteBytes,
} from "../secure-admission.mjs";
import { createSecureRouter, opaqueFrame } from "../secure-router.mjs";
function fixture(options = {}) {
  const rows = new Map(),
    delivered = [],
    closed = [];
  let clock = 1000;
  const store = {
    async get(id) {
      const value = structuredClone(rows.get(id));
      await options.get?.(id, value);
      return value;
    },
    async cas(id, rev, value) {
      if ((await options.cas?.(id, rev, value)) === false) return false;
      if ((rows.get(id)?.revision ?? 0) !== rev) return false;
      rows.set(id, structuredClone({ ...value, revision: rev + 1 }));
      return true;
    },
    async put(id, value) {
      await options.put?.(id, value);
      rows.set(id, structuredClone({ ...value, revision: 1 }));
    },
    async remove(id) {
      await options.remove?.(id);
      rows.delete(id);
    },
  };
  const router = createSecureRouter({
    store,
    storage: options.storage,
    now: () => clock,
    post: async (id, frame) => {
      await options.post?.(id, frame);
      delivered.push({ to: id, ...frame });
    },
    close: async (id) => {
      await options.close?.(id);
      closed.push(id);
    },
    checkConnection: options.checkConnection,
  });
  const send = (id, frame) =>
    router({
      requestContext: { routeKey: "$default", connectionId: id },
      body: JSON.stringify(frame),
    });
  const connect = (id, source = "1.2.3.4", v = "3") =>
    router({
      requestContext: {
        routeKey: "$connect",
        connectionId: id,
        identity: { sourceIp: source },
      },
      queryStringParameters: { v },
    });
  async function join(id, own, host = own, role = "host", certificate) {
    assert.equal((await connect(id)).statusCode, 200);
    assert.equal(
      (
        await send(id, {
          a: "hello",
          v: 3,
          namespace: host.namespace,
          hostKey: host.binding.signingKey,
          role,
          ...(options.storage
            ? { objects: options.objectCapabilities?.[role] ?? 1 }
            : {}),
        })
      ).statusCode,
      200,
    );
    const challenge = delivered.at(-1);
    const signing = role === "invite" ? certificate.signer : own;
    const signature = sign(
      null,
      challengeBytes({
        ...challenge,
        namespace: host.namespace,
        role,
        signingKey: own.binding.signingKey,
      }),
      signing.privateKey,
    ).toString("hex");
    const fields =
      role === "device"
        ? { grant: certificate }
        : role === "invite"
          ? { invite: certificate.cert }
          : {};
    return send(id, {
      a: "join",
      v: 3,
      binding: own.binding,
      signature,
      ...fields,
    });
  }
  const register = async (host, enabled = true) =>
    store.put(`v3-registration#${host.namespace}`, {
      namespace: host.namespace,
      signingKey: host.binding.signingKey,
      principalId: "fixture-owner",
      enabled,
    });
  const grant = (host, device, overrides = {}) => {
    const g = {
      id: "aa".repeat(16),
      expiresAt: clock + 100,
      epoch: 0,
      ...overrides,
    };
    g.signature = sign(
      null,
      grantBytes({ ...g, namespace: host.namespace, ...device.binding }),
      host.privateKey,
    ).toString("hex");
    return g;
  };
  return {
    rows,
    delivered,
    closed,
    router,
    send,
    connect,
    join,
    register,
    grant,
    advance: (n) => (clock += n),
  };
}
test("hard cutover refuses v2 admission and routes only pinned, signed v3 peers", async () => {
  const f = fixture(),
    h = identity(),
    d = identity();
  assert.equal((await f.connect("old", "1.2.3.4", "2")).statusCode, 426);
  await f.register(h);
  assert.equal((await f.join("host", h)).statusCode, 200);
  assert.equal(
    (await f.join("device", d, h, "device", f.grant(h, d))).statusCode,
    200,
  );
  const ready = f.delivered.find((x) => x.to === "device" && x.a === "ready");
  const frame = {
    a: "data",
    v: 3,
    sid: ready.sid,
    rid: "device",
    seq: 0,
    d: "Y2lwaGVydGV4dA==",
    authorization: "must-not-forward",
    from: "spoof",
  };
  assert.equal((await f.send("device", frame)).statusCode, 200);
  assert.deepEqual(f.delivered.at(-1), {
    to: "host",
    a: "data",
    v: 3,
    sid: ready.sid,
    rid: "device",
    seq: 0,
    d: frame.d,
  });
  assert.equal((await f.send("device", { ...frame, v: 2 })).statusCode, 400);
  assert.ok(f.closed.includes("device"));
});
test("only active host/device sessions can broker exact ciphertext objects and ACK cleanup preserves retry", async () => {
  const removed = [];
  const storage = {
    origin: "https://test.s3.us-west-2.amazonaws.com",
    async grant(key, m) {
      return {
        putUrl: `${this.origin}/${key}?put`,
        getUrl: `${this.origin}/${key}?get`,
        putHeaders: {},
      };
    },
    async head() {
      return { bytes: 100, sha256: "a".repeat(64) };
    },
    async clear(key) {
      removed.push(key);
    },
  };
  const f = fixture({ storage }),
    h = identity(),
    d = identity();
  await f.register(h);
  await f.join("host", h);
  await f.join("device", d, h, "device", f.grant(h, d));
  const ready = f.delivered.find((x) => x.to === "device" && x.a === "ready");
  assert.equal(ready.objects.origin, storage.origin);
  await f.send("host", { a: "accept", v: 3, sid: ready.sid, rid: "device" });
  const create = {
    a: "object",
    v: 3,
    op: "create",
    requestId: "1".repeat(32),
    sid: ready.sid,
    rid: "device",
    seq: 0,
    count: 2,
    bytes: 100,
    sha256: "a".repeat(64),
  };
  assert.equal((await f.send("device", create)).statusCode, 200);
  const grant = f.delivered.at(-1);
  assert.equal(grant.op, "created");
  const frame = {
    a: "data",
    v: 3,
    sid: ready.sid,
    rid: "device",
    seq: 0,
    o: {
      id: grant.id,
      bytes: 100,
      count: 2,
      sha256: create.sha256,
      url: grant.getUrl,
    },
  };
  assert.equal((await f.send("device", frame)).statusCode, 200);
  assert.equal(f.delivered.at(-1).to, "host");
  const ack = {
    a: "ack",
    v: 3,
    sid: ready.sid,
    rid: "device",
    next: 2,
    credit: 16,
    ack_seq: 7,
  };
  assert.equal((await f.send("host", ack)).statusCode, 200);
  assert.equal(removed.length, 1);
  assert.equal((await f.send("device", frame)).statusCode, 200);
  assert.deepEqual(f.delivered.at(-1), { to: "device", ...ack });
  assert.equal(
    (await f.send("device", { ...create, rid: "other" })).statusCode,
    403,
  );
});
test("cloud certificates cannot allocate paid storage until the host accepts them", async () => {
  const storage = {
    origin: "https://test.s3.us-west-2.amazonaws.com",
    async clear() {},
  };
  const f = fixture({ storage }),
    h = identity(),
    d = identity();
  await f.register(h);
  await f.join("host", h);
  await f.join("device", d, h, "device", f.grant(h, d));
  const ready = f.delivered.find((x) => x.to === "device" && x.a === "ready");
  assert.equal(
    (
      await f.send("device", {
        a: "object",
        v: 3,
        op: "create",
        requestId: "1".repeat(32),
        sid: ready.sid,
        rid: "device",
        seq: 0,
        count: 1,
        bytes: 100,
        sha256: "a".repeat(64),
      })
    ).statusCode,
    403,
  );
  assert.ok(![...f.rows.keys()].some((k) => k.startsWith("v3-objects#")));
});
test("wrong grant keys, wrong namespace and replayed admission fail closed", async () => {
  const f = fixture(),
    h = identity(),
    d = identity(),
    other = identity();
  await f.register(h);
  await f.register(h);
  await f.join("host", h);
  assert.equal(
    (await f.join("bad", d, h, "device", f.grant(h, other))).statusCode,
    403,
  );
  await f.connect("replay");
  await f.send("replay", {
    a: "hello",
    v: 3,
    namespace: h.namespace,
    hostKey: h.binding.signingKey,
    role: "host",
  });
  const c = f.delivered.at(-1);
  const proof = {
    a: "join",
    v: 3,
    binding: h.binding,
    signature: sign(
      null,
      challengeBytes({
        ...c,
        namespace: h.namespace,
        role: "host",
        signingKey: h.binding.signingKey,
      }),
      h.privateKey,
    ).toString("hex"),
  };
  await f.connect("different");
  await f.send("different", {
    a: "hello",
    v: 3,
    namespace: h.namespace,
    hostKey: h.binding.signingKey,
    role: "host",
  });
  assert.equal((await f.send("different", proof)).statusCode, 403);
});
test("per-source pending admission is bounded before any cloud relay ownership exists", async () => {
  const f = fixture();
  const results = await Promise.all(
    Array.from({ length: 8 }, (_, i) => f.connect("c" + i)),
  );
  assert.equal(results.filter((r) => r.statusCode === 200).length, 4);
  f.advance(61);
  assert.equal((await f.send("c0", { a: "hello", v: 3 })).statusCode, 403);
  assert.ok(f.closed.includes("c0"));
});
test("host epoch revocation cancels existing peers and refuses old grants", async () => {
  const f = fixture(),
    h = identity(),
    d = identity();
  await f.register(h);
  await f.register(h);
  await f.join("host", h);
  const g = f.grant(h, d);
  await f.join("device", d, h, "device", g);
  assert.equal(
    (await f.send("host", { a: "policy", v: 3, epoch: 1 })).statusCode,
    200,
  );
  assert.ok(f.closed.includes("device"));
  assert.equal((await f.join("revoked", d, h, "device", g)).statusCode, 403);
  assert.equal(
    (await f.join("fresh", d, h, "device", f.grant(h, d, { epoch: 1 })))
      .statusCode,
    200,
  );
});
test("invitation certificate is bounded, signed, single use and exposes no application credential", async () => {
  const f = fixture(),
    h = identity(),
    d = identity(),
    i = identity();
  await f.register(h);
  await f.register(h);
  await f.join("host", h);
  const cert = {
    id: "cc".repeat(16),
    signingKey: i.binding.signingKey,
    expiresAt: 1100,
    epoch: 0,
  };
  cert.signature = sign(
    null,
    inviteBytes({ ...cert, namespace: h.namespace }),
    h.privateKey,
  ).toString("hex");
  assert.equal(
    (await f.join("pair", d, h, "invite", { cert, signer: i })).statusCode,
    200,
  );
  assert.equal(
    (await f.join("pair-again", identity(), h, "invite", { cert, signer: i }))
      .statusCode,
    403,
  );
  const open = f.delivered.find((x) => x.a === "open" && x.rid === "pair");
  assert.equal(open.invite.id, cert.id);
  assert.deepEqual(open.binding, d.binding);
});
test("transient Post410 verifies live peer and retries exact ciphertext without eviction", async () => {
  let fail = false;
  const checks = [];
  const f = fixture({
    post: async (id, frame) => {
      if (fail && id === "device" && frame.a === "data") {
        fail = false;
        throw { $metadata: { httpStatusCode: 410 } };
      }
    },
    checkConnection: async (id) => checks.push(id),
  });
  const h = identity(),
    d = identity();
  await f.register(h);
  await f.join("host", h);
  await f.join("device", d, h, "device", f.grant(h, d));
  const sid = f.delivered.find((x) => x.to === "device" && x.a === "ready").sid;
  fail = true;
  assert.equal(
    (
      await f.send("host", {
        a: "data",
        v: 3,
        sid,
        rid: "device",
        seq: 0,
        d: "YQ==",
      })
    ).statusCode,
    200,
  );
  assert.deepEqual(checks, ["device"]);
  assert.ok(f.rows.has("v3-connection#device"));
});

test("unregistered or disabled owners cannot consume active slots and disable fences live sessions", async () => {
  const f = fixture(),
    h = identity(),
    d = identity();
  assert.equal((await f.join("unregistered", h)).statusCode, 403);
  assert.equal(f.rows.has("v3-hosts"), false);
  assert.equal(f.rows.has(`v3-host#${h.namespace}`), false);
  await f.register(h, false);
  assert.equal((await f.join("disabled", h)).statusCode, 403);
  assert.equal(f.rows.has("v3-hosts"), false);
  await f.register(h);
  assert.equal((await f.join("host", h)).statusCode, 200);
  assert.equal(
    (await f.join("device", d, h, "device", f.grant(h, d))).statusCode,
    200,
  );
  await f.register(h, false);
  assert.equal(
    (await f.send("host", { a: "heartbeat", v: 3 })).statusCode,
    403,
  );
  assert.ok(f.closed.includes("device"));
  assert.equal(Object.keys(f.rows.get("v3-hosts").members).length, 1);
});

test("delayed old host disconnect cannot release the replacement host capacity slot", async () => {
  let unblock, captured;
  const waiting = new Promise((resolve) => (captured = resolve));
  const blocked = new Promise((resolve) => (unblock = resolve));
  let pause = false;
  const f = fixture({
    get: async (id) => {
      if (pause && id === "v3-connection#old-host") {
        pause = false;
        captured();
        await blocked;
      }
    },
  });
  const h = identity();
  await f.register(h);
  await f.join("old-host", h);
  pause = true;
  const disconnect = f.router({
    requestContext: { routeKey: "$disconnect", connectionId: "old-host" },
  });
  await waiting;
  assert.equal((await f.join("new-host", h)).statusCode, 200);
  unblock();
  await disconnect;
  assert.ok(f.rows.get("v3-hosts").members[h.namespace]);
  assert.equal(f.rows.get(`v3-host#${h.namespace}`).connectionId, "new-host");
  assert.equal(
    (await f.send("new-host", { a: "heartbeat", v: 3 })).statusCode,
    200,
  );
});

test("hot-path fresh authority reads start together and validate the complete snapshot", async () => {
  const h = identity(),
    d = identity();
  const keys = new Set([
    `v3-host#${h.namespace}`,
    `v3-registration#${h.namespace}`,
    "v3-connection#device",
  ]);
  let armed = false,
    release;
  const allStarted = new Promise((resolve) => {
    release = resolve;
  });
  const seen = new Set();
  const f = fixture({
    get: async (key) => {
      if (!armed || !keys.has(key)) return;
      seen.add(key);
      if (seen.size === keys.size) release();
      await allStarted;
    },
  });
  await f.register(h);
  await f.join("host", h);
  await f.join("device", d, h, "device", f.grant(h, d));
  const ready = f.delivered.find((x) => x.to === "device" && x.a === "ready");
  armed = true;
  let timer;
  try {
    const result = await Promise.race([
      f.send("host", {
        a: "data",
        v: 3,
        sid: ready.sid,
        rid: "device",
        seq: 0,
        d: "Y2lwaGVydGV4dA==",
      }),
      new Promise((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("authority reads were serialized")),
          200,
        );
      }),
    ]);
    assert.equal(result.statusCode, 200);
    assert.equal(seen.size, 3);
  } finally {
    clearTimeout(timer);
    release();
  }
});

test("ciphertext ACK window permits four objects and remains bounded at sixty-four frames", () => {
  const ack = {
    a: "ack",
    v: 3,
    sid: "host",
    rid: "peer",
    next: 0,
    credit: 64,
    ack_seq: 0,
  };
  assert.deepEqual(opaqueFrame(ack), ack);
  assert.equal(opaqueFrame({ ...ack, credit: 65 }), undefined);
});

test("concurrent JOIN completion order cannot strand the winning host outside quota", async () => {
  const h = identity();
  let resume,
    entered,
    armed = true;
  const blocked = new Promise((resolve) => {
    resume = resolve;
  });
  const waiting = new Promise((resolve) => {
    entered = resolve;
  });
  const f = fixture({
    cas: async (key, _rev, value) => {
      if (
        armed &&
        key === `v3-host#${h.namespace}` &&
        value.connectionId === "second"
      ) {
        armed = false;
        entered();
        await blocked;
      }
    },
  });
  await f.register(h);
  await f.join("first", h);
  const second = f.join("second", h);
  await waiting;
  assert.equal((await f.join("third", h)).statusCode, 200);
  resume();
  assert.equal((await second).statusCode, 200);
  assert.equal(f.rows.get(`v3-host#${h.namespace}`).connectionId, "second");
  assert.equal(
    (await f.send("second", { a: "heartbeat", v: 3 })).statusCode,
    200,
  );
  assert.equal(
    (await f.send("third", { a: "heartbeat", v: 3 })).statusCode,
    403,
  );
  assert.equal(
    (await f.send("second", { a: "heartbeat", v: 3 })).statusCode,
    200,
  );
});

test("disconnected namespaces retain bounded quota only until their short lease expires", async () => {
  const f = fixture(),
    hosts = Array.from({ length: 33 }, () => identity());
  for (let i = 0; i < 32; i++) {
    await f.register(hosts[i]);
    assert.equal((await f.join(`host${i}`, hosts[i])).statusCode, 200);
  }
  await f.router({
    requestContext: { routeKey: "$disconnect", connectionId: "host0" },
  });
  await f.register(hosts[32]);
  assert.equal((await f.join("waiting", hosts[32])).statusCode, 503);
  assert.equal(Object.keys(f.rows.get("v3-hosts").members).length, 32);
  f.advance(91);
  assert.equal((await f.join("admitted", hosts[32])).statusCode, 200);
  assert.equal(Object.keys(f.rows.get("v3-hosts").members).length, 1);
});

test("object capability requires both peers and cannot be enabled by device alone", async () => {
  const storage = {
    origin: "https://test.s3.us-west-2.amazonaws.com",
    async clear() {},
  };
  const f = fixture({ storage, objectCapabilities: { host: 0, device: 1 } }),
    h = identity(),
    d = identity();
  await f.register(h);
  await f.join("host", h);
  await f.join("device", d, h, "device", f.grant(h, d));
  assert.ok(
    f.delivered
      .filter((x) => x.a === "ready" || x.a === "open")
      .every((x) => x.objects === undefined),
  );
  const ready = f.delivered.find((x) => x.to === "device" && x.a === "ready");
  assert.equal(
    (
      await f.send("device", {
        a: "object",
        v: 3,
        op: "create",
        requestId: "1".repeat(32),
        sid: ready.sid,
        rid: "device",
        seq: 0,
        count: 1,
        bytes: 100,
        sha256: "a".repeat(64),
      })
    ).statusCode,
    403,
  );
});

test("host cancellation closes cloud membership instead of leaving storage authority alive", async () => {
  const f = fixture(),
    h = identity(),
    d = identity();
  await f.register(h);
  await f.join("host", h);
  await f.join("device", d, h, "device", f.grant(h, d));
  const ready = f.delivered.find((x) => x.to === "device" && x.a === "ready");
  await f.send("host", { a: "accept", v: 3, sid: ready.sid, rid: "device" });
  assert.equal(
    (await f.send("host", { a: "cancel", v: 3, sid: ready.sid, rid: "device" }))
      .statusCode,
    200,
  );
  assert.equal(f.rows.has("v3-connection#device"), false);
  assert.ok(f.closed.includes("device"));
  assert.equal(f.rows.get(`v3-host#${h.namespace}`).guests.device, undefined);
});

function pendingLedgers(f) {
  return [...f.rows.entries()].filter(([key]) => key.startsWith("v3-pending"));
}
async function fillPending(f) {
  for (let i = 0; i < 4; i++)
    assert.equal((await f.connect(`stale${i}`)).statusCode, 200);
}
test("expired pending sockets are closed before four ordinary admissions recover", async () => {
  const f = fixture();
  await fillPending(f);
  f.advance(61);
  const results = await Promise.all(
    Array.from({ length: 4 }, (_, i) => f.connect(`fresh${i}`)),
  );
  assert.equal(results.filter((r) => r.statusCode === 200).length, 4);
  for (let i = 0; i < 4; i++) assert.ok(f.closed.includes(`stale${i}`));
  for (const [, row] of pendingLedgers(f))
    assert.equal(Object.keys(row.members).length, 4);
  assert.equal((await f.connect("over-cap")).statusCode, 429);
});
test("missing pending rows are reclaimed only after the creation grace", async () => {
  const f = fixture();
  await fillPending(f);
  f.rows.delete("v3-connection#stale0");
  assert.equal((await f.connect("creating")).statusCode, 429);
  assert.deepEqual(f.closed, []);
  f.advance(61);
  assert.equal((await f.connect("recovered")).statusCode, 200);
  assert.ok(f.closed.includes("stale0"));
  for (const [, row] of pendingLedgers(f)) assert.ok(!row.members.stale0);
});
test("completed join with leftover pending leases is reclaimed without closing active peer", async () => {
  const f = fixture();
  await fillPending(f);
  f.rows.set("v3-connection#stale0", {
    state: "active",
    sid: "live",
    revision: 2,
    expiresAt: 1090,
  });
  assert.equal((await f.connect("fresh")).statusCode, 200);
  assert.deepEqual(f.closed, []);
  assert.equal(f.rows.get("v3-connection#stale0").sid, "live");
  for (const [, row] of pendingLedgers(f)) assert.ok(!row.members.stale0);
});
test("failed socket cleanup retains both quota leases and recovers on next admission", async () => {
  let failing = true;
  const f = fixture({
    close: async () => {
      if (failing) throw Error("unavailable");
    },
  });
  await fillPending(f);
  f.advance(61);
  assert.equal((await f.connect("blocked")).statusCode, 503);
  for (const [, row] of pendingLedgers(f))
    assert.equal(Object.keys(row.members).length, 4);
  failing = false;
  assert.equal((await f.connect("recovered")).statusCode, 200);
});
test("reclamation CAS loses to a concurrent completed join and never closes that peer", async () => {
  let armed = false,
    f;
  f = fixture({
    cas: async (key, rev, next) => {
      if (armed && key === "v3-connection#stale0" && next.state === "closing") {
        armed = false;
        f.rows.set(key, {
          state: "active",
          sid: "won",
          revision: rev + 1,
          expiresAt: 1200,
        });
      }
    },
  });
  await fillPending(f);
  f.advance(61);
  armed = true;
  assert.equal((await f.connect("fresh")).statusCode, 200);
  assert.ok(!f.closed.includes("stale0"));
  assert.equal(f.rows.get("v3-connection#stale0").sid, "won");
});
test("in-flight joining retains capacity until its bounded completion grace", async () => {
  const f = fixture();
  await fillPending(f);
  for (let i = 0; i < 4; i++)
    f.rows.get(`v3-connection#stale${i}`).state = "joining";
  f.advance(61);
  assert.equal((await f.connect("waiting")).statusCode, 429);
  assert.deepEqual(f.closed, []);
  f.advance(60);
  assert.equal((await f.connect("recovered")).statusCode, 200);
});

test("concurrent reservation before row creation cannot be reclaimed as an orphan", async () => {
  let releasePut, enteredPut;
  const gate = new Promise((resolve) => {
    releasePut = resolve;
  });
  const entered = new Promise((resolve) => {
    enteredPut = resolve;
  });
  const f = fixture({
    put: async (key) => {
      if (key === "v3-connection#creating") {
        enteredPut();
        await gate;
      }
    },
  });
  const creating = f.connect("creating");
  await entered;
  for (let i = 0; i < 3; i++)
    assert.equal((await f.connect(`other${i}`)).statusCode, 200);
  assert.equal((await f.connect("over-cap")).statusCode, 429);
  assert.deepEqual(f.closed, []);
  releasePut();
  assert.equal((await creating).statusCode, 200);
  assert.equal(f.rows.get("v3-connection#creating").state, "pending");
});
test("global release failure retains source ledger for a later cleanup retry", async () => {
  let failing = false;
  const f = fixture({
    cas: async (key, _rev, row) => {
      if (failing && key === "v3-pending" && !row.members.stale0)
        throw Error("unavailable");
    },
  });
  await fillPending(f);
  f.advance(61);
  failing = true;
  assert.equal((await f.connect("blocked")).statusCode, 503);
  for (const [, row] of pendingLedgers(f)) assert.ok(row.members.stale0);
  failing = false;
  assert.equal((await f.connect("recovered")).statusCode, 200);
  for (const [, row] of pendingLedgers(f)) assert.ok(!row.members.stale0);
});
test("active installation cannot overwrite a cleanup fence after joining", async () => {
  let f;
  f = fixture({
    cas: async (key, revision, next) => {
      if (key === "v3-connection#host" && next.state === "active") {
        f.rows.set(key, {
          state: "closing",
          revision: revision + 1,
          expiresAt: 1600,
        });
      }
    },
  });
  const h = identity();
  await f.register(h);
  assert.equal((await f.join("host", h)).statusCode, 409);
  assert.notEqual(f.rows.get("v3-connection#host")?.state, "active");
  assert.ok(f.closed.includes("host"));
});

test("exhausted global cleanup CAS retains source lease instead of silently dropping it", async () => {
  let failing = false,
    losses = 0;
  const f = fixture({
    cas: async (key, _rev, row) => {
      if (failing && key === "v3-pending" && !row.members.stale0) {
        losses++;
        return false;
      }
    },
  });
  await fillPending(f);
  f.advance(61);
  failing = true;
  assert.equal((await f.connect("blocked")).statusCode, 503);
  assert.equal(losses, 32);
  for (const [, row] of pendingLedgers(f)) assert.ok(row.members.stale0);
  failing = false;
  assert.equal((await f.connect("recovered")).statusCode, 200);
});
