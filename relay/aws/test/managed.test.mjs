import test from "node:test";
import assert from "node:assert/strict";
import {
  createManagedEnrollment,
  managedConfig,
  resolveNamespace,
  ownerAllowed,
  promoteManagedHost,
  reserveManagedSlot,
  OWNER_PREFIX,
} from "../managed.mjs";
const config = {
  domain: "phones.example.com",
  origin: "https://link.example.com",
  relayURL: "wss://ws.example.com/live",
};
function fixture() {
  const rows = new Map();
  let time = 100;
  const store = {
    get: async (k) => structuredClone(rows.get(k)),
    put: async (k, v) => rows.set(k, { ...structuredClone(v), revision: 1 }),
    remove: async (k) => rows.delete(k),
    cas: async (k, r, v) => {
      if ((rows.get(k)?.revision ?? 0) !== r) return false;
      rows.set(k, { ...structuredClone(v), revision: r + 1 });
      return true;
    },
  };
  const enroll = createManagedEnrollment({ store, config, now: () => time });
  return { rows, store, enroll, advance: (n) => (time += n) };
}
const request = (method = "POST", id = "", token, ip = "192.0.2.1") => ({
  method,
  route: `/_mold/relay/enroll${id ? "/" + id : ""}`,
  headers: {
    authorization: token ? "Bearer " + token : undefined,
    "x-forwarded-for": "fake",
  },
  sourceIP: ip,
  domainName: "link.example.com",
});
test("configuration is disabled unless complete and rejects paths and unsafe origins", () => {
  assert.equal(managedConfig({}), undefined);
  assert.throws(() =>
    managedConfig({
      MANAGED_HOST_DOMAIN: "phones.example.com",
      PUBLIC_ORIGIN: "https://link.example.com/path",
      WS_ENDPOINT: config.relayURL,
    }),
  );
  assert.throws(() =>
    managedConfig({
      MANAGED_HOST_DOMAIN: "phones.example.com/path",
      PUBLIC_ORIGIN: config.origin,
      WS_ENDPOINT: config.relayURL,
    }),
  );
});
test("enroll stores only verifier; owner renew/delete; trusted hostname and source; capacity reclaim", async () => {
  const f = fixture();
  const response = await f.enroll(request());
  assert.equal(response.status, 201);
  const a = response.value;
  assert.match(a.host_id, /^[a-f0-9]{32}$/);
  assert.match(a.token, /^[A-Za-z0-9_-]{43}$/);
  assert.equal(a.public_url, `https://${a.host_id}.${config.domain}`);
  assert.equal(JSON.stringify([...f.rows]).includes(a.token), false);
  assert.equal(
    await ownerAllowed(f.store, a.host_id, "Bearer " + a.token, 100),
    true,
  );
  assert.equal(
    (await f.enroll(request("POST", a.host_id, "wrong"))).status,
    403,
  );
  assert.equal(
    (await f.enroll(request("POST", a.host_id, a.token))).status,
    200,
  );
  assert.equal(
    await resolveNamespace(
      { domainName: `${a.host_id}.${config.domain}` },
      f.store,
      config,
      100,
    ),
    a.host_id,
  );
  await assert.rejects(
    resolveNamespace(
      { domainName: "a".repeat(32) + "." + config.domain },
      f.store,
      config,
      100,
    ),
  );
  assert.equal(
    (await f.enroll({ ...request(), sourceIP: undefined })).status,
    403,
  );
  assert.equal(
    (await f.enroll({ ...request(), domainName: "evil.example.com" })).status,
    404,
  );
  assert.equal(
    (await f.enroll(request("DELETE", a.host_id, a.token))).status,
    204,
  );
  assert.equal(
    await ownerAllowed(f.store, a.host_id, "Bearer " + a.token, 100),
    false,
  );
  const results = await Promise.all(
    Array.from({ length: 40 }, (_, i) =>
      f.enroll(request("POST", "", "", `192.0.2.${i + 10}`)),
    ),
  );
  assert.equal(results.filter((r) => r.status === 201).length, 7); // initial enrollment used one burst token
  assert.equal(results.filter((r) => r.status === 429).length, 33);
  f.advance(121);
  assert.equal((await f.enroll(request())).status, 201);
});
test("source quotas ignore spoofed forwarding headers and remain bounded", async () => {
  const f = fixture();
  const results = [];
  for (let i = 0; i < 6; i++)
    results.push(
      await f.enroll({
        ...request(),
        headers: { "x-forwarded-for": String(i) },
      }),
    );
  assert.equal(results.filter((r) => r.status === 201).length, 5);
  assert.equal(results.at(-1).status, 429);
  f.advance(3601);
  assert.equal((await f.enroll(request())).status, 201);
});

test("rate roster stays bounded and expired quotas are reclaimed atomically", async () => {
  const { ROSTER } = await import("../managed.mjs");
  const f = fixture();
  const sources = Object.fromEntries(
    Array.from({ length: 1024 }, (_, i) => [
      String(i),
      { count: 1, expiresAt: 110 },
    ]),
  );
  f.rows.set(ROSTER, { revision: 1, hosts: {}, sources, expiresAt: 110 });
  assert.equal((await f.enroll(request())).status, 429);
  f.advance(11);
  assert.equal((await f.enroll(request())).status, 201);
  assert.equal(Object.keys(f.rows.get(ROSTER).sources).length, 1);
});

test("disabled configuration gives actionable enrollment failure", async () => {
  const f = fixture();
  const enrollment = createManagedEnrollment({
    store: f.store,
    config: undefined,
  });
  const result = await enrollment(request());
  assert.equal(result.status, 503);
  assert.match(result.value.error, /configure the managed gateway/);
  assert.equal(f.rows.size, 0);
});

test("unconnected reservations expire quickly and owner renew cannot prolong them", async () => {
  const f = fixture();
  const a = (await f.enroll(request())).value;
  assert.equal(a.expires_at, 220);
  f.advance(100);
  const renewed = await f.enroll(request("POST", a.host_id, a.token));
  assert.equal(renewed.status, 200);
  assert.equal(renewed.value.expires_at, 220);
  f.advance(21);
  assert.equal(
    await ownerAllowed(f.store, a.host_id, "Bearer " + a.token, 221),
    false,
  );
  assert.equal(
    (await f.enroll(request("POST", a.host_id, a.token))).status,
    403,
  );
  assert.equal(
    (await f.enroll(request("POST", "", "", "192.0.2.2"))).status,
    201,
  );
  assert.equal(Object.keys(f.rows.get("managed-roster").hosts).length, 1);
});

test("working connector promotion retains identity, heartbeat capacity expires, owner reconnect preserves identity", async () => {
  const f = fixture();
  const a = (await f.enroll(request())).value;
  assert.equal(await reserveManagedSlot(f.store, a.host_id, 100), true);
  assert.equal(await promoteManagedHost(f.store, a.host_id, 100), true);
  const owner = f.rows.get(OWNER_PREFIX + a.host_id);
  assert.equal(owner.established, true);
  assert.equal(owner.expiresAt, 100 + 30 * 86400);
  f.advance(80);
  assert.equal(await promoteManagedHost(f.store, a.host_id, 180), true);
  assert.equal(f.rows.get("managed-roster").hosts[a.host_id].expiresAt, 270);
  f.advance(91);
  assert.equal(
    await ownerAllowed(f.store, a.host_id, "Bearer " + a.token, 271),
    true,
  );
  await assert.rejects(
    resolveNamespace(
      { domainName: a.host_id + "." + config.domain },
      f.store,
      config,
      271,
    ),
  );
  const renewed = await f.enroll(request("POST", a.host_id, a.token));
  assert.equal(renewed.status, 200);
  assert.equal(renewed.value.expires_at, owner.expiresAt);
  assert.equal(f.rows.get("managed-roster").hosts[a.host_id], undefined);
  assert.equal(await reserveManagedSlot(f.store, a.host_id, 271), true);
  assert.equal(await promoteManagedHost(f.store, a.host_id, 271), true);
  assert.equal(renewed.value.host_id, a.host_id);
  assert.equal(renewed.value.token, a.token);
});

test("capacity leases stay atomic at32 and old owners reacquire without enrollment budget", async () => {
  const f = fixture();
  const a = (await f.enroll(request())).value;
  await promoteManagedHost(f.store, a.host_id, 100);
  const roster = f.rows.get("managed-roster");
  const hosts = Object.fromEntries(
    Array.from({ length: 32 }, (_, i) => [
      i.toString(16).padStart(32, "0"),
      { expiresAt: 190 },
    ]),
  );
  f.rows.set("managed-roster", {
    ...roster,
    hosts,
    budget: { tokens: 0, updatedAt: 100 },
  });
  assert.equal(await reserveManagedSlot(f.store, a.host_id, 100), false);
  const renew = await f.enroll(request("POST", a.host_id, a.token));
  assert.equal(renew.status, 200);
  f.advance(91);
  assert.equal(await reserveManagedSlot(f.store, a.host_id, 191), true);
  assert.equal(Object.keys(f.rows.get("managed-roster").hosts).length, 1);
});

test("global cadence bounds distributed anonymous creation after initial burst", async () => {
  const f = fixture();
  const enrollMany = () =>
    Promise.all(
      Array.from({ length: 20 }, (_, i) =>
        f.enroll(request("POST", "", "", `198.51.100.${i + 1}`)),
      ),
    );
  assert.equal((await enrollMany()).filter((r) => r.status === 201).length, 8);
  f.advance(29);
  assert.equal(
    (await f.enroll(request("POST", "", "", "203.0.113.1"))).status,
    429,
  );
  f.advance(1);
  assert.equal(
    (await f.enroll(request("POST", "", "", "203.0.113.1"))).status,
    201,
  );
});

test("concurrent reconnect owners cannot overbook32 live slots", async () => {
  const f = fixture(),
    expiresAt = 100 + 30 * 86400;
  const owners = Array.from({ length: 40 }, (_, i) =>
    i.toString(16).padStart(32, "0"),
  );
  for (const id of owners)
    f.rows.set(OWNER_PREFIX + id, {
      verifier: "a".repeat(64),
      established: true,
      revision: 1,
      expiresAt,
    });
  const results = await Promise.all(
    owners.map((id) => reserveManagedSlot(f.store, id, 100)),
  );
  assert.equal(results.filter(Boolean).length, 32);
  assert.equal(Object.keys(f.rows.get("managed-roster").hosts).length, 32);
});

test("old inline identities migrate while long idle slots are reclaimed", async () => {
  const { createHash } = await import("node:crypto");
  const f = fixture(),
    id = "a".repeat(32),
    token = "b".repeat(43),
    expiresAt = 100 + 30 * 86400;
  f.rows.set("managed-roster", {
    revision: 1,
    hosts: {
      [id]: {
        verifier: createHash("sha256").update(token).digest("hex"),
        expiresAt,
      },
    },
    sources: {},
    expiresAt,
  });
  assert.equal((await f.enroll(request("POST", id, token))).status, 200);
  assert.equal(f.rows.get(OWNER_PREFIX + id).established, true);
  assert.equal(f.rows.get(OWNER_PREFIX + id).expiresAt, expiresAt);
  assert.equal(f.rows.get("managed-roster").hosts[id], undefined);
  assert.equal(await reserveManagedSlot(f.store, id, 100), true);
});
