import test from "node:test";
import assert from "node:assert/strict";
import {
  createManagedEnrollment,
  managedConfig,
  resolveNamespace,
  ownerAllowed,
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
  assert.equal(results.filter((r) => r.status === 201).length, 32);
  f.advance(30 * 86400 + 1);
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
