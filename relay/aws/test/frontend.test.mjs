import test from "node:test";
import assert from "node:assert/strict";
import { Writable } from "node:stream";
import { createFrontend } from "../frontend.mjs";
function writer() {
  const chunks = [];
  const out = new Writable({
    write(chunk, _encoding, done) {
      chunks.push(Buffer.from(chunk));
      done();
    },
  });
  out.setMetadata = (status, headers) => {
    out.status = status;
    out.headers = headers;
  };
  out.value = () => Buffer.concat(chunks).toString();
  return out;
}
const config = {
  websocket_url: "wss://gateway.execute-api.us-west-2.amazonaws.com/production",
  host_domain: "mold-link.urandom.io",
  studio_url: "https://utensils.io/mold/studio/",
};
const event = (path, method = "GET") => ({
  path,
  httpMethod: method,
  headers: {
    authorization: "Bearer SECRET",
    "x-mold-request-target": "/api/status",
  },
  body: "PRIVATE INPUT",
});
test("old HTTP content/API/media/upload endpoints fail closed without invoking a host", async () => {
  const f = createFrontend({ config });
  for (const path of [
    "/api/status",
    "/api/events",
    "/api/generate",
    "/_mold/relay/uploads",
    "/_mold/objects/anything",
    "/_mold/relay/media",
    "/metrics",
  ]) {
    const out = writer();
    await f(event(path, "POST"), out);
    assert.equal(out.status, 426);
    assert.equal(JSON.parse(out.value()).error, "secure-client-required");
    assert.doesNotMatch(out.value(), /SECRET|PRIVATE/);
    assert.equal(out.headers["x-mold-relay-protocol"], "3");
  }
});
test("discovery and enrollment return only pinned static public metadata", async () => {
  const f = createFrontend({ config });
  for (const [path, method] of [
    ["/_mold/relay/info", "GET"],
    ["/_mold/relay/enroll", "POST"],
  ]) {
    const out = writer();
    await f(event(path, method), out);
    assert.equal(out.status, 200);
    assert.deepEqual(JSON.parse(out.value()), { protocol: 3, ...config });
  }
});
test("cloud serves no executable shell and redirects only to configured independent Studio", async () => {
  const f = createFrontend({ config });
  const out = writer();
  await f(
    { ...event("/"), queryStringParameters: { url: "https://evil.invalid" } },
    out,
  );
  assert.equal(out.status, 303);
  assert.equal(out.headers.location, config.studio_url);
  assert.equal(out.value(), "");
});
test("HEAD remains empty, OPTIONS advertises only metadata and unsafe config fails closed", async () => {
  const f = createFrontend({ config });
  let out = writer();
  await f(event("/api/status", "HEAD"), out);
  assert.equal(out.status, 426);
  assert.equal(out.value(), "");
  out = writer();
  await f(event("/_mold/relay/info", "OPTIONS"), out);
  assert.equal(out.status, 204);
  assert.equal(out.value(), "");
  assert.equal(out.headers["access-control-allow-headers"], "content-type");
  for (const unsafe of [
    { websocket_url: "ws://localhost" },
    { studio_url: "https://evil.invalid/" },
    { host_domain: "user:secret@host" },
  ])
    assert.throws(() => createFrontend({ config: { ...config, ...unsafe } }));
});
