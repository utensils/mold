import test from "node:test";
import assert from "node:assert/strict";
import { Writable } from "node:stream";
import { createFrontend, handler } from "../frontend.mjs";
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
test("Lambda first-write framing includes metadata for bodies and empty responses", async () => {
  // AWS HttpResponseStream.from installs metadata on the first write(), not
  // end(). Match that runtime boundary rather than ordinary Writable.end(data).
  // https://github.com/aws/aws-lambda-nodejs-runtime-interface-client/blob/main/src/HttpResponseStream.js
  const before = globalThis.awslambda;
  globalThis.awslambda = {
    HttpResponseStream: {
      from(raw, metadata) {
        raw.beforeWrite = () => {
          raw.chunks.push(
            Buffer.from(JSON.stringify(metadata)),
            Buffer.alloc(8),
          );
        };
        return raw;
      },
    },
  };
  try {
    const f = createFrontend({ config });
    for (const [path, method, status, empty, dispatch] of [
      ["/_mold/relay/info", "GET", 200, false],
      ["/", "GET", 303, true],
      ["/_mold/relay/info", "OPTIONS", 204, true],
      ["/api/status", "GET", 426, false],
      ["/api/status", "HEAD", 426, true],
      ["/_mold/relay/info", "GET", 503, false, handler],
    ]) {
      const raw = {
        chunks: [],
        write(chunk) {
          this.beforeWrite?.();
          this.beforeWrite = undefined;
          this.chunks.push(Buffer.from(chunk));
        },
        end(chunk) {
          if (chunk !== undefined) this.chunks.push(Buffer.from(chunk));
        },
      };
      const previousEndpoint = process.env.WS_ENDPOINT;
      try {
        if (dispatch) process.env.WS_ENDPOINT = "invalid";
        await (dispatch ?? f)(event(path, method), raw);
      } finally {
        if (previousEndpoint === undefined) delete process.env.WS_ENDPOINT;
        else process.env.WS_ENDPOINT = previousEndpoint;
      }
      const bytes = Buffer.concat(raw.chunks);
      const boundary = bytes.indexOf(Buffer.alloc(8));
      assert.ok(
        boundary > 0,
        `${method} ${path} omitted Lambda metadata prelude`,
      );
      const metadata = JSON.parse(bytes.subarray(0, boundary));
      assert.equal(metadata.statusCode, status);
      assert.equal(metadata.headers["x-mold-relay-protocol"], "3");
      assert.equal(bytes.subarray(boundary + 8).length === 0, empty);
    }
  } finally {
    globalThis.awslambda = before;
  }
});
