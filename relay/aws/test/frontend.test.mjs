import test from "node:test";
import assert from "node:assert/strict";
import { Writable, Readable } from "node:stream";
import { createFrontend, normalizeEvent } from "../frontend.mjs";
function writer() {
  const chunks = [];
  const output = new Writable({
    write(chunk, encoding, done) {
      chunks.push(Buffer.from(chunk));
      done();
    },
  });
  output.setMetadata = (status, headers) => {
    output.status = status;
    output.headers = headers;
  };
  output.value = () => Buffer.concat(chunks).toString();
  return output;
}
function event(path, method = "GET", body = "", headers = {}) {
  return { path, httpMethod: method, body, headers };
}
function response(body, headers = {}, statusCode = 200) {
  const stream = Readable.from([Buffer.from(body)]);
  Object.assign(stream, { headers, statusCode });
  return stream;
}
test("REST original percent-encoded target survives; spoofed preserved Bearer is discarded", () => {
  const req = normalizeEvent({
    ...event("/api/gallery/image/a b.png", "GET", "", {
      "x-mold-request-target": "/api/gallery/image/a%20b.png?x=1&x=2",
      authorization: "Bearer actual",
      "x-mold-viewer-authorization": "Bearer spoof",
    }),
    multiValueQueryStringParameters: { x: ["1", "2"] },
  });
  assert.equal(req.path, "/api/gallery/image/a%20b.png?x=1&x=2");
  assert.equal(req.headers.authorization, "Bearer actual");
  assert.equal(req.headers["x-mold-viewer-authorization"], undefined);
});
test("offline browser shell and info load without host access; metrics aliases do not forward", async () => {
  let calls = 0;
  const frontend = createFrontend({
    request: async () => {
      calls++;
      return {
        response: response("secret", { "content-length": "6" }),
        sid: "s",
        close() {},
      };
    },
    shell: async () => ({
      Body: Readable.from(["<html>fixture</html>"]),
      ContentType: "text/html",
    }),
  });
  let out = writer();
  await frontend(event("/create"), out);
  assert.equal(out.status, 200);
  assert.match(out.value(), /fixture/);
  out = writer();
  await frontend(event("/_mold/relay/info"), out);
  assert.equal(JSON.parse(out.value()).protocol, 2);
  assert.equal(calls, 0);
  out = writer();
  await frontend(
    event("/api/status", "GET", "", { "x-mold-request-target": "/metrics" }),
    out,
  );
  assert.equal(out.status, 503); // Routed-path aliases are rejected before forwarding.
  assert.equal(calls, 0);
});
test("unknown finite response above 200MB stages once before viewer headers; mutation is not replayed", async () => {
  let calls = 0,
    stageBytes = 0;
  const frontend = createFrontend({
    request: async () => {
      calls++;
      const body = Readable.from(
        (async function* () {
          for (let i = 0; i < 201; i++) yield Buffer.alloc(1024 * 1024);
        })(),
      );
      Object.assign(body, {
        headers: {
          "content-type": "application/octet-stream",
          "x-mold-seed-used": "42",
        },
        statusCode: 201,
      });
      return { response: body, sid: "s", close() {} };
    },
    stage: async (stream) => {
      for await (const b of stream) stageBytes += b.length;
      return { url: "https://fixture.invalid/object" };
    },
  });
  const out = writer();
  await frontend(
    event("/api/generate", "POST", "{}", { "x-api-key": "fixture" }),
    out,
  );
  assert.equal(calls, 1);
  assert.equal(stageBytes, 201 * 1024 * 1024);
  assert.equal(out.headers["x-mold-relay-object"], "1");
  assert.equal(JSON.parse(out.value()).status, 201);
  assert.equal(JSON.parse(out.value()).headers["x-mold-seed-used"], "42");
});
test("unauthorized callers obtain no upload grant or async media work", async () => {
  let grants = 0;
  const frontend = createFrontend({
    request: async () => ({
      response: response("{}", { "content-length": "2" }, 401),
      sid: "s",
      close() {},
    }),
    prepareUpload: async () => {
      grants++;
    },
    invoke: async () => {
      grants++;
    },
  });
  const out = writer();
  await frontend(
    event("/_mold/relay/uploads", "POST", "{}", { "x-api-key": "wrong" }),
    out,
  );
  assert.equal(grants, 0);
  assert.equal(out.status, 503);
});
test("streaming Lambda metadata is emitted even for JSON and empty HEAD responses", async () => {
  const before = globalThis.awslambda;
  let wrote = false;
  globalThis.awslambda = {
    HttpResponseStream: {
      from(raw, metadata) {
        raw.setMetadata(metadata.statusCode, metadata.headers);
        return {
          write(value) {
            wrote = true;
            return raw.write(value);
          },
          end(value) {
            raw.end(value);
          },
        };
      },
    },
  };
  try {
    const frontend = createFrontend();
    const out = writer();
    await frontend(event("/_mold/relay/info"), out);
    assert.equal(wrote, true);
    assert.equal(out.status, 200);
  } finally {
    globalThis.awslambda = before;
  }
});
test("missing keys never allocate a host stream and preflight204 has no body", async () => {
  let calls = 0;
  const frontend = createFrontend({
    request: async () => {
      calls++;
      throw new Error("unexpected");
    },
  });
  let out = writer();
  await frontend(event("/api/events"), out);
  assert.equal(out.status, 401);
  assert.equal(calls, 0);
  out = writer();
  await frontend(event("/api/status", "OPTIONS"), out);
  assert.equal(out.status, 204);
  assert.equal(out.value(), "");
  assert.equal(out.headers["access-control-max-age"], "600");
});
test("client disconnect during SSE backpressure releases its host session", async () => {
  let closed = false;
  const upstream = new Readable({
    read() {
      this.push(Buffer.alloc(4096));
    },
  });
  Object.assign(upstream, {
    headers: { "content-type": "text/event-stream" },
    statusCode: 200,
  });
  const out = new Writable({
    highWaterMark: 1,
    write(chunk, encoding, done) {},
  });
  out.setMetadata = () => {};
  const frontend = createFrontend({
    request: async () => ({
      response: upstream,
      sid: "s",
      close() {
        closed = true;
        upstream.destroy();
      },
    }),
  });
  const running = frontend(
    event("/api/events", "GET", "", { "x-api-key": "fixture" }),
    out,
  );
  setTimeout(() => out.destroy(), 20);
  await Promise.race([
    running,
    new Promise((resolve, reject) =>
      setTimeout(() => reject(new Error("host session leaked")), 200),
    ),
  ]);
  assert.equal(closed, true);
});
test("media jobs persist object identity instead of signed credential URLs", async () => {
  const rows = new Map();
  let invoke;
  const headers = { "x-api-key": "fixture" };
  const frontend = createFrontend({
    store: {
      put: async (id, value) => rows.set(id, { ...value, revision: 1 }),
      get: async (id) => rows.get(id),
      cas: async (id, revision, value) => {
        rows.set(id, { ...value, revision: 2 });
        return true;
      },
    },
    request: async () => ({
      response: response("{}", { "content-length": "2" }),
      sid: "epoch",
      close() {},
    }),
    invoke: async (event) => {
      invoke = event;
    },
    stage: async () => ({
      key: "_mold/objects/fixture",
      url: "https://secret.invalid/signed",
      expires_at: Math.floor(Date.now() / 1000) + 900,
    }),
    objectURL: async (key) => {
      assert.equal(key, "_mold/objects/fixture");
      return "https://fresh.invalid/signed";
    },
  });
  let out = writer();
  await frontend(
    event(
      "/_mold/relay/media",
      "POST",
      JSON.stringify({ path: "/api/gallery/image/a.png" }),
      headers,
    ),
    out,
  );
  const id = JSON.parse(out.value()).relay.id;
  await frontend(invoke, writer());
  assert.equal(rows.get("media#" + id).url, undefined);
  out = writer();
  await frontend(event("/_mold/relay/media/" + id, "GET", "", headers), out);
  assert.equal(JSON.parse(out.value()).url, "https://fresh.invalid/signed");
});

test("relay media enrichment preserves the original host ticket expiry", async () => {
  const frontend = createFrontend({
    request: async (req) => ({
      response:
        req.path === "/api/gallery/media-token"
          ? response(
              JSON.stringify({
                auth_required: true,
                token: "host-ticket",
                expires_at: 1234,
              }),
              { "content-length": "73" },
            )
          : response("{}", { "content-length": "2" }),
      sid: "epoch",
      close() {},
    }),
    store: { put: async () => {} },
    invoke: async () => {},
  });
  const out = writer();
  await frontend(
    event(
      "/api/gallery/media-token",
      "POST",
      JSON.stringify({ path: "/api/gallery/image/a.png" }),
      { "x-api-key": "fixture" },
    ),
    out,
  );
  const ticket = JSON.parse(out.value());
  assert.equal(ticket.expires_at, 1234);
  assert.equal(ticket.relay.state, "pending");
});
test("original request target cannot change routed path or bypass key admission", async () => {
  let calls = 0;
  const frontend = createFrontend({
    request: async () => {
      calls++;
      throw Error("unexpected");
    },
  });
  const out = writer();
  await frontend(
    event("/api/pairing/claim", "POST", "", {
      "x-mold-request-target": "/api/queue",
    }),
    out,
  );
  assert.equal(calls, 0);
  assert.notEqual(out.status, 200);
});
test("expired pending stage jobs fail promptly without another tunnel", async () => {
  const id = "11111111-1111-4111-8111-111111111111";
  const { credentialDigest } = await import("../transfers.mjs");
  const headers = { "x-api-key": "fixture" };
  const frontend = createFrontend({
    store: {
      get: async () => ({
        state: "pending",
        createdAt: Math.floor(Date.now() / 1000) - 91,
        expiresAt: Math.floor(Date.now() / 1000) + 800,
        credential: credentialDigest(headers),
      }),
    },
  });
  const out = writer();
  await frontend(event(`/_mold/relay/media/${id}`, "GET", "", headers), out);
  assert.equal(JSON.parse(out.value()).state, "failed");
});
test("stager refuses SSE and closes its upstream", async () => {
  let closed = false,
    staged = false;
  const frontend = createFrontend({
    store: {
      get: async () => ({
        state: "pending",
        sid: "s",
        expiresAt: Math.floor(Date.now() / 1000) + 900,
        revision: 1,
      }),
      cas: async () => true,
      put: async (_id, value) => {
        assert.equal(value.state, "failed");
      },
    },
    request: async () => ({
      sid: "s",
      response: response("event", { "content-type": "text/event-stream" }),
      close() {
        closed = true;
      },
    }),
    stage: async () => {
      staged = true;
    },
  });
  await frontend(
    {
      kind: "stage",
      id: "fixture",
      sid: "s",
      request: { path: "/api/events" },
    },
    writer(),
  );
  assert.equal(staged, false);
  assert.equal(closed, true);
});
test("shell HEAD destroys unused S3 response body", async () => {
  const body = Readable.from(["shell"]);
  const frontend = createFrontend({ shell: async () => ({ Body: body }) });
  await frontend(event("/create", "HEAD"), writer());
  assert.equal(body.destroyed, true);
});
test("host CORS cannot override relay wildcard; outcome header exposed", async () => {
  const frontend = createFrontend({
    request: async () => ({
      sid: "s",
      response: response("ok", {
        "content-length": "2",
        "access-control-allow-origin": "http://private",
      }),
      close() {},
    }),
  });
  const out = writer();
  await frontend(
    event("/api/status", "GET", "", { "x-api-key": "fixture" }),
    out,
  );
  assert.equal(out.headers["access-control-allow-origin"], "*");
  assert.match(
    out.headers["access-control-expose-headers"],
    /x-mold-relay-request-state/,
  );
});
test("mutation outcomes distinguish connect refusal from forwarding attempts", async () => {
  for (const forwarded of [false, true]) {
    const frontend = createFrontend({
      request: async (req) => {
        if (forwarded) req.onForwardAttempt();
        throw Error("offline");
      },
    });
    const out = writer();
    await frontend(
      event("/api/queue", "POST", "{}", { "x-api-key": "fixture" }),
      out,
    );
    assert.equal(
      out.headers["x-mold-relay-request-state"],
      forwarded ? "outcome-unknown" : "not-forwarded",
    );
  }
});
test("health and public API documents forward instead of shell or admission401", async () => {
  const paths = [];
  const frontend = createFrontend({
    request: async (req) => {
      paths.push(req.path);
      return {
        sid: "s",
        response: response("up", { "content-length": "2" }),
        close() {},
      };
    },
  });
  for (const path of ["/health", "/api/docs", "/api/openapi.json"]) {
    const out = writer();
    await frontend(event(path), out);
    assert.equal(out.status, 200);
    assert.equal(out.value(), "up");
  }
  assert.deepEqual(paths, ["/health", "/api/docs", "/api/openapi.json"]);
});
test("encoded HEAD target preserves exact bytes on matching routed path", async () => {
  let forwarded;
  const frontend = createFrontend({
    request: async (req) => {
      forwarded = req;
      return {
        sid: "s",
        response: response("", { "content-length": "0" }),
        close() {},
      };
    },
  });
  const out = writer();
  await frontend(
    event("/api/gallery/image/a b.png", "HEAD", "", {
      "x-api-key": "fixture",
      "x-mold-request-target":
        "/api/gallery/image/a%20b.png?media_token=fixture%2Bticket",
    }),
    out,
  );
  assert.equal(out.status, 200);
  assert.equal(
    forwarded.path,
    "/api/gallery/image/a%20b.png?media_token=fixture%2Bticket",
  );
  assert.equal(forwarded.method, "HEAD");
});
test("encoded API route aliases still require credentials", async () => {
  let calls = 0;
  const f = createFrontend({
    request: async () => {
      calls++;
      throw Error();
    },
  });
  const out = writer();
  await f(
    event("/%61pi/status", "GET", "", {
      "x-mold-request-target": "/api/status",
    }),
    out,
  );
  assert.equal(out.status, 401);
  assert.equal(calls, 0);
});
test("host private cache policy survives relay CORS enforcement", async () => {
  const f = createFrontend({
    request: async () => ({
      sid: "s",
      response: response("ok", {
        "content-length": "2",
        "cache-control": "private,max-age=300",
      }),
      close() {},
    }),
  });
  const out = writer();
  await f(event("/api/status", "GET", "", { "x-api-key": "fixture" }), out);
  assert.equal(out.headers["cache-control"], "private,max-age=300");
});
test("working media jobs fail after transfer deadline, not overall grant expiry", async () => {
  const { credentialDigest } = await import("../transfers.mjs");
  const headers = { "x-api-key": "fixture" };
  const frontend = createFrontend({
    store: {
      get: async () => ({
        state: "working",
        workingAt: Math.floor(Date.now() / 1000) - 841,
        expiresAt: Math.floor(Date.now() / 1000) + 50,
        credential: credentialDigest(headers),
      }),
    },
  });
  const out = writer();
  await frontend(
    event(
      "/_mold/relay/media/11111111-1111-4111-8111-111111111111",
      "GET",
      "",
      headers,
    ),
    out,
  );
  assert.equal(JSON.parse(out.value()).state, "failed");
});
test("credential-free connection proof forwards only bounded exact POST contract", async () => {
  const calls = [];
  const frontend = createFrontend({
    request: async (request) => {
      calls.push(request);
      return {
        response: response('{"id":"fixture","proof":"bounded"}', {
          "content-length": "34",
        }),
        sid: "s",
        close() {},
      };
    },
  });
  const body = JSON.stringify({
    kind: "api",
    key_tag: "0123456789abcdef",
    nonce: "a".repeat(64),
  });
  const out = writer();
  await frontend(
    event("/api/connection-probe", "POST", body, {
      "content-type": "application/json",
    }),
    out,
  );
  assert.equal(out.status, 200);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].headers["x-api-key"], undefined);
  assert.equal(calls[0].headers.authorization, undefined);
  assert.equal(calls[0].body.toString(), body);
});
test("anonymous connection proof rejects malformed oversized and credential-bearing requests before host access", async () => {
  let calls = 0;
  const frontend = createFrontend({
    request: async () => {
      calls++;
      throw Error("unexpected");
    },
  });
  const valid = JSON.stringify({
    kind: "pairing",
    key_tag: "0123456789abcdef",
    nonce: "a".repeat(64),
  });
  for (const [method, body, headers, path] of [
    ["GET", "", {}, "/api/connection-probe"],
    ["POST", "{}", {}, "/api/connection-probe"],
    ["POST", "x".repeat(513), {}, "/api/connection-probe"],
    [
      "POST",
      JSON.stringify({
        kind: "api",
        key_tag: ["0123456789abcdef"],
        nonce: "a".repeat(64),
      }),
      {},
      "/api/connection-probe",
    ],
    ["POST", valid, { "x-api-key": "fixture" }, "/api/connection-probe"],
    [
      "POST",
      valid,
      { authorization: "Bearer fixture" },
      "/api/connection-probe",
    ],
    ["POST", valid, {}, "/api/connection-addresses"],
  ]) {
    const out = writer();
    await frontend(event(path, method, body, headers), out);
    assert.equal(out.status, 401);
  }
  assert.equal(calls, 0);
});
