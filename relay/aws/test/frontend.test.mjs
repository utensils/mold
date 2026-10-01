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
  assert.equal(out.status, 404);
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
