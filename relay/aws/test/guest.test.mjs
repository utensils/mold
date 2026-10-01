import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import net from "node:net";
import { openRequest, cleanHeaders } from "../guest.mjs";
test("HTTP facade parses original response and preserves upload without caller FIN", async () => {
  let observed;
  const server = http.createServer(async (req, res) => {
    const chunks = [];
    for await (const b of req) chunks.push(b);
    observed = { headers: req.headers, body: Buffer.concat(chunks) };
    res.writeHead(206, {
      "content-type": "application/octet-stream",
      "content-range": "bytes 0-2/9",
    });
    res.end(Buffer.from([0, 255, 7]));
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  try {
    const socket = net.connect(server.address().port, "127.0.0.1");
    const close = () => socket.destroy();
    const result = await openRequest(
      {
        method: "POST",
        path: "/api/example?q=1",
        headers: {
          "x-api-key": "fixture",
          forwarded: "spoof",
          "x-forwarded-for": "spoof",
          "content-type": "application/octet-stream",
        },
        body: Buffer.from([1, 0, 3]),
      },
      { guestFactory: async () => ({ socket, sid: "s", close }) },
    );
    const chunks = [];
    for await (const b of result.response) chunks.push(b);
    assert.deepEqual(Buffer.concat(chunks), Buffer.from([0, 255, 7]));
    assert.equal(result.response.statusCode, 206);
    assert.equal(result.response.headers["content-range"], "bytes 0-2/9");
    assert.equal(observed.headers["x-api-key"], "fixture");
    assert.equal(observed.headers.forwarded, undefined);
    assert.deepEqual(observed.body, Buffer.from([1, 0, 3]));
    close();
  } finally {
    await new Promise((r) => server.close(r));
  }
});
test("hop-by-hop headers and metrics are fenced before forwarding", async () => {
  assert.deepEqual(
    cleanHeaders({
      "content-type": "image/png",
      connection: "close",
      "x-forwarded-for": "spoof",
    }),
    { "content-type": "image/png" },
  );
  await assert.rejects(openRequest({ path: "/metrics" }), /Forbidden/);
  await assert.rejects(
    openRequest({ path: "//attacker.invalid/" }),
    /Forbidden/,
  );
});
