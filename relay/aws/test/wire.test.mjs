import test from "node:test";
import assert from "node:assert/strict";
import { RelayDuplex } from "../wire.mjs";
const tick = () => new Promise((r) => setImmediate(r));
function fixture() {
  const sent = [];
  const socket = new RelayDuplex({
    sid: "s",
    rid: "r",
    send: (f) => sent.push(f),
    gapMs: 50,
    retryMs: 20,
  });
  return { socket, sent };
}
test("ordered delivery, matching duplicates, EOF and conflicting duplicate refusal", async () => {
  const { socket, sent } = fixture();
  const chunks = [];
  socket.on("data", (b) => chunks.push(b));
  socket.on("error", () => {});
  socket.receive({ a: "data", v: 2, sid: "s", rid: "r", seq: 1, d: "Yg==" });
  socket.receive({ a: "data", v: 2, sid: "s", rid: "r", seq: 0, d: "YQ==" });
  socket.receive({ a: "data", v: 2, sid: "s", rid: "r", seq: 0, d: "YQ==" });
  await tick();
  assert.equal(Buffer.concat(chunks).toString(), "ab");
  assert.equal(sent.at(-1).next, 2);
  socket.receive({ a: "data", v: 2, sid: "s", rid: "r", seq: 0, d: "Yw==" });
  assert.equal(socket.destroyed, true);
});
test("bounded sender waits for credit and sends EOF in sequence", async () => {
  const { socket, sent } = fixture();
  socket.on("error", () => {});
  socket.write(Buffer.alloc(6 * 16384));
  await tick();
  assert.equal(sent.filter((f) => f.a === "data").length, 4);
  socket.receive({ a: "ack", v: 2, sid: "s", rid: "r", next: 4, credit: 4 });
  await tick();
  assert.equal(sent.filter((f) => f.a === "data").length, 6);
  socket.receive({ a: "ack", v: 2, sid: "s", rid: "r", next: 6, credit: 4 });
  socket.end();
  await tick();
  assert.equal(sent.at(-1).a, "eof");
  assert.equal(sent.at(-1).seq, 6);
  socket.destroy();
});
test("unresolved gap and stale epoch terminate the request without replay", async () => {
  const { socket } = fixture();
  socket.on("error", () => {});
  socket.receive({ a: "data", v: 2, sid: "s", rid: "r", seq: 2, d: "Yg==" });
  await new Promise((r) => setTimeout(r, 80));
  assert.equal(socket.destroyed, true);
  const f = fixture();
  f.socket.on("error", () => {});
  f.socket.receive({
    a: "data",
    v: 2,
    sid: "other",
    rid: "r",
    seq: 0,
    d: "YQ==",
  });
  assert.equal(f.socket.destroyed, true);
});

test("duplicate out-of-order frames cannot extend the missing sequence deadline", async () => {
  const { socket } = fixture();
  socket.on("error", () => {});
  const frame = { a: "data", v: 2, sid: "s", rid: "r", seq: 1, d: "Yg==" };
  socket.receive(frame);
  const timer = setInterval(() => socket.receive(frame), 10);
  await new Promise((r) => setTimeout(r, 80));
  clearInterval(timer);
  assert.equal(socket.destroyed, true);
  socket.destroy();
});
