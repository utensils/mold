import { Duplex } from "node:stream";
import { createHash } from "node:crypto";
import { validFrame, FRAME_LIMIT } from "./router-core.mjs";
const wireError = (message) =>
  Object.assign(new Error(message), {
    name:
      {
        "Invalid relay frame": "RelayInvalidFrame",
        "Relay peer disconnected": "RelayPeerDisconnected",
        "Invalid relay acknowledgement": "RelayInvalidAcknowledgement",
        "Conflicting relay duplicate": "RelayConflictingDuplicate",
        "Relay reorder bound exceeded": "RelayReorderBound",
        "Relay bytes after EOF": "RelayBytesAfterEOF",
        "Relay sequence gap": "RelaySequenceGap",
        "Relay acknowledgement timed out": "RelayAcknowledgementTimeout",
      }[message] ?? "RelayConnectionClosed",
  });
const digest = (frame) =>
  createHash("sha256")
    .update(frame.a === "eof" ? "eof" : frame.d)
    .digest("hex");
export class RelayDuplex extends Duplex {
  constructor({ sid, rid, send, gapMs = 30000, retryMs = 10000 }) {
    super({
      readableHighWaterMark: 65536,
      writableHighWaterMark: 16384,
      allowHalfOpen: true,
    });
    Object.assign(this, { sid, rid, send, gapMs, retryMs });
    this.ackOut = 0;
    this.lastAckSeq = -1;
    this.out = 0;
    this.next = 0;
    this.credit = 4;
    this.pending = new Map();
    this.incoming = new Map();
    this.seen = new Map();
    this.waiters = [];
    this.blocked = false;
    this.remoteEnded = false;
    this.timer = setInterval(() => this.retry(), retryMs);
    this.timer.unref();
  }
  setTimeout() {
    return this;
  }
  setNoDelay() {
    return this;
  }
  setKeepAlive() {
    return this;
  }
  async transmit(a, d) {
    while (!this.destroyed && (this.credit === 0 || this.pending.size >= 4))
      await new Promise((resolve, reject) =>
        this.waiters.push({ resolve, reject }),
      );
    if (this.destroyed) throw new Error("Relay connection closed");
    const frame = {
      a,
      v: 2,
      sid: this.sid,
      rid: this.rid,
      seq: this.out++,
      ...(d === undefined ? {} : { d }),
    };
    this.pending.set(frame.seq, { frame, time: Date.now(), tries: 0 });
    this.credit--;
    this.send(frame);
  }
  _write(chunk, encoding, callback) {
    (async () => {
      for (let i = 0; i < chunk.length; i += 16384)
        await this.transmit(
          "data",
          chunk.subarray(i, i + 16384).toString("base64"),
        );
    })().then(() => callback(), callback);
  }
  _final(callback) {
    this.transmit("eof").then(() => callback(), callback);
  }
  _read() {
    this.blocked = false;
    this.drainIncoming();
    this.ack();
  }
  _destroy(error, callback) {
    clearInterval(this.timer);
    clearTimeout(this.gapTimer);
    for (const waiter of this.waiters.splice(0))
      waiter.reject(error ?? new Error("Relay closed"));
    this.pending.clear();
    this.incoming.clear();
    callback(error);
  }
  ack() {
    if (!this.destroyed)
      this.send({
        a: "ack",
        v: 2,
        sid: this.sid,
        rid: this.rid,
        next: this.next,
        ack_seq: this.ackOut++,
        credit: this.blocked ? 0 : Math.max(0, 4 - this.incoming.size),
      });
  }
  receive(frame) {
    if (this.destroyed) return;
    if (
      !validFrame(frame) ||
      frame.sid !== this.sid ||
      frame.rid !== this.rid
    ) {
      this.destroy(wireError("Invalid relay frame"));
      return;
    }
    if (frame.a === "cancel") {
      this.destroy(wireError("Relay peer disconnected"));
      return;
    }
    if (frame.a === "accept") return;
    if (frame.a === "ack") {
      if (frame.ack_seq <= this.lastAckSeq) return;
      this.lastAckSeq = frame.ack_seq;
      if (frame.next > this.out) {
        this.destroy(wireError("Invalid relay acknowledgement"));
        return;
      }
      for (const seq of this.pending.keys())
        if (seq < frame.next) this.pending.delete(seq);
      this.credit = frame.credit;
      for (const waiter of this.waiters.splice(0)) waiter.resolve();
      return;
    }
    if (frame.seq < this.next) {
      if (this.seen.get(frame.seq) !== digest(frame)) {
        this.destroy(wireError("Conflicting relay duplicate"));
        return;
      }
      this.ack();
      return;
    }
    if (this.remoteEnded || frame.seq >= this.next + 4) {
      this.destroy(wireError("Relay reorder bound exceeded"));
      return;
    }
    const old = this.incoming.get(frame.seq);
    if (old && digest(old) !== digest(frame)) {
      this.destroy(wireError("Conflicting relay duplicate"));
      return;
    }
    this.incoming.set(frame.seq, frame);
    this.drainIncoming();
    this.ack();
  }
  drainIncoming() {
    while (!this.blocked && !this.destroyed && this.incoming.has(this.next)) {
      const frame = this.incoming.get(this.next);
      this.incoming.delete(this.next);
      this.seen.set(this.next, digest(frame));
      this.next++;
      if (this.seen.size > 32) this.seen.delete(this.seen.keys().next().value);
      if (frame.a === "eof") {
        this.remoteEnded = true;
        this.push(null);
        if (this.incoming.size)
          this.destroy(wireError("Relay bytes after EOF"));
        break;
      }
      this.blocked = !this.push(Buffer.from(frame.d, "base64"));
    }
    const hasGap = this.incoming.size && !this.incoming.has(this.next);
    if (!hasGap || this.gapSequence !== this.next) {
      clearTimeout(this.gapTimer);
      this.gapTimer = undefined;
    }
    if (hasGap && !this.gapTimer) {
      this.gapSequence = this.next;
      this.gapTimer = setTimeout(
        () => this.destroy(wireError("Relay sequence gap")),
        this.gapMs,
      );
    }
  }
  retry() {
    if (this.destroyed) return;
    for (const value of this.pending.values())
      if (Date.now() - value.time >= this.retryMs) {
        if (value.tries++ >= 3) {
          this.destroy(wireError("Relay acknowledgement timed out"));
          return;
        }
        value.time = Date.now();
        this.send(value.frame);
      }
  }
}
