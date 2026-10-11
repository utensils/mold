import { afterEach, describe, expect, it, vi } from "vitest";
import { SecureWire, PacketReader, type WireFrame } from "./secureWire";

afterEach(() => vi.useRealTimers());
describe("opaque relay byte delivery", () => {
  it("applies advertised credit relative to the acknowledged sequence, including outstanding frames", async () => {
    const sent: WireFrame[] = [];
    const wire = new SecureWire(
      "sid",
      "rid",
      (frame) => sent.push(frame),
      () => {},
    );
    try {
      await Promise.all(
        Array.from({ length: 4 }, () => wire.write(new Uint8Array([1]))),
      );
      wire.receive({
        a: "ack",
        v: 3,
        sid: "sid",
        rid: "rid",
        next: 1,
        credit: 3,
        ack_seq: 0,
      });
      let completed = false;
      const pending = wire.write(new Uint8Array([2])).then(() => {
        completed = true;
      });
      await Promise.resolve();
      await Promise.resolve();
      expect(completed).toBe(false);
      expect(sent).toHaveLength(4);
      wire.receive({
        a: "ack",
        v: 3,
        sid: "sid",
        rid: "rid",
        next: 4,
        credit: 16,
        ack_seq: 1,
      });
      await pending;
      expect(sent.at(-1)).toMatchObject({ seq: 4 });
    } finally {
      wire.close();
    }
  });
  it("starts retry budgets when ciphertext is transmitted, after bounded object preparation", async () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const failed = vi.fn();
    const wire = new SecureWire("sid", "rid", send, () => {}, failed, {
      deferSend: true,
      gapTimeoutMs: 90000,
    });
    await wire.write(new Uint8Array([1]));
    await vi.advanceTimersByTimeAsync(45000);
    expect(failed).not.toHaveBeenCalled();
    expect(send).toHaveBeenCalledTimes(1);
    wire.markSent(0, 1);
    await vi.advanceTimersByTimeAsync(10000);
    expect(send).toHaveBeenCalledTimes(2);
    wire.close();
  });
  it("discards delayed duplicates beyond the bounded history without decrypting again", () => {
    const consume = vi.fn();
    const send = vi.fn();
    const wire = new SecureWire("sid", "rid", send, consume);
    const frame = (seq: number) => ({
      a: "data",
      v: 3,
      sid: "sid",
      rid: "rid",
      seq,
      d: "AA==",
    });
    for (let seq = 0; seq < 40; seq++) wire.receive(frame(seq));
    expect(() => wire.receive(frame(0))).not.toThrow();
    expect(consume).toHaveBeenCalledTimes(40);
    expect(send.mock.calls.at(-1)?.[0]).toMatchObject({ a: "ack", next: 40 });
    wire.close();
  });
  it("deduplicates and reorders before bytes reach Noise", () => {
    const sent: WireFrame[] = [];
    const received: number[] = [];
    const wire = new SecureWire(
      "sid",
      "rid",
      (frame) => sent.push(frame),
      (bytes) => received.push(...bytes),
    );
    const frame = (seq: number, value: number) => ({
      a: "data",
      v: 3,
      sid: "sid",
      rid: "rid",
      seq,
      d: btoa(String.fromCharCode(value)),
    });
    wire.receive(frame(1, 2));
    expect(received).toEqual([]);
    wire.receive(frame(0, 1));
    wire.receive(frame(1, 2));
    expect(received).toEqual([1, 2]);
    expect(sent.at(-1)).toMatchObject({ a: "ack", next: 2, credit: 16 });
    expect(() => wire.receive(frame(1, 9))).toThrow(/duplicate/);
    wire.close();
  });

  it("retries the exact ciphertext and never exceeds the sixteen-frame window", async () => {
    vi.useFakeTimers();
    const sent: WireFrame[] = [];
    const wire = new SecureWire(
      "sid",
      "rid",
      (frame) => sent.push(frame),
      () => {},
    );
    const pending = Promise.all(
      Array.from({ length: 4 }, () => wire.write(new Uint8Array(65539))),
    );
    await vi.advanceTimersByTimeAsync(0);
    expect(sent.filter((frame) => frame.a === "data")).toHaveLength(16);
    await vi.advanceTimersByTimeAsync(10000);
    expect(sent.slice(16)).toEqual(sent.slice(0, 16));
    wire.receive({
      a: "ack",
      v: 3,
      sid: "sid",
      rid: "rid",
      next: 16,
      credit: 16,
      ack_seq: 0,
    });
    await pending;
    expect(sent.at(-1)).toMatchObject({ a: "data", seq: 19 });
    wire.close();
  });

  it("rejects cross-session, malformed base64, and missing-frame overflow", () => {
    const invalid = [
      { sid: "different", seq: 0, d: "AA==" },
      { seq: 0, d: "AA" },
      { seq: 16, d: "AA==" },
    ];
    for (const fields of invalid) {
      const wire = new SecureWire(
        "sid",
        "rid",
        () => {},
        () => {},
      );
      expect(() =>
        wire.receive({ a: "data", v: 3, sid: "sid", rid: "rid", ...fields }),
      ).toThrow();
      wire.close();
    }
  });

  it("frames Noise packets across arbitrary network boundaries with a bounded length", () => {
    const packets: Uint8Array[] = [];
    const reader = new PacketReader((bytes) => packets.push(bytes));
    reader.push(new Uint8Array([0, 0]));
    reader.push(new Uint8Array([0, 3, 7]));
    reader.push(new Uint8Array([8, 9, 0, 0, 0, 1, 10]));
    expect(packets).toEqual([new Uint8Array([7, 8, 9]), new Uint8Array([10])]);
    expect(() => reader.push(new Uint8Array([0, 1, 0, 0]))).toThrow(/length/);
  });
});
