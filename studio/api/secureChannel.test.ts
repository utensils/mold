import { describe, expect, it, vi } from "vitest";
import { SecureChannel } from "./secureChannel";
import { SecureWire } from "./secureWire";

describe("secure WebSocket backpressure", () => {
  it("treats a bounded gateway throttle envelope as lost delivery and retries exact ciphertext", async () => {
    vi.useFakeTimers();
    const sent: Array<Record<string, unknown>> = [];
    const wire = new SecureWire(
      "session",
      "peer",
      (frame) => sent.push(frame),
      () => {},
    );
    const channel = Object.assign(Object.create(SecureChannel.prototype), {
      failed: null,
      wire,
    }) as { receive(frame: Record<string, unknown>): void };
    const failure = {
      connectionId: "gateway-connection",
      message: "Internal server error",
      requestId: "gateway-request",
    };
    try {
      await wire.write(new Uint8Array([1, 2, 3]));
      expect(() => channel.receive(failure)).not.toThrow();
      await vi.advanceTimersByTimeAsync(11000);
      expect(sent.length).toBeGreaterThan(1);
      expect(
        sent.every(
          (frame) => JSON.stringify(frame) === JSON.stringify(sent[0]),
        ),
      ).toBe(true);
      for (const invalid of [
        { ...failure, v: 2 },
        { ...failure, a: "data" },
        { ...failure, connectionId: "" },
        { ...failure, requestId: "x".repeat(129) },
      ]) {
        expect(() => channel.receive(invalid)).toThrow();
      }
      const admitting = Object.assign(Object.create(SecureChannel.prototype), {
        failed: null,
        wire: null,
      }) as typeof channel;
      expect(() => admitting.receive(failure)).toThrow(/temporarily busy/);
    } finally {
      wire.close();
      vi.useRealTimers();
    }
  });

  it("accepts a complete reliable window before the browser flushes its socket", async () => {
    const socket = {
      readyState: WebSocket.OPEN,
      bufferedAmount: 0,
      send(frame: string) {
        this.bufferedAmount += new TextEncoder().encode(frame).byteLength;
      },
    };
    const channel = Object.assign(Object.create(SecureChannel.prototype), {
      failed: null,
      socket,
      wire: null,
    }) as { send(frame: Record<string, unknown>): void };
    const wire = new SecureWire(
      "session",
      "peer",
      (frame) => channel.send(frame),
      () => {},
    );
    try {
      await expect(
        Promise.all(
          Array.from({ length: 64 }, () => wire.write(new Uint8Array(16384))),
        ),
      ).resolves.toHaveLength(64);
      expect(socket.bufferedAmount).toBeGreaterThan(1024 * 1024);
      expect(socket.bufferedAmount).toBeLessThan(2 * 1024 * 1024);
      socket.bufferedAmount = 2 * 1024 * 1024;
      expect(() => channel.send({ a: "heartbeat", v: 3 })).toThrow(/buffer/);
    } finally {
      wire.close();
    }
  });
});
