import { describe, expect, it } from "vitest";
import { SecureChannel } from "./secureChannel";
import { SecureWire } from "./secureWire";

describe("secure WebSocket backpressure", () => {
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
          Array.from({ length: 16 }, () => wire.write(new Uint8Array(16384))),
        ),
      ).resolves.toHaveLength(16);
      expect(socket.bufferedAmount).toBeGreaterThan(256 * 1024);
      expect(socket.bufferedAmount).toBeLessThan(512 * 1024);
      socket.bufferedAmount = 512 * 1024;
      expect(() => channel.send({ a: "heartbeat", v: 3 })).toThrow(/buffer/);
    } finally {
      wire.close();
    }
  });
});
