/** Bounded, ordered ciphertext delivery over API Gateway's at-least-once frames. */
export type WireFrame = Record<string, unknown> & {
  a: string;
  v: number;
  sid: string;
  rid: string;
};
const WINDOW = 16;
const integer = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const encode = (bytes: Uint8Array) => btoa(String.fromCharCode(...bytes));
const decode = (value: unknown): Uint8Array => {
  if (
    typeof value !== "string" ||
    !value.length ||
    value.length > 21848 ||
    value.length % 4 ||
    !/^[A-Za-z0-9+/]*={0,2}$/.test(value)
  )
    throw new Error("Invalid relay bytes.");
  const bytes = Uint8Array.from(atob(value), (char) => char.charCodeAt(0));
  if (bytes.length > 16384 || encode(bytes) !== value)
    throw new Error("Invalid relay bytes.");
  return bytes;
};

export class SecureWire {
  private next = 0;
  private outgoing = 0;
  private credit = WINDOW;
  private ackSerial = 0;
  private lastAckSerial = -1;
  private advertisedNext = -1;
  private advertisedCredit = -1;
  private incoming = new Map<number, WireFrame>();
  private seen = new Map<number, string>();
  private pending = new Map<
    number,
    { frame: WireFrame; time: number; retries: number; sent: boolean }
  >();
  private wake: Array<{ resolve: () => void; reject: (error: Error) => void }> =
    [];
  private failure: Error | null = null;
  private gapSince: number | null = null;
  private readonly timer: ReturnType<typeof setInterval>;
  private writes: Promise<void> = Promise.resolve();

  constructor(
    readonly sid: string,
    readonly rid: string,
    private readonly send: (frame: WireFrame) => void,
    private readonly consume: (bytes: Uint8Array) => void,
    private readonly failed: (error: Error) => void = () => {},
    private readonly options: {
      deferSend?: boolean;
      gapTimeoutMs?: number;
    } = {},
  ) {
    this.timer = setInterval(() => this.retry(), 1000);
  }

  write(bytes: Uint8Array): Promise<void> {
    if (bytes.length > 65539)
      return Promise.reject(new Error("Secure packet exceeds its bound."));
    const operation = this.writes.then(async () => {
      for (let offset = 0; offset < bytes.length; offset += 16384) {
        while (
          !this.failure &&
          (this.credit === 0 || this.pending.size >= WINDOW)
        )
          await new Promise<void>((resolve, reject) =>
            this.wake.push({ resolve, reject }),
          );
        if (this.failure) throw this.failure;
        if (this.outgoing >= Number.MAX_SAFE_INTEGER)
          throw new Error("Relay sequence exhausted.");
        const frame: WireFrame = {
          a: "data",
          v: 3,
          sid: this.sid,
          rid: this.rid,
          seq: this.outgoing++,
          d: encode(bytes.subarray(offset, offset + 16384)),
        };
        this.credit--;
        this.pending.set(frame.seq as number, {
          frame,
          time: Date.now(),
          retries: 0,
          sent: !this.options.deferSend,
        });
        this.send(frame);
      }
    });
    this.writes = operation.catch((error: unknown) => {
      this.close(
        error instanceof Error ? error : new Error("Relay send failed."),
      );
    });
    return operation;
  }

  get receiveNext(): number {
    return this.next;
  }

  /** Allocation/PUT preparation has its own bound and consumes no ACK retries. */
  markSent(sequence: number, count = 1): void {
    for (let n = sequence; n < sequence + count; n++) {
      const pending = this.pending.get(n);
      if (pending) {
        pending.sent = true;
        pending.time = Date.now();
      }
    }
  }

  receive(value: unknown): void {
    if (this.failure) return;
    try {
      if (!value || typeof value !== "object")
        throw new Error("Invalid relay frame.");
      const frame = value as WireFrame;
      if (frame.v !== 3 || frame.sid !== this.sid || frame.rid !== this.rid)
        throw new Error("Relay session changed.");
      if (frame.a === "cancel" || frame.a === "eof")
        throw new Error("The host disconnected.");
      if (frame.a === "accept") return;
      if (frame.a === "ack") {
        if (
          !integer(frame.ack_seq) ||
          !integer(frame.next) ||
          !integer(frame.credit) ||
          frame.credit > WINDOW ||
          frame.next > this.outgoing
        )
          throw new Error("Invalid relay acknowledgement.");
        if (frame.ack_seq <= this.lastAckSerial) return;
        this.lastAckSerial = frame.ack_seq;
        for (const sequence of this.pending.keys())
          if (sequence < frame.next) this.pending.delete(sequence);
        this.credit = frame.credit;
        for (const waiter of this.wake.splice(0)) waiter.resolve();
        return;
      }
      if (frame.a !== "data" || !integer(frame.seq))
        throw new Error("Invalid relay data frame.");
      decode(frame.d);
      if (frame.seq < this.next) {
        const retained = this.seen.get(frame.seq);
        if (retained !== undefined && retained !== frame.d)
          throw new Error("Conflicting relay duplicate.");
        this.ack(true);
        return;
      }
      if (frame.seq >= this.next + WINDOW)
        throw new Error("Relay reorder window exceeded.");
      const old = this.incoming.get(frame.seq);
      if (old && old.d !== frame.d)
        throw new Error("Conflicting relay duplicate.");
      this.incoming.set(frame.seq, frame);
      while (this.incoming.has(this.next)) {
        const current = this.incoming.get(this.next)!;
        this.incoming.delete(this.next);
        this.seen.set(this.next++, current.d as string);
        if (this.seen.size > 32)
          this.seen.delete(this.seen.keys().next().value!);
        this.consume(decode(current.d));
      }
      this.gapSince = this.incoming.size ? (this.gapSince ?? Date.now()) : null;
      this.ack(Boolean(old));
    } catch (error) {
      this.close(
        error instanceof Error ? error : new Error("Invalid relay frame."),
      );
      throw error;
    }
  }

  private ack(force = false): void {
    const credit = WINDOW - this.incoming.size;
    if (
      !force &&
      this.advertisedNext === this.next &&
      this.advertisedCredit === credit
    )
      return;
    this.advertisedNext = this.next;
    this.advertisedCredit = credit;
    this.send({
      a: "ack",
      v: 3,
      sid: this.sid,
      rid: this.rid,
      next: this.next,
      credit,
      ack_seq: this.ackSerial++,
    });
  }

  private retry(): void {
    try {
      if (
        this.gapSince !== null &&
        Date.now() - this.gapSince >= (this.options.gapTimeoutMs ?? 30000)
      )
        throw new Error("A relay frame did not arrive.");
      for (const value of this.pending.values()) {
        if (!value.sent || Date.now() - value.time < 10000) continue;
        if (value.retries++ >= 3)
          throw new Error("Relay acknowledgement timed out.");
        value.time = Date.now();
        this.send(value.frame);
      }
    } catch (error) {
      this.close(
        error instanceof Error ? error : new Error("Relay retry failed."),
      );
    }
  }

  close(error: Error = new Error("Relay closed.")): void {
    if (this.failure) return;
    this.failure = error;
    clearInterval(this.timer);
    for (const waiter of this.wake.splice(0)) waiter.reject(error);
    this.pending.clear();
    this.incoming.clear();
    this.seen.clear();
    this.failed(error);
  }
}

export function framePacket(packet: Uint8Array): Uint8Array {
  if (!packet.length || packet.length > 65535)
    throw new Error("Invalid secure packet length.");
  const result = new Uint8Array(packet.length + 4);
  new DataView(result.buffer).setUint32(0, packet.length);
  result.set(packet, 4);
  return result;
}

/** Never allocate a peer-selected size before validating its maximum. */
export class PacketReader {
  private readonly buffer = new Uint8Array(65539);
  private size = 0;
  private wanted = 4;
  constructor(private readonly consume: (packet: Uint8Array) => void) {}
  push(bytes: Uint8Array): void {
    for (let offset = 0; offset < bytes.length;) {
      const count = Math.min(this.wanted - this.size, bytes.length - offset);
      this.buffer.set(bytes.subarray(offset, offset + count), this.size);
      this.size += count;
      offset += count;
      if (this.size < this.wanted) continue;
      if (this.wanted === 4) {
        const length = new DataView(this.buffer.buffer).getUint32(0);
        if (length < 1 || length > 65535)
          throw new Error("Invalid secure packet length.");
        this.wanted = 4 + length;
      } else {
        const packet = this.buffer.slice(4, this.wanted);
        this.size = 0;
        this.wanted = 4;
        this.consume(packet);
      }
    }
  }
}
