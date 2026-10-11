/** Optional S3 carrier for existing Noise bytes; see relay/aws/OPAQUE-OBJECTS.md. */
export const PRODUCTION_OBJECT_ORIGIN =
  "https://mold-relay-042506291754-us-west-2.s3.us-west-2.amazonaws.com";
const MAX_BYTES = 262144;
const MAX_PENDING = 1048576;
const THRESHOLD = 131072;
const integer = (n: unknown): n is number =>
  typeof n === "number" && Number.isSafeInteger(n) && n >= 0;
const hex32 = (s: unknown): s is string =>
  typeof s === "string" && /^[a-f0-9]{32}$/.test(s);
const hex64 = (s: unknown): s is string =>
  typeof s === "string" && /^[a-f0-9]{64}$/.test(s);
const hex = (b: Uint8Array) =>
  Array.from(b, (n) => n.toString(16).padStart(2, "0")).join("");
const encode = (b: Uint8Array) => btoa(String.fromCharCode(...b));
const decode = (s: unknown) => {
  if (typeof s !== "string" || s.length > 21848)
    throw new Error("Invalid ciphertext chunk.");
  const b = Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  if (!b.length || b.length > 16384 || encode(b) !== s)
    throw new Error("Invalid ciphertext chunk.");
  return b;
};
const digest = async (bytes: Uint8Array) =>
  hex(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes.slice().buffer)),
  );
const segment = (s: string) =>
  encodeURIComponent(s).replace(
    /[!'()*]/g,
    (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`,
  );
type Frame = Record<string, unknown>;
export type ObjectContext = {
  namespace: string;
  sid: string;
  rid: string;
  origin: string;
};
type ObjectDescriptor = { bytes: number; count: number; sha256: string };
type Reference = ObjectDescriptor & { id: string; url: string };
type Batch = {
  frames: Frame[];
  chunks: Uint8Array[];
  reference?: Frame;
  deadline: number;
  expiry?: ReturnType<typeof setTimeout>;
};
type RequestWaiter = {
  frame: Frame;
  resolve: (frame: Frame) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
  deadline: number;
};

export function objectPolicy(
  value: unknown,
  trustedOrigin = PRODUCTION_OBJECT_ORIGIN,
): { origin: string } | null {
  if (!value || typeof value !== "object") return null;
  const p = value as Frame;
  return p.version === 1 &&
    p.origin === trustedOrigin &&
    p.maxBytes === MAX_BYTES &&
    p.maxCount === 16 &&
    p.maxInFlightBytes === MAX_PENDING &&
    p.maxInFlightObjects === 4
    ? (p as { origin: string })
    : null;
}
export function validateObjectUrl(
  value: unknown,
  context: ObjectContext,
  direction: "host" | "device",
  id: string,
): string {
  if (
    typeof value !== "string" ||
    value.length > 8192 ||
    !hex32(id) ||
    !hex32(context.namespace)
  )
    throw new Error("Invalid ciphertext object address.");
  const url = new URL(value);
  const path = `/ciphertext/v3/${context.namespace}/${segment(context.sid)}/${segment(context.rid)}/${direction}/${id}`;
  if (
    url.protocol !== "https:" ||
    url.origin !== context.origin ||
    url.username ||
    url.password ||
    url.hash ||
    url.pathname !== path ||
    url.href !== value
  )
    throw new Error("Untrusted ciphertext object address.");
  return value;
}
export async function packObject(
  chunks: Uint8Array[],
): Promise<{ bytes: Uint8Array; sha256: string }> {
  const length = chunks.reduce((sum, b) => sum + b.length + 4, 0);
  if (
    !chunks.length ||
    chunks.length > 16 ||
    length > MAX_BYTES ||
    chunks.some((b) => !b.length || b.length > 16384)
  )
    throw new Error("Ciphertext object exceeds its bound.");
  const bytes = new Uint8Array(length);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  for (const chunk of chunks) {
    view.setUint32(offset, chunk.length);
    bytes.set(chunk, offset + 4);
    offset += chunk.length + 4;
  }
  return { bytes, sha256: await digest(bytes) };
}
function descriptor(value: unknown): ObjectDescriptor {
  if (!value || typeof value !== "object")
    throw new Error("Invalid ciphertext object.");
  const o = value as Frame;
  if (
    !integer(o.bytes) ||
    o.bytes < 5 ||
    o.bytes > MAX_BYTES ||
    !integer(o.count) ||
    o.count < 1 ||
    o.count > 16 ||
    !hex64(o.sha256)
  )
    throw new Error("Invalid ciphertext object bounds.");
  return o as ObjectDescriptor;
}
export async function unpackObject(
  bytes: Uint8Array,
  expected: ObjectDescriptor,
): Promise<Uint8Array[]> {
  descriptor(expected);
  if (
    bytes.length !== expected.bytes ||
    (await digest(bytes)) !== expected.sha256
  )
    throw new Error("Ciphertext object verification failed.");
  const chunks: Uint8Array[] = [];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  for (let n = 0; n < expected.count; n++) {
    if (offset + 4 > bytes.length)
      throw new Error("Truncated ciphertext object.");
    const length = view.getUint32(offset);
    offset += 4;
    if (length < 1 || length > 16384 || offset + length > bytes.length)
      throw new Error("Invalid ciphertext object chunk.");
    chunks.push(bytes.subarray(offset, offset + length));
    offset += length;
  }
  if (offset !== bytes.length)
    throw new Error("Ciphertext object has trailing bytes.");
  return chunks;
}

/** One device/session. Memory and fetch authority are independent of peer input. */
export class CipherObjectCarrier {
  private readonly abort = new AbortController();
  private staged: { frame: Frame; bytes: Uint8Array }[] = [];
  private stagedBytes = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private ack: Frame | undefined;
  private lastAck: Frame | undefined;
  private ackTimer: ReturnType<typeof setTimeout> | undefined;
  private readonly batches = new Map<number, Batch>();
  private readonly resend = new Set<Batch>();
  private readonly requests = new Map<string, RequestWaiter>();
  private readonly downloads = new Map<
    string,
    { signature: string; task: Promise<void> }
  >();
  private downloadingBytes = 0;
  private disabled = false;

  constructor(
    private readonly context: ObjectContext,
    private readonly emit: (frame: Frame) => void,
    private readonly deliver: (frame: Frame) => void,
    private readonly receiveNext: () => number,
    private readonly failed: (error: Error) => void,
  ) {}

  send(frame: Frame): void {
    if (this.abort.signal.aborted) return;
    if (frame.a === "ack") {
      this.ack = frame;
      this.lastAck = frame;
      this.ackTimer ??= setTimeout(() => this.guard(() => this.flushAck()), 8);
      return;
    }
    if (frame.a !== "data" || this.disabled) {
      this.emit(frame);
      return;
    }
    if (
      !integer(frame.seq) ||
      frame.sid !== this.context.sid ||
      frame.rid !== this.context.rid
    )
      throw new Error("Invalid outgoing object session.");
    const cached = this.batches.get(frame.seq);
    if (cached) {
      if (cached.frames.find((f) => f.seq === frame.seq)?.d !== frame.d)
        throw new Error("Conflicting ciphertext retry.");
      if (cached.reference) this.resend.add(cached);
      this.timer ??= setTimeout(() => this.guard(() => this.flush()), 8);
      return;
    }
    const staged = this.staged.find((entry) => entry.frame.seq === frame.seq);
    if (staged) {
      if (staged.frame.d !== frame.d)
        throw new Error("Conflicting ciphertext retry.");
      return;
    }
    const bytes = decode(frame.d);
    if (
      this.staged.length &&
      (this.stagedBytes + bytes.length + 4 > MAX_BYTES ||
        this.staged.length === 16 ||
        (this.staged.at(-1)!.frame.seq as number) + 1 !== frame.seq)
    )
      this.flush();
    this.staged.push({ frame, bytes });
    this.stagedBytes += bytes.length + 4;
    if (this.staged.length === 16) this.flush();
    else this.timer ??= setTimeout(() => this.guard(() => this.flush()), 8);
  }

  async receive(frame: Frame): Promise<void> {
    this.abort.signal.throwIfAborted();
    if (
      frame.v !== 3 ||
      frame.sid !== this.context.sid ||
      frame.rid !== this.context.rid
    )
      throw new Error("Ciphertext object session changed.");
    if (frame.a === "object") {
      this.reply(frame);
      return;
    }
    if (frame.a !== "data" || !Object.hasOwn(frame, "o")) {
      this.deliver(frame);
      if (frame.a === "ack" && integer(frame.next)) {
        for (const batch of new Set(this.batches.values())) {
          const last = batch.frames.at(-1)!.seq as number;
          if (frame.next > last) {
            clearTimeout(batch.expiry);
            this.resend.delete(batch);
            for (const f of batch.frames) this.batches.delete(f.seq as number);
          }
        }
      }
      return;
    }
    if (Object.hasOwn(frame, "d") || !integer(frame.seq))
      throw new Error("Invalid ciphertext object envelope.");
    const o = descriptor(frame.o) as Reference;
    validateObjectUrl(o.url, this.context, "host", o.id);
    const end = frame.seq + o.count;
    const next = this.receiveNext();
    if (!Number.isSafeInteger(end) || end > next + 16)
      throw new Error("Ciphertext object exceeds the receive window.");
    if (end <= next) {
      if (this.ack) this.flushAck();
      else if (this.lastAck) this.emit(this.lastAck);
      return;
    }
    const key = `${frame.seq}:${o.id}`;
    const signature = JSON.stringify([o.id, o.bytes, o.count, o.sha256, o.url]);
    const pending = this.downloads.get(key);
    if (pending) {
      if (pending.signature !== signature)
        throw new Error("Conflicting ciphertext object retry.");
      return pending.task;
    }
    if (
      this.downloads.size >= 4 ||
      this.downloadingBytes + o.bytes > MAX_PENDING
    )
      throw new Error("Ciphertext object receive budget exceeded.");
    this.downloadingBytes += o.bytes;
    const task = this.download(frame.seq, o)
      .catch((error: unknown) => {
        // An overlapping retry may finish after another copy was consumed and
        // the broker cleared its bytes. Only a fully acknowledged range is stale.
        if (!this.abort.signal.aborted && end <= this.receiveNext()) {
          if (this.ack) this.flushAck();
          else if (this.lastAck) this.emit(this.lastAck);
          return;
        }
        throw error;
      })
      .finally(() => {
        this.downloads.delete(key);
        this.downloadingBytes -= o.bytes;
      });
    this.downloads.set(key, { signature, task });
    return task;
  }

  private flushAck(): void {
    clearTimeout(this.ackTimer);
    this.ackTimer = undefined;
    const ack = this.ack;
    this.ack = undefined;
    if (ack && !this.abort.signal.aborted) this.emit(ack);
  }
  private guard(work: () => void): void {
    try {
      if (!this.abort.signal.aborted) work();
    } catch (e) {
      this.fail(e);
    }
  }
  private fail(error: unknown): void {
    this.close();
    this.failed(
      error instanceof Error
        ? error
        : new Error("Ciphertext object transport failed."),
    );
  }
  private flush(): void {
    clearTimeout(this.timer);
    this.timer = undefined;
    for (const batch of this.resend)
      if (batch.reference) this.emit(batch.reference);
    this.resend.clear();
    const staged = this.staged;
    const size = this.stagedBytes;
    this.staged = [];
    this.stagedBytes = 0;
    if (!staged.length) return;
    if (size < THRESHOLD || this.disabled) {
      for (const entry of staged) this.emit(entry.frame);
      return;
    }
    if (new Set(this.batches.values()).size >= 4)
      throw new Error("Ciphertext object send budget exceeded.");
    const batch: Batch = {
      frames: staged.map((e) => e.frame),
      chunks: staged.map((e) => e.bytes),
      deadline: Date.now() + 300000,
    };
    for (const f of batch.frames) this.batches.set(f.seq as number, batch);
    void this.upload(batch).catch((error) => this.fail(error));
  }

  private async upload(batch: Batch): Promise<void> {
    const packed = await packObject(batch.chunks);
    this.abort.signal.throwIfAborted();
    const requestId = hex(crypto.getRandomValues(new Uint8Array(16)));
    const request: Frame = {
      a: "object",
      v: 3,
      op: "create",
      requestId,
      sid: this.context.sid,
      rid: this.context.rid,
      seq: batch.frames[0]!.seq,
      count: batch.frames.length,
      bytes: packed.bytes.length,
      sha256: packed.sha256,
    };
    const reply = await new Promise<Frame>((resolve, reject) => {
      const waiter: RequestWaiter = {
        frame: request,
        resolve,
        reject,
        deadline: Date.now() + 30000,
        timer: setTimeout(() => this.retryRequest(requestId), 5000),
      };
      this.requests.set(requestId, waiter);
      this.emit(request);
    });
    if (reply.op === "error" && reply.code === "disabled") {
      this.disabled = true;
      for (const frame of batch.frames) {
        this.batches.delete(frame.seq as number);
        this.emit(frame);
      }
      return;
    }
    for (const key of ["seq", "count", "bytes", "sha256"])
      if (reply[key] !== request[key])
        throw new Error("Ciphertext object allocation changed.");
    if (
      !hex32(reply.id) ||
      !integer(reply.expiresAt) ||
      reply.expiresAt <= Date.now() / 1000 ||
      reply.expiresAt > Date.now() / 1000 + 330
    )
      throw new Error("Invalid ciphertext object expiry.");
    const putUrl = validateObjectUrl(
      reply.putUrl,
      this.context,
      "device",
      reply.id,
    );
    const getUrl = validateObjectUrl(
      reply.getUrl,
      this.context,
      "device",
      reply.id,
    );
    const expectedHeaders = {
      "content-type": "application/octet-stream",
      "x-amz-checksum-sha256": encode(
        Uint8Array.from(packed.sha256.match(/../g)!, (s) => parseInt(s, 16)),
      ),
      "if-none-match": "*",
    };
    if (
      !reply.putHeaders ||
      typeof reply.putHeaders !== "object" ||
      Array.isArray(reply.putHeaders)
    )
      throw new Error("Invalid ciphertext upload headers.");
    const headers = new Headers(reply.putHeaders as Record<string, string>);
    if (
      [...headers].length !== 3 ||
      Object.entries(expectedHeaders).some(
        ([name, value]) => headers.get(name) !== value,
      )
    )
      throw new Error("Untrusted ciphertext upload headers.");
    batch.deadline = reply.expiresAt * 1000;
    batch.expiry = setTimeout(
      () =>
        this.fail(
          new Error("Ciphertext transfer expired before acknowledgement."),
        ),
      batch.deadline - Date.now(),
    );
    const response = await fetch(putUrl, {
      method: "PUT",
      headers,
      body: packed.bytes.slice().buffer,
      credentials: "omit",
      redirect: "error",
      referrerPolicy: "no-referrer",
      signal: AbortSignal.any([this.abort.signal, AbortSignal.timeout(20000)]),
    });
    if (!response.ok && response.status !== 412)
      throw new Error("Ciphertext upload failed.");
    await response.body?.cancel();
    this.abort.signal.throwIfAborted();
    batch.reference = {
      a: "data",
      v: 3,
      sid: this.context.sid,
      rid: this.context.rid,
      seq: request.seq,
      o: {
        id: reply.id,
        bytes: request.bytes,
        sha256: request.sha256,
        url: getUrl,
        count: request.count,
      },
    };
    this.emit(batch.reference);
  }

  private retryRequest(id: string): void {
    const waiter = this.requests.get(id);
    if (!waiter) return;
    clearTimeout(waiter.timer);
    if (this.abort.signal.aborted || Date.now() >= waiter.deadline) {
      this.requests.delete(id);
      waiter.reject(new Error("Ciphertext allocation timed out."));
      return;
    }
    this.guard(() => this.emit(waiter.frame));
    waiter.timer = setTimeout(() => this.retryRequest(id), 5000);
  }
  private reply(frame: Frame): void {
    if (!hex32(frame.requestId))
      throw new Error("Invalid ciphertext allocation response.");
    const waiter = this.requests.get(frame.requestId);
    if (!waiter) return; // Identical broker retries can arrive after their response was consumed.
    if (
      frame.op === "error" &&
      ["capacity", "not_ready", "unavailable"].includes(String(frame.code))
    ) {
      clearTimeout(waiter.timer);
      waiter.timer = setTimeout(
        () => this.retryRequest(frame.requestId as string),
        250,
      );
      return;
    }
    clearTimeout(waiter.timer);
    this.requests.delete(frame.requestId);
    if (
      frame.op === "created" ||
      (frame.op === "error" && frame.code === "disabled")
    )
      waiter.resolve(frame);
    else waiter.reject(new Error("Ciphertext allocation was refused."));
  }
  private async download(seq: number, o: Reference): Promise<void> {
    const response = await fetch(o.url, {
      credentials: "omit",
      redirect: "error",
      referrerPolicy: "no-referrer",
      signal: AbortSignal.any([this.abort.signal, AbortSignal.timeout(20000)]),
    });
    if (!response.ok || !response.body) {
      await response.body?.cancel();
      throw new Error("Ciphertext download failed.");
    }
    const length = response.headers.get("content-length");
    if (length !== null && Number(length) !== o.bytes) {
      await response.body.cancel();
      throw new Error("Ciphertext download length changed.");
    }
    const bytes = new Uint8Array(o.bytes);
    const reader = response.body.getReader();
    let offset = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (offset + value.length > o.bytes)
          throw new Error("Ciphertext download exceeds its bound.");
        bytes.set(value, offset);
        offset += value.length;
      }
    } finally {
      await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
    if (offset !== o.bytes)
      throw new Error("Ciphertext download was truncated.");
    const chunks = await unpackObject(bytes, o);
    this.abort.signal.throwIfAborted();
    for (const [index, chunk] of chunks.entries())
      this.deliver({
        a: "data",
        v: 3,
        sid: this.context.sid,
        rid: this.context.rid,
        seq: seq + index,
        d: encode(chunk),
      });
  }

  close(): void {
    if (this.abort.signal.aborted) return;
    this.abort.abort();
    clearTimeout(this.timer);
    clearTimeout(this.ackTimer);
    for (const batch of new Set(this.batches.values()))
      clearTimeout(batch.expiry);
    for (const waiter of this.requests.values()) {
      clearTimeout(waiter.timer);
      waiter.reject(new Error("Ciphertext transport closed."));
    }
    this.requests.clear();
    this.batches.clear();
    this.resend.clear();
    this.staged = [];
    this.stagedBytes = 0;
    this.ack = undefined;
  }
}
