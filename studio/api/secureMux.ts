/** HTTP semantics inside an authenticated Noise session. The relay sees none of these records. */
export interface SecureRecord {
  kind: number;
  streamId: number;
  payload: Uint8Array;
}

const WINDOW = 1048576;
const CHUNK = 16320;
const MAX_UPLOAD = 64 * 1024 * 1024;
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });
const json = (value: unknown) => encoder.encode(JSON.stringify(value));

interface Stream {
  method: string;
  resolve: (response: Response) => void;
  reject: (error: Error) => void;
  response: ReadableStreamDefaultController<Uint8Array> | null;
  headers: boolean;
  noBody: boolean;
  length: number | null;
  received: number;
  incoming: Array<{ bytes: Uint8Array; length: number }>;
  receiveCredit: number;
  consumedCredit: number;
  sendCredit: number;
  wakeUpload: (() => void) | null;
  ended: boolean;
  pulling: boolean;
  upload: ReadableStreamDefaultReader<Uint8Array> | null;
  dispose: () => void;
}

/** One session, at most 32 requests and 1 MiB of unread data per direction per request.
 * Reconnection creates a NEW mux. There is deliberately no replay queue. */
export class SecureMux {
  private readonly streams = new Map<number, Stream>();
  private nextId = 1;
  private failure: Error | null = null;

  constructor(private readonly send: (record: SecureRecord) => Promise<void>) {}

  get activeStreams(): number {
    return this.streams.size;
  }

  fetch(request: Request): Promise<Response> {
    if (this.failure) return Promise.reject(this.failure);
    if (request.signal.aborted) return Promise.reject(request.signal.reason);
    if (this.streams.size >= 32 || this.nextId > 0xffff_fffd)
      return Promise.reject(
        new Error("The secure connection is busy. Try again shortly."),
      );
    const id = this.nextId;
    this.nextId += 2;
    let resolve!: Stream["resolve"];
    let reject!: Stream["reject"];
    const result = new Promise<Response>((yes, no) => {
      resolve = yes;
      reject = no;
    });
    const abort = () =>
      this.cancel(
        id,
        request.signal.reason instanceof Error
          ? request.signal.reason
          : new Error("Request cancelled."),
      );
    const stream: Stream = {
      method: request.method,
      resolve,
      reject,
      response: null,
      headers: false,
      noBody: false,
      incoming: [],
      length: null,
      received: 0,
      receiveCredit: WINDOW,
      consumedCredit: 0,
      sendCredit: WINDOW,
      wakeUpload: null,
      ended: false,
      pulling: false,
      upload: request.body?.getReader() ?? null,
      dispose: () => request.signal.removeEventListener("abort", abort),
    };
    this.streams.set(id, stream);
    request.signal.addEventListener("abort", abort, { once: true });
    const url = new URL(request.url);
    const excluded = new Set([
      "host",
      "connection",
      "content-length",
      "transfer-encoding",
      "upgrade",
      "cookie",
    ]);
    const headers = [...request.headers.entries()]
      .map(([name, value]): [string, string] => [name.toLowerCase(), value])
      .filter(([name]) => !excluded.has(name));
    const declaredLength = request.headers.get("content-length");
    const length = !stream.upload
      ? 0
      : declaredLength !== null && /^\d+$/.test(declaredLength)
        ? Number(declaredLength)
        : null;
    if (
      length !== null &&
      (!Number.isSafeInteger(length) || length > MAX_UPLOAD)
    ) {
      this.cancel(
        id,
        new Error("This host accepts request bodies up to 64 MiB."),
      );
      return result;
    }
    void this.send({
      kind: 1,
      streamId: id,
      payload: json({
        method: request.method,
        path: url.pathname + url.search,
        headers,
        length,
      }),
    })
      .then(() => this.upload(id, stream, length))
      .catch((error: unknown) => this.close(asError(error)));
    return result;
  }

  private async upload(
    id: number,
    stream: Stream,
    expected: number | null,
  ): Promise<void> {
    let total = 0;
    try {
      if (stream.upload) {
        while (this.streams.has(id)) {
          const part = await stream.upload.read();
          if (part.done) break;
          total += part.value.length;
          if (total > MAX_UPLOAD || (expected !== null && total > expected)) {
            this.cancel(
              id,
              new Error("Upload size exceeds the permitted length."),
            );
            return;
          }
          for (let offset = 0; offset < part.value.length;) {
            while (stream.sendCredit === 0 && this.streams.has(id))
              await new Promise<void>((resolve) => {
                stream.wakeUpload = resolve;
              });
            if (!this.streams.has(id)) return;
            const size = Math.min(
              CHUNK,
              stream.sendCredit,
              part.value.length - offset,
            );
            stream.sendCredit -= size;
            await this.send({
              kind: 3,
              streamId: id,
              payload: part.value.slice(offset, offset + size),
            });
            offset += size;
          }
        }
      }
      if (!this.streams.has(id)) return;
      if (expected !== null && total !== expected) {
        this.cancel(id, new Error("Upload ended before its declared length."));
        return;
      }
      await this.send({ kind: 4, streamId: id, payload: new Uint8Array() });
    } finally {
      stream.upload?.releaseLock();
      stream.upload = null;
    }
  }

  /** Called only after authenticated decryption and Rust record validation. */
  receive(record: SecureRecord): void {
    if (this.failure) return;
    try {
      const { kind, streamId: id, payload } = record;
      if (id % 2 !== 1 || id >= this.nextId)
        throw new Error("Invalid secure response stream.");
      const stream = this.streams.get(id);
      if (!stream) return; // Frames already in flight when this local stream was cancelled.
      switch (kind) {
        case 2: {
          if (stream.headers || stream.ended)
            throw new Error("Duplicate response headers.");
          const value = JSON.parse(decoder.decode(payload)) as {
            status: number;
            headers: [string, string][];
          };
          if (
            !Number.isInteger(value.status) ||
            value.status < 200 ||
            value.status > 599 ||
            !Array.isArray(value.headers)
          )
            throw new Error("Invalid response headers.");
          stream.headers = true;
          stream.noBody =
            stream.method === "HEAD" || [204, 205, 304].includes(value.status);
          const headers = new Headers(value.headers);
          const length = headers.get("content-length");
          if (
            length !== null &&
            (!/^\d+$/.test(length) || !Number.isSafeInteger(Number(length)))
          )
            throw new Error("Invalid response length.");
          stream.length =
            stream.noBody || length === null ? null : Number(length);
          const body = stream.noBody
            ? null
            : new ReadableStream<Uint8Array>(
                {
                  start: (controller) => {
                    stream.response = controller;
                  },
                  pull: () => {
                    stream.pulling = true;
                    this.deliver(id, stream);
                  },
                  cancel: () =>
                    this.cancel(id, new Error("Response cancelled.")),
                },
                { highWaterMark: 0 },
              );
          stream.resolve(new Response(body, { status: value.status, headers }));
          break;
        }
        case 3:
          if (!stream.headers || stream.noBody || stream.ended)
            throw new Error("Response data arrived without body headers.");
          if (
            !payload.length ||
            payload.length > 16384 ||
            payload.length > stream.receiveCredit
          )
            throw new Error("Secure response exceeded its byte credit.");
          stream.received += payload.length;
          if (stream.length !== null && stream.received > stream.length)
            throw new Error("Response exceeds its declared length.");
          stream.receiveCredit -= payload.length;
          // Bound allocation count as well as bytes. A paused consumer must
          // not retain a million tiny arrays inside its one-MiB credit window.
          for (let offset = 0; offset < payload.length;) {
            let tail = stream.incoming.at(-1);
            if (!tail || tail.length === 16384) {
              tail = { bytes: new Uint8Array(16384), length: 0 };
              stream.incoming.push(tail);
            }
            const count = Math.min(
              16384 - tail.length,
              payload.length - offset,
            );
            tail.bytes.set(
              payload.subarray(offset, offset + count),
              tail.length,
            );
            tail.length += count;
            offset += count;
          }
          this.deliver(id, stream);
          break;
        case 4:
          if (!stream.headers || stream.ended)
            throw new Error("Response ended without headers.");
          if (stream.length !== null && stream.received !== stream.length)
            throw new Error("Response ended before its declared length.");
          stream.ended = true;
          this.deliver(id, stream);
          break;
        case 5:
          this.finish(
            id,
            stream,
            new Error("The host cancelled this request."),
          );
          break;
        case 6: {
          if (payload.length !== 4) throw new Error("Invalid upload credit.");
          const grant = new DataView(
            payload.buffer,
            payload.byteOffset,
            payload.byteLength,
          ).getUint32(0);
          if (grant < 1 || grant > WINDOW || stream.sendCredit + grant > WINDOW)
            throw new Error("Invalid upload credit.");
          stream.sendCredit += grant;
          stream.wakeUpload?.();
          stream.wakeUpload = null;
          break;
        }
        default:
          throw new Error("Unexpected secure response record.");
      }
    } catch (error) {
      this.close(asError(error));
      throw error;
    }
  }

  private deliver(id: number, stream: Stream): void {
    if (stream.pulling && stream.incoming.length) {
      const chunk = stream.incoming.shift()!;
      const bytes = chunk.bytes.subarray(0, chunk.length);
      stream.pulling = false;
      stream.response!.enqueue(bytes);
      stream.consumedCredit += bytes.length;
      if (stream.consumedCredit >= WINDOW / 2) {
        const payload = new Uint8Array(4);
        new DataView(payload.buffer).setUint32(0, stream.consumedCredit);
        stream.receiveCredit += stream.consumedCredit;
        stream.consumedCredit = 0;
        void this.send({ kind: 6, streamId: id, payload }).catch(
          (error: unknown) => this.close(asError(error)),
        );
      }
    }
    if (stream.ended && stream.incoming.length === 0) this.finish(id, stream);
  }

  private cancel(id: number, error: Error): void {
    const stream = this.streams.get(id);
    if (!stream) return;
    this.finish(id, stream, error);
    void this.send({
      kind: 5,
      streamId: id,
      payload: encoder.encode("cancelled"),
    }).catch((failure: unknown) => this.close(asError(failure)));
  }

  private finish(id: number, stream: Stream, error?: Error): void {
    this.streams.delete(id);
    stream.dispose();
    stream.wakeUpload?.();
    stream.wakeUpload = null;
    if (stream.upload) void stream.upload.cancel(error).catch(() => {});
    stream.incoming.length = 0;
    if (error) {
      stream.reject(error);
      stream.response?.error(error);
    } else stream.response?.close();
  }

  close(
    error: Error = new Error(
      "The secure connection closed. Check the host before retrying changes.",
    ),
  ): void {
    if (this.failure) return;
    this.failure = error;
    for (const [id, stream] of this.streams) this.finish(id, stream, error);
  }
}

function asError(value: unknown): Error {
  return value instanceof Error
    ? value
    : new Error("Secure connection failed.");
}
