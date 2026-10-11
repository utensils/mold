import { SecureMux } from "./secureMux";
import { PacketReader, SecureWire, framePacket } from "./secureWire";
import { CipherObjectCarrier, objectPolicy } from "./secureObjects";
import {
  Identity,
  Session,
  binding,
  checkGrant,
  checkHost,
  decodeRecord,
  encodeRecord,
  hex,
  joinBytes,
  loadSecureCrypto,
  type SecureGrant,
  type SecureHostIdentity,
  type SecureInvite,
} from "./secureCrypto";

/** One authenticated end-to-end channel. A reconnect never carries queued requests. */
export class SecureChannel {
  private socket: WebSocket;
  private wire: SecureWire | null = null;
  private session: Session | null = null;
  private readonly mux: SecureMux;
  private readonly ready: Promise<void>;
  private resolveReady!: () => void;
  private rejectReady!: (reason: Error) => void;
  private heartbeat: ReturnType<typeof setInterval> | undefined;
  private timeout: ReturnType<typeof setTimeout>;
  private failed: Error | null = null;
  private challenged = false;
  private admitted: { sid: string; rid: string } | null = null;
  private accepted: { sid: string; rid: string } | null = null;
  private started = false;
  private objectOrigin: string | null = null;
  private objects: CipherObjectCarrier | null = null;

  static async connect(options: {
    host: SecureHostIdentity;
    identity: Identity;
    grant?: SecureGrant;
    invite?: { certificate: SecureInvite; identity: Identity };
    websocket?: string;
    signal?: AbortSignal;
    /** Independent operator configuration for isolated tests; never discovery data. */
    trustedObjectOrigin?: string;
  }): Promise<SecureChannel> {
    await loadSecureCrypto();
    checkHost(options.host);
    if (options.grant)
      checkGrant(options.host, options.identity, options.grant);
    if (Boolean(options.grant) === Boolean(options.invite))
      throw new Error(
        "A secure connection requires one device grant or pairing invitation.",
      );
    options.signal?.throwIfAborted();
    const channel = new SecureChannel(options);
    const abort = () =>
      channel.close(new Error("Secure connection cancelled."));
    options.signal?.addEventListener("abort", abort, { once: true });
    try {
      await channel.ready;
      options.signal?.throwIfAborted();
      return channel;
    } finally {
      options.signal?.removeEventListener("abort", abort);
    }
  }

  private constructor(
    private readonly options: {
      host: SecureHostIdentity;
      identity: Identity;
      grant?: SecureGrant;
      invite?: { certificate: SecureInvite; identity: Identity };
      websocket?: string;
      trustedObjectOrigin?: string;
    },
  ) {
    const endpoint = options.websocket ?? options.host.websocket_url;
    if (!endpoint)
      throw new Error("This host has no secure connection address.");
    const url = new URL(endpoint);
    if (
      !["ws:", "wss:"].includes(url.protocol) ||
      url.username ||
      url.password ||
      url.hash ||
      url.search
    )
      throw new Error("Invalid secure connection address.");
    if (
      url.protocol === "ws:" &&
      typeof location !== "undefined" &&
      location.protocol === "https:"
    )
      throw new Error("This browser requires the secure proxy for this host.");
    url.searchParams.set("v", "3");
    this.ready = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    this.socket = new WebSocket(url.href);
    this.mux = new SecureMux(async (record) => {
      if (!this.session?.established() || !this.wire || this.failed)
        throw new Error("Secure connection is closed.");
      const encoded = encodeRecord(
        record.kind,
        record.streamId,
        record.payload,
      );
      const cipher = this.session.seal(encoded);
      encoded.fill(0);
      await this.wire.write(framePacket(cipher));
    });
    this.timeout = setTimeout(
      () =>
        this.close(
          new Error(
            "The host did not establish a secure connection. Check that remote access is enabled.",
          ),
        ),
      20000,
    );
    this.socket.onopen = () => {
      try {
        this.send({
          a: "hello",
          v: 3,
          namespace: options.host.namespace,
          hostKey: options.host.host_key,
          role: options.invite ? "invite" : "device",
          ...(!options.invite ? { objects: 1 } : {}),
        });
      } catch {
        this.close(new Error("Could not contact the secure proxy."));
      }
    };
    this.socket.onerror = () =>
      this.close(new Error("Could not reach this host securely."));
    this.socket.onclose = () =>
      this.close(
        new Error(
          "The secure connection closed. Check the host before retrying changes.",
        ),
      );
    this.socket.onmessage = (event: MessageEvent<unknown>) => {
      try {
        if (typeof event.data !== "string" || event.data.length > 24576)
          throw new Error("Invalid secure relay message.");
        this.receive(JSON.parse(event.data) as Record<string, unknown>);
      } catch (error) {
        this.close(
          error instanceof Error
            ? error
            : new Error("Secure authentication failed."),
        );
      }
    };
  }

  get closed(): boolean {
    return this.failed !== null;
  }
  handshakeHash(): Uint8Array {
    if (!this.session?.established() || this.failed)
      throw new Error("Secure connection is closed.");
    return this.session.handshakeHash();
  }
  async fetch(request: Request): Promise<Response> {
    await this.ready;
    return this.mux.fetch(request);
  }

  private send(frame: Record<string, unknown>): void {
    if (
      this.failed ||
      this.socket.readyState !== WebSocket.OPEN ||
      this.socket.bufferedAmount > 131072
    )
      throw new Error("Secure connection is not accepting data.");
    this.socket.send(JSON.stringify(frame));
    if (frame.a === "data" && typeof frame.seq === "number")
      this.wire?.markSent(
        frame.seq,
        frame.o ? (frame.o as { count: number }).count : 1,
      );
  }

  private receive(frame: Record<string, unknown>): void {
    if (this.failed) return;
    if (frame.v !== 3)
      throw new Error(
        "This proxy requires the current secure protocol. Upgrade the host and pair again.",
      );
    if (frame.a === "challenge") {
      if (this.challenged || this.admitted)
        throw new Error("Repeated relay challenge.");
      this.challenged = true;
      const device = binding(this.options.identity);
      const role = this.options.invite ? "invite" : "device";
      const bytes = joinBytes(
        this.options.host,
        device,
        role,
        frame as unknown as {
          connectionId: string;
          nonce: string;
          expiresAt: number;
        },
      );
      const signer = this.options.invite?.identity ?? this.options.identity;
      this.send({
        a: "join",
        v: 3,
        binding: device,
        signature: hex(signer.sign(bytes)),
        ...(this.options.invite
          ? { invite: this.options.invite.certificate }
          : { grant: this.options.grant }),
      });
      return;
    }
    if (frame.a === "ready" || frame.a === "accept") {
      if (
        typeof frame.sid !== "string" ||
        typeof frame.rid !== "string" ||
        !/^[\x21-\x7e]{1,128}$/.test(frame.sid) ||
        !/^[\x21-\x7e]{1,128}$/.test(frame.rid)
      )
        throw new Error("Invalid secure relay session.");
      if (frame.a === "ready") {
        if (!this.challenged || this.admitted)
          throw new Error("Unexpected secure admission.");
        this.admitted = { sid: frame.sid, rid: frame.rid };
        if (!this.options.invite)
          this.objectOrigin =
            objectPolicy(frame.objects, this.options.trustedObjectOrigin)
              ?.origin ?? null;
      } else {
        if (
          this.accepted &&
          (this.accepted.sid !== frame.sid || this.accepted.rid !== frame.rid)
        )
          throw new Error("Secure peer changed.");
        this.accepted = { sid: frame.sid, rid: frame.rid };
      }
      this.start();
      return;
    }
    if (frame.a === "heartbeat") return;
    if (frame.a === "error")
      throw new Error(
        "This host is unavailable or this device is no longer paired. Check the host and retry.",
      );
    if (!this.wire) throw new Error("Data arrived before secure admission.");
    if (this.objects) {
      void this.objects
        .receive(frame)
        .catch((error) =>
          this.close(
            error instanceof Error
              ? error
              : new Error("Encrypted transfer failed."),
          ),
        );
    } else this.wire.receive(frame);
  }

  private start(): void {
    if (this.started || !this.admitted || !this.accepted) return;
    if (
      this.admitted.sid !== this.accepted.sid ||
      this.admitted.rid !== this.accepted.rid
    )
      throw new Error("Secure admission did not match the host.");
    this.started = true;
    const { host, identity } = this.options;
    const { sid, rid } = this.admitted;
    this.session = Session.create(
      identity,
      JSON.stringify({
        initiator: true,
        namespace: host.namespace,
        host_key: host.host_key,
        device_key: hex(identity.publicSigningKey()),
        host_noise_key: host.host_noise_key,
        sid,
        rid,
      }),
    );
    const packets = new PacketReader((bytes) => {
      const session = this.session!;
      if (!session.established()) {
        session.handshakeRead(bytes);
        const reply = session.handshakeWrite();
        if (!session.established())
          throw new Error("Secure handshake did not complete.");
        void this.wire!.write(framePacket(reply)).then(
          () => {
            clearTimeout(this.timeout);
            this.heartbeat = setInterval(() => {
              try {
                this.send({ a: "heartbeat", v: 3 });
              } catch {
                this.close();
              }
            }, 30000);
            this.resolveReady();
          },
          () => this.close(),
        );
      } else {
        const plain = session.open(bytes);
        const record = decodeRecord(plain);
        plain.fill(0);
        try {
          this.mux.receive({
            kind: record.kind(),
            streamId: record.streamId(),
            payload: record.payload(),
          });
        } finally {
          record.free();
        }
      }
    });
    this.wire = new SecureWire(
      sid,
      rid,
      (frame) => (this.objects ? this.objects.send(frame) : this.send(frame)),
      (bytes) => packets.push(bytes),
      (error) => this.close(error),
      this.objectOrigin ? { deferSend: true, gapTimeoutMs: 90000 } : {},
    );
    if (this.objectOrigin) {
      this.objects = new CipherObjectCarrier(
        { namespace: host.namespace, sid, rid, origin: this.objectOrigin },
        (frame) => this.send(frame),
        (frame) => this.wire!.receive(frame),
        () => this.wire!.receiveNext,
        (error) => this.close(error),
      );
    }
    void this.wire
      .write(framePacket(this.session.handshakeWrite()))
      .catch(() => this.close());
  }

  close(error: Error = new Error("The secure connection closed.")): void {
    if (this.failed) return;
    this.failed = error;
    clearTimeout(this.timeout);
    clearInterval(this.heartbeat);
    this.rejectReady(error);
    this.mux.close(error);
    this.objects?.close();
    this.wire?.close(error);
    this.session?.close();
    this.session?.free();
    this.session = null;
    this.socket.onmessage = null;
    this.socket.onclose = null;
    this.socket.onerror = null;
    this.socket.close();
  }
}
