import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { initSync } from "./generated/secure/mold_secure";
import {
  Identity,
  Session,
  binding,
  concat,
  encodeRecord,
  grantBytes,
  hex,
  inviteBytes,
  joinBytes,
  namespace,
  verifySignature,
} from "./secureCrypto";
import { SecureChannel } from "./secureChannel";
import fixture from "../../relay/aws/test/fixtures/admission-v3.json";

beforeAll(() =>
  initSync({
    module: readFileSync(
      join(
        dirname(fileURLToPath(import.meta.url)),
        "generated/secure/mold_secure_bg.wasm",
      ),
    ),
  }),
);
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
describe("the same Rust protocol in the browser", () => {
  it("matches the cloud's canonical challenge and certificate bytes", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(1899999990 * 1000));
    const host = {
      namespace: fixture.namespace,
      host_key: fixture.hostKey,
      host_noise_key: "00".repeat(32),
      host_binding_signature: "00".repeat(64),
      websocket_url: null,
    };
    const device = {
      ...fixture.binding.value,
      signature: fixture.binding.signature,
    };
    expect(
      hex(joinBytes(host, device, "device", fixture.challenge.value)),
    ).toBe(fixture.challenge.bytes);
    expect(
      hex(
        grantBytes(host, device, {
          ...fixture.grant.value,
          signature: fixture.grant.signature,
        }),
      ),
    ).toBe(fixture.grant.bytes);
    expect(
      hex(
        inviteBytes(host, {
          ...fixture.invite.value,
          signature: fixture.invite.signature,
        }),
      ),
    ).toBe(fixture.invite.bytes);
  });

  it("authenticates a host, exchanges binary data, and rejects replay", () => {
    const host = Identity.fromBytes(new Uint8Array(64).fill(11));
    const device = Identity.fromBytes(new Uint8Array(64).fill(12));
    const publicHost = binding(host);
    const publicDevice = binding(device);
    const config = {
      namespace: namespace(host.publicSigningKey()),
      host_key: publicHost.signingKey,
      device_key: publicDevice.signingKey,
      host_noise_key: publicHost.noiseKey,
      device_noise_key: publicDevice.noiseKey,
      sid: "web-test-session",
      rid: "web-test-device",
    };
    const client = Session.create(
      device,
      JSON.stringify({ ...config, initiator: true }),
    );
    const server = Session.create(
      host,
      JSON.stringify({ ...config, initiator: false }),
    );
    try {
      server.handshakeRead(client.handshakeWrite());
      client.handshakeRead(server.handshakeWrite());
      server.handshakeRead(client.handshakeWrite());
      expect(client.established() && server.established()).toBe(true);
      expect(client.handshakeHash()).toEqual(server.handshakeHash());
      const message = encodeRecord(
        3,
        1,
        new TextEncoder().encode("api-key and request stay inside the channel"),
      );
      const encrypted = client.seal(message);
      expect(new TextDecoder().decode(encrypted)).not.toContain("api-key");
      expect(server.open(encrypted)).toEqual(message);
      expect(() => server.open(encrypted)).toThrow();
      const bytes = concat(
        new TextEncoder().encode("MOLD3-BIND\0"),
        device.publicSigningKey(),
        device.publicNoiseKey(),
      );
      expect(
        verifySignature(device.publicSigningKey(), bytes, device.sign(bytes)),
      ).toBe(true);
    } finally {
      client.close();
      client.free();
      server.close();
      server.free();
      host.close();
      host.free();
      device.close();
      device.free();
    }
  });
});

it("lets browser policy decide whether pinned LAN WebSockets are available from HTTPS", async () => {
  vi.stubGlobal("location", { protocol: "https:" });
  const browserRefusal = new DOMException(
    "Local network permission denied",
    "SecurityError",
  );
  const socket = vi.fn(function () {
    throw browserRefusal;
  });
  vi.stubGlobal("WebSocket", socket);
  const hostIdentity = Identity.fromBytes(new Uint8Array(64).fill(11));
  const identity = Identity.fromBytes(new Uint8Array(64).fill(12));
  try {
    const keys = binding(hostIdentity);
    const host = {
      namespace: namespace(hostIdentity.publicSigningKey()),
      host_key: keys.signingKey,
      host_noise_key: keys.noiseKey,
      host_binding_signature: keys.signature,
      websocket_url: null,
    };
    const grant = {
      id: "01".repeat(16),
      expiresAt: Math.floor(Date.now() / 1000) + 600,
      epoch: 0,
      signature: "",
    };
    grant.signature = hex(
      hostIdentity.sign(grantBytes(host, binding(identity), grant)),
    );
    await expect(
      SecureChannel.connect({
        host,
        identity,
        grant,
        websocket: "ws://192.168.1.4:7680/_mold/secure",
      }),
    ).rejects.toBe(browserRefusal);
    expect(socket).toHaveBeenCalledWith(
      "ws://192.168.1.4:7680/_mold/secure?v=3",
    );
  } finally {
    hostIdentity.close();
    hostIdentity.free();
    identity.close();
    identity.free();
  }
});

it.each(["before-open", "after-open", "protocol", "auth"] as const)(
  "bounds socket startup retries to the pre-open phase: %s",
  async (phase) => {
    vi.useFakeTimers();
    const socket = vi.fn(function () {
      const transport = {
        readyState: 0,
        bufferedAmount: 0,
        onopen: null as (() => void) | null,
        onerror: null as (() => void) | null,
        onmessage: null as ((event: { data: string }) => void) | null,
        onclose: null,
        send: vi.fn(),
        close: vi.fn(),
      };
      queueMicrotask(() => {
        if (phase !== "before-open") {
          transport.readyState = 1;
          transport.onopen?.();
        }
        if (phase === "protocol")
          transport.onmessage?.({ data: JSON.stringify({ v: 2 }) });
        else if (phase === "auth")
          transport.onmessage?.({ data: JSON.stringify({ v: 3, a: "error" }) });
        else transport.onerror?.();
      });
      return transport;
    });
    vi.stubGlobal("WebSocket", Object.assign(socket, { OPEN: 1 }));
    const hostIdentity = Identity.fromBytes(new Uint8Array(64).fill(21));
    const identity = Identity.fromBytes(new Uint8Array(64).fill(22));
    try {
      const keys = binding(hostIdentity);
      const host = {
        namespace: namespace(hostIdentity.publicSigningKey()),
        host_key: keys.signingKey,
        host_noise_key: keys.noiseKey,
        host_binding_signature: keys.signature,
        websocket_url: "wss://proxy.example.invalid/production",
      };
      const grant = {
        id: "02".repeat(16),
        expiresAt: Math.floor(Date.now() / 1000) + 600,
        epoch: 0,
        signature: "",
      };
      grant.signature = hex(
        hostIdentity.sign(grantBytes(host, binding(identity), grant)),
      );
      const failed = expect(
        SecureChannel.connect({ host, identity, grant }),
      ).rejects.toThrow(
        phase === "protocol"
          ? /current secure protocol/
          : phase === "auth"
            ? /no longer paired/
            : /reach this host/,
      );
      await vi.advanceTimersByTimeAsync(1000);
      await failed;
      expect(socket).toHaveBeenCalledTimes(phase === "before-open" ? 3 : 1);
      expect(
        socket.mock.results.every(
          ({ value }) => value.close.mock.calls.length === 1,
        ),
      ).toBe(true);
      if (phase === "before-open")
        expect(
          socket.mock.results.every(
            ({ value }) => value.send.mock.calls.length === 0,
          ),
        ).toBe(true);
    } finally {
      hostIdentity.close();
      hostIdentity.free();
      identity.close();
      identity.free();
    }
  },
);
