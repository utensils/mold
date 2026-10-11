import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { claimSecurePairing } from "./securePairing";
import type { MobilePairingPayload } from "./pairing";
const mock = vi.hoisted(() => ({ connect: vi.fn(), save: vi.fn() }));
const identity = vi.hoisted(() => ({
  close: vi.fn(),
  free: vi.fn(),
  sign: () => new Uint8Array(64),
}));
vi.mock("./secureCrypto", () => ({
  loadSecureCrypto: async () => {},
  checkInvite: () => identity,
  checkGrant: () => {},
  claimBytes: () => new Uint8Array(),
  hex: () => "proof",
}));
vi.mock("./secureStore", () => ({
  deviceIdentity: async () => identity,
  saveSecureHost: (...args: unknown[]) => mock.save(...args),
}));
vi.mock("./secureChannel", () => ({
  SecureChannel: { connect: (...args: unknown[]) => mock.connect(...args) },
}));
const payload = {
  base_url: "https://relay.test",
  endpoints: [{ kind: "lan", url: "http://192.168.1.4:7680" }],
  websocket_url: "wss://relay.test/production",
  instance_id: "host",
  namespace: "pinned",
  invite: {},
} as MobilePairingPayload;
beforeEach(() => {
  vi.stubGlobal("location", { protocol: "https:" });
  mock.connect.mockResolvedValue({
    fetch: async () =>
      new Response(
        JSON.stringify({
          instance_id: "host",
          api_key: "mold_pair_fixture",
          grant: {},
        }),
      ),
    handshakeHash: () => new Uint8Array(32),
    close: vi.fn(),
  });
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
  vi.resetAllMocks();
  vi.unstubAllGlobals();
});
it("claims over pinned LAN first from HTTPS Studio", async () => {
  await claimSecurePairing(payload, { name: "Browser", kind: "browser" });
  expect(mock.connect).toHaveBeenCalledTimes(1);
  expect(mock.connect.mock.calls[0]![0]).toMatchObject({
    websocket: "ws://192.168.1.4:7680/_mold/secure",
  });
  expect(mock.save).toHaveBeenCalledTimes(1);
});
it("uses the encrypted cloud when browser policy blocks local WebSockets", async () => {
  mock.connect.mockRejectedValueOnce(
    new DOMException("Blocked", "SecurityError"),
  );
  await claimSecurePairing(payload, { name: "Browser", kind: "browser" });
  expect(mock.connect).toHaveBeenCalledTimes(2);
  expect(mock.connect.mock.calls[1]![0]).not.toHaveProperty("websocket");
  expect(mock.save).toHaveBeenCalledTimes(1);
});

it("allows time to accept a first local-network permission prompt", async () => {
  vi.useFakeTimers();
  vi.spyOn(AbortSignal, "timeout").mockImplementation((milliseconds) => {
    const controller = new AbortController();
    setTimeout(
      () => controller.abort(new DOMException("Timed out", "TimeoutError")),
      milliseconds,
    );
    return controller.signal;
  });
  const connected = mock.connect.getMockImplementation()!;
  mock.connect.mockImplementation(async (options) => {
    await new Promise((resolve) => setTimeout(resolve, 4000));
    options.signal.throwIfAborted();
    return connected(options);
  });
  const result = expect(
    claimSecurePairing(
      { ...payload, base_url: payload.endpoints![0]!.url, websocket_url: null },
      { name: "Browser", kind: "browser" },
    ),
  ).resolves.toMatchObject({ instance_id: "host" });
  await vi.advanceTimersByTimeAsync(4000);
  await result;
  expect(mock.save).toHaveBeenCalledTimes(1);
});
