import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  connectionHealth,
  forgetConnectionRoutes,
  parseConnectionEndpoints,
  selectConnectionRoute,
} from "./connectionRoutes";
const mock = vi.hoisted(() => ({
  fetch: vi.fn(),
  profile: vi.fn(),
  save: vi.fn(),
}));
vi.mock("./relayTransport", () => ({
  relayFetch: (...args: unknown[]) => mock.fetch(...args),
}));
vi.mock("./secureStore", () => ({
  secureHostForOrigin: (...args: unknown[]) => mock.profile(...args),
  updateSecureHostRoutes: (...args: unknown[]) => mock.save(...args),
}));
const endpoints = [
  { url: "http://192.168.1.4:7680", kind: "lan" as const },
  { url: "https://relay.test", kind: "relay" as const },
];
const profile = {
  namespace: "pinned",
  instance_id: "host",
  endpoints,
  grant: { signature: "grant" },
};
beforeEach(() => {
  mock.profile.mockResolvedValue(profile);
  mock.fetch.mockImplementation(
    async () => new Response(JSON.stringify({ instance_id: "host" })),
  );
});
afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});
it("rejects unsafe origins and unbounded candidate lists", () => {
  for (const url of [
    "https://host/path",
    "https://user:secret@host",
    "http://127.0.0.1",
    "http://169.254.1.1",
  ])
    expect(() => parseConnectionEndpoints([{ url, kind: "lan" }])).toThrow();
  expect(() => parseConnectionEndpoints(Array(9).fill(endpoints[0]))).toThrow();
});
it("returns a proven LAN route without waiting for the relay head start", async () => {
  vi.useFakeTimers();
  expect(
    await selectConnectionRoute({
      endpoints,
      expectedInstanceId: "host",
    }),
  ).toBe(endpoints[0]!.url);
  expect(mock.fetch).toHaveBeenCalledTimes(1);
  expect(mock.fetch.mock.calls[0]![0]).toBe(`${endpoints[0]!.url}/api/status`);
  expect(mock.fetch.mock.calls[0]![1]).not.toHaveProperty("body");
});
it("never accepts a different instance or an unpaired route", async () => {
  mock.fetch.mockResolvedValue(
    new Response(JSON.stringify({ instance_id: "other" })),
  );
  await expect(
    selectConnectionRoute({
      endpoints: [endpoints[0]!],
      expectedInstanceId: "host",
    }),
  ).rejects.toThrow("not reachable");
  mock.profile.mockResolvedValue(null);
  await expect(
    selectConnectionRoute({
      endpoints: [endpoints[0]!],
      expectedInstanceId: "host",
    }),
  ).rejects.toThrow("not reachable");
  expect(mock.fetch).toHaveBeenCalledTimes(1);
});
it("probes the pinned encrypted LAN route from HTTPS Studio", async () => {
  vi.stubGlobal("location", { protocol: "https:" });
  expect(
    await selectConnectionRoute({
      endpoints,
      expectedInstanceId: "host",
    }),
  ).toBe(endpoints[0]!.url);
  expect(mock.fetch).toHaveBeenCalledTimes(1);
});
it("falls back to the encrypted relay when the browser refuses local access", async () => {
  vi.stubGlobal("location", { protocol: "https:" });
  mock.fetch.mockImplementation(async (url: string) => {
    if (url.startsWith(endpoints[0]!.url))
      throw new DOMException("Blocked", "SecurityError");
    return new Response(JSON.stringify({ instance_id: "host" }));
  });
  expect(
    await selectConnectionRoute({ endpoints, expectedInstanceId: "host" }),
  ).toBe(endpoints[1]!.url);
  expect(mock.fetch).toHaveBeenCalledTimes(2);
});
it("keeps an explicit refusal authoritative instead of roaming", async () => {
  const error = Object.assign(new Error("Revoked"), { status: 403 });
  const read = vi.fn().mockRejectedValue(error);
  await expect(
    connectionHealth({
      hostId: "refused",
      baseUrl: endpoints[0]!.url,
      apiKey: null,
      instanceId: "host",
      read,
    }),
  ).rejects.toBe(error);
  expect(read).toHaveBeenCalledTimes(1);
});
it("fences a late health result after its machine was removed", async () => {
  let finish!: (value: { instance_id: string }) => void;
  const read = vi.fn(
    () =>
      new Promise<{ instance_id: string }>((resolve) => {
        finish = resolve;
      }),
  );
  const pending = connectionHealth({
    hostId: "removed",
    baseUrl: endpoints[0]!.url,
    apiKey: null,
    read,
  });
  await vi.waitFor(() => expect(read).toHaveBeenCalled());
  forgetConnectionRoutes("removed");
  finish({ instance_id: "host" });
  await expect(pending).rejects.toThrow("Host connection changed");
});
