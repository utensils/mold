import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => {
  vi.resetModules();
  return {
    lookup: vi.fn(),
    connect: vi.fn(),
    fetch: vi.fn(),
    current: vi.fn(),
    identity: vi.fn(),
    knownCredential: vi.fn(),
  };
});
vi.mock("./secureStore", () => ({
  secureHostForOrigin: mocks.lookup,
  isKnownSecureCredential: mocks.knownCredential,
  deviceIdentity: mocks.identity,
  assertSecureHostCurrent: mocks.current,
}));
vi.mock("./secureChannel", () => ({
  SecureChannel: { connect: mocks.connect },
}));
import {
  clearRelayInfoCache,
  configureSecureRequestAdapter,
  relayFetch,
} from "./relayTransport";

beforeEach(() => {
  configureSecureRequestAdapter(undefined);
  clearRelayInfoCache();
  vi.clearAllMocks();
  mocks.current.mockImplementation(() => {});
  mocks.identity.mockResolvedValue({ close() {}, free() {} });
  mocks.lookup.mockResolvedValue(null);
  mocks.knownCredential.mockResolvedValue(false);
  mocks.connect.mockResolvedValue({
    closed: false,
    close() {},
    fetch: mocks.fetch,
  });
});
afterEach(() => clearRelayInfoCache());

describe("E2EE-only remote requests", () => {
  it("refuses an old remote API key without making any plaintext network request", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    await expect(
      relayFetch("https://host.example/api/status", {
        headers: { "x-api-key": "old-key" },
      }),
    ).rejects.toThrow(/secure pairing/);
    expect(fetch).not.toHaveBeenCalled();
    expect(mocks.connect).not.toHaveBeenCalled();
  });
  it("places the device bearer and exact request only inside the secure channel", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    mocks.lookup.mockResolvedValue({
      namespace: "host",
      origins: ["https://host.example"],
      websocket_url: "wss://relay.example/ws",
      endpoints: [{ url: "https://host.example", kind: "relay" }],
      grant: { expiresAt: 4000000000 },
      apiKey: "new-device-secret",
    });
    mocks.fetch.mockResolvedValue(new Response("ok"));
    const response = await relayFetch(
      "https://host.example/api/queue?a=1&a=2",
      {
        method: "POST",
        headers: { "x-api-key": "stale", "x-mold-secure-peer": "spoof" },
        body: "body",
      },
    );
    expect(await response.text()).toBe("ok");
    expect(fetch).not.toHaveBeenCalled();
    const request = mocks.fetch.mock.calls[0]![0] as Request;
    expect(request.url).toBe("https://host.example/api/queue?a=1&a=2");
    expect(request.headers.get("x-api-key")).toBe("new-device-secret");
    expect(request.headers.has("x-mold-secure-peer")).toBe(false);
    expect(await request.text()).toBe("body");
  });
  it("never retries a mutation whose response was lost", async () => {
    mocks.lookup.mockResolvedValue({
      namespace: "host",
      origins: ["https://host.example"],
      websocket_url: "wss://relay.example/ws",
      endpoints: [{ url: "https://host.example", kind: "relay" }],
      grant: { expiresAt: 4000000000 },
      apiKey: "secret",
    });
    mocks.fetch.mockRejectedValue(new Error("response lost"));
    await expect(
      relayFetch("https://host.example/api/queue", {
        method: "POST",
        body: "job",
      }),
    ).rejects.toThrow("response lost");
    expect(mocks.fetch).toHaveBeenCalledTimes(1);
    expect(mocks.connect).toHaveBeenCalledTimes(1);
  });
  it("keeps the private embedded-engine loopback boundary usable", async () => {
    const fetch = vi.fn(async () => new Response("local"));
    vi.stubGlobal("fetch", fetch);
    expect(
      await (await relayFetch("http://127.0.0.1:7680/api/status")).text(),
    ).toBe("local");
    expect(mocks.lookup).toHaveBeenCalledWith("http://127.0.0.1:7680", true);
  });
  it("keeps an explicitly paired loopback host on Noise", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    mocks.lookup.mockResolvedValue({
      namespace: "host",
      origins: ["http://127.0.0.1:7680"],
      websocket_url: null,
      endpoints: [{ url: "http://127.0.0.1:7680", kind: "lan" }],
      grant: { expiresAt: 4000000000 },
      apiKey: "paired",
    });
    mocks.fetch.mockResolvedValue(new Response("secure"));
    expect(
      await (await relayFetch("http://127.0.0.1:7680/api/status")).text(),
    ).toBe("secure");
    expect(fetch).not.toHaveBeenCalled();
    expect(mocks.connect).toHaveBeenCalledTimes(1);
  });

  it("retires an old channel when the same host is paired with a replacement grant", async () => {
    const profile = {
      namespace: "host",
      origins: ["https://host.example"],
      endpoints: [],
      grant: { id: "old", expiresAt: 4000000000 },
      apiKey: "old-secret",
    };
    const close = vi.fn();
    mocks.lookup.mockResolvedValue(profile);
    mocks.connect.mockResolvedValue({
      closed: false,
      close,
      fetch: mocks.fetch,
    });
    mocks.fetch.mockImplementation(async () => new Response("ok"));
    await relayFetch("https://host.example/api/status");
    mocks.lookup.mockResolvedValue({
      ...profile,
      grant: { ...profile.grant, id: "replacement" },
      apiKey: "replacement-secret",
    });
    await relayFetch("https://host.example/api/status");
    expect(mocks.connect).toHaveBeenCalledTimes(2);
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("cancels a losing connection probe without cancelling another active waiter", async () => {
    mocks.lookup.mockResolvedValue({
      namespace: "host",
      endpoints: [],
      grant: { expiresAt: 4000000000 },
    });
    let connectionSignal!: AbortSignal;
    mocks.connect.mockImplementation(({ signal }: { signal: AbortSignal }) => {
      connectionSignal = signal;
      return new Promise((_, reject) =>
        signal.addEventListener("abort", () => reject(signal.reason), {
          once: true,
        }),
      );
    });
    const a = new AbortController();
    const b = new AbortController();
    const first = relayFetch("https://host.example/api/status", {
      signal: a.signal,
    });
    const second = relayFetch("https://host.example/api/status", {
      signal: b.signal,
    });
    const firstResult = expect(first).rejects.toThrow();
    const secondResult = expect(second).rejects.toThrow();
    await vi.waitFor(() => expect(mocks.connect).toHaveBeenCalledTimes(1));
    a.abort();
    await firstResult;
    expect(connectionSignal.aborted).toBe(false);
    b.abort();
    await secondResult;
    expect(connectionSignal.aborted).toBe(true);
    expect(mocks.fetch).not.toHaveBeenCalled();
  });

  it("closes a connection completed after its pairing was forgotten without dispatching", async () => {
    mocks.lookup.mockResolvedValue({
      namespace: "host",
      endpoints: [],
      grant: { expiresAt: 4000000000 },
      apiKey: "secret",
    });
    let finish!: (value: unknown) => void;
    mocks.connect.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const pending = relayFetch("https://host.example/api/queue", {
      method: "POST",
      body: "job",
    });
    const rejected = expect(pending).rejects.toThrow("pairing changed");
    await vi.waitFor(() => expect(mocks.connect).toHaveBeenCalledOnce());
    mocks.current.mockImplementation(() => {
      throw new Error("pairing changed");
    });
    const close = vi.fn();
    finish({ closed: false, close, fetch: mocks.fetch });
    await rejected;
    expect(close).toHaveBeenCalled();
    expect(mocks.fetch).not.toHaveBeenCalled();
  });

  it("refuses to connect after Forget races an asynchronous identity read", async () => {
    mocks.lookup.mockResolvedValue({
      namespace: "host",
      endpoints: [],
      grant: { expiresAt: 4000000000 },
      apiKey: "secret",
    });
    let finish!: (value: unknown) => void;
    mocks.identity.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const pending = relayFetch("https://host.example/api/queue", {
      method: "POST",
      body: "job",
    });
    const rejected = expect(pending).rejects.toThrow("pairing changed");
    await vi.waitFor(() => expect(mocks.identity).toHaveBeenCalledOnce());
    mocks.current.mockImplementation(() => {
      throw new Error("pairing changed");
    });
    const close = vi.fn();
    finish({ close, free() {} });
    await rejected;
    expect(close).toHaveBeenCalledOnce();
    expect(mocks.connect).not.toHaveBeenCalled();
    expect(mocks.fetch).not.toHaveBeenCalled();
  });

  it("refuses stale authority before reusing an already-connected channel", async () => {
    mocks.lookup.mockResolvedValue({
      namespace: "host",
      endpoints: [],
      grant: { expiresAt: 4000000000 },
      apiKey: "secret",
    });
    mocks.fetch.mockImplementation(async () => new Response("ok"));
    await relayFetch("https://host.example/api/status");
    mocks.current.mockImplementation(() => {
      throw new Error("pairing changed");
    });
    await expect(
      relayFetch("https://host.example/api/queue", {
        method: "POST",
        body: "job",
      }),
    ).rejects.toThrow("pairing changed");
    expect(mocks.fetch).toHaveBeenCalledOnce();
    expect(mocks.connect).toHaveBeenCalledOnce();
  });
});

describe("pairing-change channel isolation", () => {
  function profile(namespace: string) {
    return {
      namespace,
      origins: [`https://${namespace}.example`],
      endpoints: [],
      grant: { id: namespace, expiresAt: 4000000000, epoch: 0 },
      apiKey: `${namespace}-credential`,
    };
  }
  async function connections() {
    const closeA = vi.fn();
    const closeB = vi.fn();
    mocks.lookup.mockImplementation(async (origin: string) =>
      profile(new URL(origin).hostname.split(".")[0]!),
    );
    mocks.connect.mockImplementation(async ({ host }) => ({
      closed: false,
      close: host.namespace === "a" ? closeA : closeB,
      fetch: async () => new Response("ok"),
    }));
    await relayFetch("https://a.example/api/status");
    await relayFetch("https://b.example/api/status");
    return { closeA, closeB };
  }
  it("keeps established channels alive for route updates and certificate renewal", async () => {
    const { closeA, closeB } = await connections();
    window.dispatchEvent(
      new CustomEvent("mold:secure-pairings-changed", {
        detail: { namespace: "a", authority: "unchanged" },
      }),
    );
    expect(closeA).not.toHaveBeenCalled();
    expect(closeB).not.toHaveBeenCalled();
    await relayFetch("https://a.example/api/status");
    expect(mocks.connect).toHaveBeenCalledTimes(2);
  });
  it.each(["removed", "replaced"])(
    "closes only the affected namespace when authority is %s",
    async (authority) => {
      const { closeA, closeB } = await connections();
      window.dispatchEvent(
        new CustomEvent("mold:secure-pairings-changed", {
          detail: { namespace: "a", authority },
        }),
      );
      expect(closeA).toHaveBeenCalledOnce();
      expect(closeB).not.toHaveBeenCalled();
      await relayFetch("https://b.example/api/status");
      expect(mocks.connect).toHaveBeenCalledTimes(2);
    },
  );
});

describe("explicit Regular transport", () => {
  it("carries a regular credential directly, strips the selector and refuses redirects", async () => {
    const fetch = vi.fn(async (_request: Request) => new Response("regular"));
    vi.stubGlobal("fetch", fetch);
    await relayFetch("http://203.0.113.1:7680/api/status", {
      headers: {
        "x-api-key": "regular-key",
        "x-mold-connection-mode": "regular",
      },
    });
    const request = fetch.mock.calls[0]![0] as Request;
    expect(request.headers.get("x-api-key")).toBe("regular-key");
    expect(request.headers.has("x-mold-connection-mode")).toBe(false);
    expect(request.redirect).toBe("error");
    expect(mocks.connect).not.toHaveBeenCalled();
  });
  it("never sends a known secure credential in Regular mode", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    mocks.lookup.mockResolvedValue({ apiKey: "secure-key", endpoints: [] });
    await expect(
      relayFetch("http://host.test/api/status", {
        headers: {
          "x-api-key": "secure-key",
          "x-mold-connection-mode": "regular",
        },
      }),
    ).rejects.toThrow(/secure credential/i);
    expect(fetch).not.toHaveBeenCalled();
  });
  it("never enables Regular requests through the managed proxy", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);
    await expect(
      relayFetch("https://a.mold-link.urandom.io/api/status", {
        headers: { "x-mold-connection-mode": "regular" },
      }),
    ).rejects.toThrow(/proxy/i);
    expect(fetch).not.toHaveBeenCalled();
  });
});

it("does not transmit a secure key through Bearer authentication in Regular mode", async () => {
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  mocks.lookup.mockResolvedValue({ apiKey: "secure-key", endpoints: [] });
  await expect(
    relayFetch("http://host.test/api/status", {
      headers: {
        authorization: "Bearer secure-key",
        "x-mold-connection-mode": "regular",
      },
    }),
  ).rejects.toThrow(/secure credential/i);
  expect(fetch).not.toHaveBeenCalled();
});

it("rejects a known Secure credential on an unrelated Regular origin", async () => {
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  mocks.knownCredential.mockImplementation(
    async (key: string) => key === "secure-other-origin",
  );
  for (const credentials of [
    { "x-api-key": "secure-other-origin" },
    { authorization: "Bearer secure-other-origin" },
    { authorization: "Bearer harmless, Bearer secure-other-origin" },
    { "x-api-key": "harmless, secure-other-origin" },
  ] as Record<string, string>[]) {
    await expect(
      relayFetch("http://203.0.113.7:7680/api/status", {
        headers: { "x-mold-connection-mode": "regular", ...credentials },
      }),
    ).rejects.toThrow(/Secure credential/);
  }
  expect(fetch).not.toHaveBeenCalled();
});
