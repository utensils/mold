import { afterEach, expect, it, vi } from "vitest";
import { hmac } from "@noble/hashes/hmac.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import {
  selectConnectionRoute,
  parseConnectionEndpoints,
} from "./connectionRoutes";
afterEach(() => vi.unstubAllGlobals());
it("validates origins and bounded endpoint lists", () => {
  expect(() =>
    parseConnectionEndpoints([{ url: "https://host/path", kind: "relay" }]),
  ).toThrow();
  expect(() =>
    parseConnectionEndpoints(
      Array(9).fill({ url: "https://host", kind: "relay" }),
    ),
  ).toThrow();
  expect(() =>
    parseConnectionEndpoints([
      { url: "https://user:secret@host", kind: "relay" },
    ]),
  ).toThrow();
});
it("proves direct routes without sending secrets and prefers LAN", async () => {
  const secret = "fixture-secret";
  const key = sha256(new TextEncoder().encode(secret));
  const fetch = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body));
    expect(body.key_tag).toBe(bytesToHex(key).slice(0, 16));
    expect(new Headers(init?.headers).has("x-api-key")).toBe(false);
    expect(String(init?.body)).not.toContain(secret);
    expect(init?.redirect).toBe("error");
    expect(init?.credentials).toBe("omit");
    return new Response(
      JSON.stringify({
        instance_id: "instance",
        proof: bytesToHex(
          hmac(
            sha256,
            key,
            new TextEncoder().encode(
              `mold-connection-proof-v1\napi\n${body.nonce}\ninstance`,
            ),
          ),
        ),
      }),
    );
  });
  vi.stubGlobal("fetch", fetch);
  expect(
    await selectConnectionRoute({
      endpoints: [
        { url: "https://relay.test", kind: "relay" },
        { url: "http://lan.test:7680", kind: "lan" },
      ],
      expectedInstanceId: "instance",
      secret,
      kind: "api",
      secureContext: false,
    }),
  ).toBe("http://lan.test:7680");
});
it("rejects forged identity/proof without credential fallback", async () => {
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValue(
        new Response(
          JSON.stringify({ instance_id: "other", proof: "0".repeat(64) }),
        ),
      ),
  );
  await expect(
    selectConnectionRoute({
      endpoints: [{ url: "https://host.test", kind: "relay" }],
      expectedInstanceId: "instance",
      secret: "key",
      kind: "api",
    }),
  ).rejects.toThrow();
});
it("does not trust a persisted active URL and does not roam on authentication refusal", async () => {
  const { connectionHealth, rememberConnectionRoutes, forgetConnectionRoutes } =
    await import("./connectionRoutes");
  rememberConnectionRoutes("auth-test", "instance", "key", [
    { url: "https://host.test", kind: "relay" },
  ]);
  const read = vi
    .fn()
    .mockRejectedValue(Object.assign(new Error("refused"), { status: 401 }));
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  await expect(
    connectionHealth({
      hostId: "auth-test",
      baseUrl: "https://host.test",
      instanceId: "instance",
      apiKey: "key",
      read,
    }),
  ).rejects.toThrow("refused");
  expect(read).toHaveBeenCalledTimes(1);
  expect(fetch).not.toHaveBeenCalled();
  forgetConnectionRoutes("auth-test");
});
it("does not learn after a host has been removed while status is pending", async () => {
  const { connectionHealth } = await import("./connectionRoutes");
  let current = true;
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  await expect(
    connectionHealth({
      hostId: "removed-test",
      baseUrl: "https://host.test",
      apiKey: "key",
      isCurrent: () => current,
      read: async () => {
        current = false;
        return { instance_id: "instance" };
      },
    }),
  ).rejects.toThrow();
  expect(fetch).not.toHaveBeenCalled();
});
it("HTTPS browsers never probe insecure advertised addresses", async () => {
  const secret = "key",
    key = sha256(new TextEncoder().encode(secret));
  const fetch = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
    const b = JSON.parse(String(init?.body));
    return new Response(
      JSON.stringify({
        instance_id: "instance",
        proof: bytesToHex(
          hmac(
            sha256,
            key,
            new TextEncoder().encode(
              `mold-connection-proof-v1\napi\n${b.nonce}\ninstance`,
            ),
          ),
        ),
      }),
    );
  });
  vi.stubGlobal("fetch", fetch);
  expect(
    await selectConnectionRoute({
      endpoints: [
        { url: "http://lan.test", kind: "lan" },
        { url: "https://relay.test", kind: "relay" },
      ],
      expectedInstanceId: "instance",
      secret,
      kind: "api",
      secureContext: true,
    }),
  ).toBe("https://relay.test");
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch.mock.calls[0]?.[0]).toBe(
    "https://relay.test/api/connection-probe",
  );
});
it("retains a healthy proven route briefly rather than probing every health refresh", async () => {
  const { connectionHealth, rememberConnectionRoutes, forgetConnectionRoutes } =
    await import("./connectionRoutes");
  const secret = "key",
    key = sha256(new TextEncoder().encode(secret));
  rememberConnectionRoutes("hysteresis", "instance", secret, [
    { url: "http://lan.test", kind: "lan" },
    { url: "https://relay.test", kind: "relay" },
  ]);
  const fetch = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
    const b = JSON.parse(String(init?.body));
    return new Response(
      JSON.stringify({
        instance_id: "instance",
        proof: bytesToHex(
          hmac(
            sha256,
            key,
            new TextEncoder().encode(
              `mold-connection-proof-v1\napi\n${b.nonce}\ninstance`,
            ),
          ),
        ),
      }),
    );
  });
  vi.stubGlobal("fetch", fetch);
  const read = vi.fn(async () => ({ instance_id: "instance" }));
  const options = {
    hostId: "hysteresis",
    baseUrl: "https://relay.test",
    apiKey: secret,
    instanceId: "instance",
    read,
    secureContext: false,
  };
  expect((await connectionHealth(options)).baseUrl).toBe("http://lan.test");
  expect((await connectionHealth(options)).baseUrl).toBe("http://lan.test");
  expect(fetch).toHaveBeenCalledTimes(2);
  expect(read).toHaveBeenNthCalledWith(2, "http://lan.test");
  forgetConnectionRoutes("hysteresis");
});
it("does not delay authenticated status while additive address discovery stalls", async () => {
  const { connectionHealth } = await import("./connectionRoutes");
  const controller = new AbortController();
  vi.stubGlobal(
    "fetch",
    vi.fn(
      () =>
        new Promise((_resolve, reject) =>
          controller.signal.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          ),
        ),
    ),
  );
  const result = await Promise.race([
    connectionHealth({
      hostId: "slow-learning",
      baseUrl: "https://host.test",
      apiKey: "key",
      signal: controller.signal,
      read: async () => ({ instance_id: "instance" }),
    }),
    new Promise<null>((resolve) => setTimeout(() => resolve(null), 25)),
  ]);
  controller.abort();
  expect(result).toEqual({
    value: { instance_id: "instance" },
    baseUrl: "https://host.test",
  });
});
it("matches the independent cross-language proof vector", () => {
  const key = sha256(new TextEncoder().encode("test-only-route-secret"));
  expect(bytesToHex(key).slice(0, 16)).toBe("e4c8e720b2b762b8");
  expect(
    bytesToHex(
      hmac(
        sha256,
        key,
        new TextEncoder().encode(
          `mold-connection-proof-v1\napi\n${"0123456789abcdef".repeat(4)}\nfixture-machine`,
        ),
      ),
    ),
  ).toBe("09588691253e789f49c73ec7c6bbe10c6373119985a328df283f2bb610ad6979");
});
it("never forwards a credential to an unproven cached active URL", async () => {
  const { connectionHealth, forgetConnectionRoutes } =
    await import("./connectionRoutes");
  const keyTag = bytesToHex(sha256(new TextEncoder().encode("key"))).slice(
    0,
    16,
  );
  localStorage.setItem(
    "mold.connection-routes.v1",
    JSON.stringify({
      tampered: {
        instanceId: "instance",
        keyTag,
        endpoints: [{ url: "https://host.test", kind: "relay" }],
        activeURL: "https://attacker.test",
        checkedAt: Date.now(),
        learnedAt: Date.now(),
      },
    }),
  );
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(new Response("", { status: 404 })),
  );
  const read = vi.fn(async () => ({ instance_id: "instance" }));
  await connectionHealth({
    hostId: "tampered",
    baseUrl: "https://host.test",
    apiKey: "key",
    instanceId: "instance",
    read,
  });
  expect(read).toHaveBeenCalledExactlyOnceWith("https://host.test");
  forgetConnectionRoutes("tampered");
});
