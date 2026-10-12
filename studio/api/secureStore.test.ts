import "fake-indexeddb/auto";
import { beforeAll, expect, it, vi } from "vitest";
import { webcrypto } from "node:crypto";
import type { SecureHostProfile } from "./secureCrypto";
vi.hoisted(() => vi.resetModules());
// These tests isolate persistence/concurrency. The real WASM signatures and
// cross-language vectors are exercised separately in secureCrypto.test.ts.
vi.mock("./secureCrypto", () => ({
  loadSecureCrypto: async () => {},
  checkHost() {},
  checkGrant() {},
  Identity: { generate: () => identity(), fromBytes: () => identity() },
}));
function identity() {
  return { toBytes: () => new Uint8Array(64).fill(31), close() {}, free() {} };
}
import {
  configureSecurePlatform,
  secureHostForOrigin,
  saveSecureHost,
  listSecureHosts,
  forgetSecureHost,
  updateSecureHostRoutes,
  renewSecureHostGrant,
  assertSecureHostCurrent,
} from "./secureStore";
beforeAll(() => vi.stubGlobal("crypto", webcrypto));
const original = (namespace: string) =>
  ({
    namespace,
    instance_id: "machine",
    host_key: "pin",
    host_noise_key: "noise",
    host_binding_signature: "binding",
    websocket_url: null,
    grant: {
      id: "device",
      signature: "initial",
      expiresAt: 4000000000,
      epoch: 1,
    },
    origins: ["https://host.test"],
    endpoints: [{ url: "https://host.test", kind: "relay" }],
    apiKey: "private-device-bearer",
  }) as SecureHostProfile;
async function stored(key: string): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("mold-secure-v3", 1);
    request.onsuccess = () => {
      const read = request.result
        .transaction("secrets")
        .objectStore("secrets")
        .get(key);
      read.onsuccess = () => {
        resolve(read.result);
        request.result.close();
      };
      read.onerror = reject;
    };
    request.onerror = reject;
  });
}
it("wraps identity and credentials with a nonextractable storage key", async () => {
  const profile = original("storage");
  await saveSecureHost(profile);
  const key = (await stored("wrapping-key")) as CryptoKey;
  expect(key.extractable).toBe(false);
  await expect(crypto.subtle.exportKey("raw", key)).rejects.toThrow();
  const sealed = (await stored("host:storage")) as { cipher: ArrayBuffer };
  expect(new TextDecoder().decode(sealed.cipher)).not.toContain(profile.apiKey);
  expect(
    (await listSecureHosts()).find((value) => value.namespace === "storage"),
  ).toEqual(profile);
  await forgetSecureHost("storage");
});
it("never revives a forgotten host from delayed discovery or renewal", async () => {
  const profile = original("forgotten");
  await saveSecureHost(profile);
  await forgetSecureHost(profile.namespace);
  await updateSecureHostRoutes(
    profile.namespace,
    profile.grant.signature,
    [],
    ["https://old.test"],
  );
  expect(
    await renewSecureHostGrant(profile, {
      ...profile.grant,
      signature: "renewed",
    }),
  ).toBe(false);
  expect(
    (await listSecureHosts()).some(
      (value) => value.namespace === profile.namespace,
    ),
  ).toBe(false);
});
it("does not let stale discovery or renewal roll back a replacement pairing", async () => {
  const profile = original("replacement");
  await saveSecureHost(profile);
  const replacement = {
    ...profile,
    grant: { ...profile.grant, id: "new-device", signature: "new-grant" },
    apiKey: "new-bearer",
  };
  await saveSecureHost(replacement);
  await updateSecureHostRoutes(
    profile.namespace,
    profile.grant.signature,
    [],
    ["https://old.test"],
  );
  expect(
    await renewSecureHostGrant(profile, {
      ...profile.grant,
      signature: "renewed-old",
    }),
  ).toBe(false);
  expect(
    (await listSecureHosts()).find(
      (value) => value.namespace === profile.namespace,
    ),
  ).toEqual(replacement);
  await forgetSecureHost(profile.namespace);
});
it("updates aliases without replacing the live credentials", async () => {
  const profile = original("aliases");
  await saveSecureHost(profile);
  await updateSecureHostRoutes(
    profile.namespace,
    profile.grant.signature,
    [{ url: "http://192.168.1.2:7680", kind: "lan" }],
    ["https://host.test", "http://192.168.1.2:7680"],
  );
  expect(
    (await listSecureHosts()).find(
      (value) => value.namespace === profile.namespace,
    ),
  ).toMatchObject({
    apiKey: profile.apiKey,
    grant: profile.grant,
    origins: ["https://host.test", "http://192.168.1.2:7680"],
  });
  await forgetSecureHost(profile.namespace);
});

it("waits for native credential removal and keeps a failed removal retryable", async () => {
  const profile = original("native-forget");
  const remove = vi
    .fn()
    .mockRejectedValueOnce(new Error("native key store unavailable"))
    .mockResolvedValue(undefined);
  configureSecurePlatform(vi.fn(), remove);
  try {
    await saveSecureHost(profile);
    await expect(forgetSecureHost(profile.namespace)).rejects.toThrow(
      "native key store unavailable",
    );
    expect(
      (await listSecureHosts()).some((p) => p.namespace === profile.namespace),
    ).toBe(true);
    await forgetSecureHost(profile.namespace);
    expect(remove).toHaveBeenCalledTimes(2);
    expect(
      (await listSecureHosts()).some((p) => p.namespace === profile.namespace),
    ).toBe(false);
  } finally {
    configureSecurePlatform(undefined);
  }
});
it("serializes forgetting after in-flight native registration without recreating credentials", async () => {
  const profile = original("native-race");
  let finish!: () => void;
  const registered = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const register = vi.fn(() => registered);
  const remove = vi.fn();
  configureSecurePlatform(register, remove);
  try {
    const saving = saveSecureHost(profile);
    await vi.waitFor(() => expect(register).toHaveBeenCalledOnce());
    const forgetting = forgetSecureHost(profile.namespace);
    await Promise.resolve();
    expect(remove).not.toHaveBeenCalled();
    finish();
    await Promise.all([saving, forgetting]);
    expect(remove).toHaveBeenCalledExactlyOnceWith(profile);
    expect(await secureHostForOrigin(profile.origins[0]!)).toBeNull();
  } finally {
    configureSecurePlatform(undefined);
  }
});

it("registers changed native routes even when the device grant is unchanged", async () => {
  const profile = original("native-aliases");
  const register = vi.fn();
  configureSecurePlatform(register);
  try {
    await saveSecureHost(profile);
    expect(register).toHaveBeenCalledOnce();
    const origin = "http://192.168.1.23:7680";
    await updateSecureHostRoutes(
      profile.namespace,
      profile.grant.signature,
      [{ url: origin, kind: "lan" }],
      [origin],
    );
    await secureHostForOrigin(origin);
    expect(register).toHaveBeenCalledTimes(2);
    expect(register.mock.calls[1]?.[0].origins).toEqual([origin]);
    await forgetSecureHost(profile.namespace);
  } finally {
    configureSecurePlatform(undefined);
  }
});

it.each([false, true])(
  "refuses pending authority while Forget retires a %s warm registration",
  async (warm) => {
    configureSecurePlatform(undefined);
    const profile = original(`pending-forget-${warm}`);
    await saveSecureHost(profile);
    let finish!: () => void;
    const paused = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const remove = vi.fn(() => paused);
    const register = vi.fn();
    configureSecurePlatform(register, remove);
    try {
      if (warm) await secureHostForOrigin(profile.origins[0]!);
      const forgetting = forgetSecureHost(profile.namespace);
      await vi.waitFor(() => expect(remove).toHaveBeenCalledOnce());
      const pending = secureHostForOrigin(profile.origins[0]!);
      const rejected = expect(pending).rejects.toThrow(/pairing changed/i);
      await Promise.resolve();
      finish();
      await forgetting;
      await rejected;
      expect(register).toHaveBeenCalledTimes(warm ? 1 : 0);
      expect(await secureHostForOrigin(profile.origins[0]!)).toBeNull();
    } finally {
      finish();
      configureSecurePlatform(undefined);
    }
  },
);

it("invalidates existing snapshots only for the forgotten namespace", async () => {
  const removed = original("snapshot-removed");
  const kept = {
    ...original("snapshot-kept"),
    origins: ["https://other.test"],
  };
  await saveSecureHost(removed);
  await saveSecureHost(kept);
  const old = (await secureHostForOrigin(removed.origins[0]!))!;
  const unrelated = (await secureHostForOrigin(kept.origins[0]!))!;
  await forgetSecureHost(removed.namespace);
  expect(() => assertSecureHostCurrent(old)).toThrow(/pairing changed/i);
  expect(() => assertSecureHostCurrent(unrelated)).not.toThrow();
  await forgetSecureHost(kept.namespace);
});

it.each(["save", "route", "renew"])(
  "does not let a pending %s recreate a forgotten pairing",
  async (operation) => {
    const profile = original(`pending-${operation}-forget`);
    await saveSecureHost(profile);
    const encrypt = crypto.subtle.encrypt.bind(crypto.subtle);
    let finish!: () => void;
    let began!: () => void;
    const paused = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const started = new Promise<void>((resolve) => {
      began = resolve;
    });
    const spy = vi
      .spyOn(crypto.subtle, "encrypt")
      .mockImplementationOnce(async (...args) => {
        began();
        await paused;
        return encrypt(...args);
      });
    try {
      const saving =
        operation === "save"
          ? saveSecureHost({ ...profile, apiKey: "replacement" })
          : operation === "route"
            ? updateSecureHostRoutes(
                profile.namespace,
                profile.grant.signature,
                [],
                ["https://old.test"],
              )
            : renewSecureHostGrant(profile, {
                ...profile.grant,
                signature: "renewed",
              });
      const completed =
        operation === "save"
          ? expect(saving).rejects.toThrow(/pairing changed/i)
          : saving;
      await started;
      await forgetSecureHost(profile.namespace);
      finish();
      await completed;
      expect(await secureHostForOrigin(profile.origins[0]!)).toBeNull();
    } finally {
      finish();
      spy.mockRestore();
    }
  },
);

it("invalidates another tab's removed authority without invalidating renewals or other hosts", async () => {
  const profile = original("cross-tab-removed");
  const kept = {
    ...original("cross-tab-kept"),
    origins: ["https://other.test"],
  };
  await saveSecureHost(profile);
  await saveSecureHost(kept);
  const old = (await secureHostForOrigin(profile.origins[0]!))!;
  const unrelated = (await secureHostForOrigin(kept.origins[0]!))!;
  const other = new BroadcastChannel("mold-secure-v3");
  try {
    other.postMessage({ namespace: profile.namespace, authority: "unchanged" });
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(() => assertSecureHostCurrent(old)).not.toThrow();
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open("mold-secure-v3", 1);
      request.onsuccess = () => {
        const transaction = request.result.transaction("secrets", "readwrite");
        transaction.objectStore("secrets").delete(`host:${profile.namespace}`);
        transaction.oncomplete = () => {
          request.result.close();
          resolve();
        };
        transaction.onerror = reject;
      };
      request.onerror = reject;
    });
    other.postMessage({ namespace: profile.namespace, authority: "removed" });
    await vi.waitFor(() =>
      expect(() => assertSecureHostCurrent(old)).toThrow(/pairing changed/i),
    );
    expect(() => assertSecureHostCurrent(unrelated)).not.toThrow();
    expect(await secureHostForOrigin(profile.origins[0]!)).toBeNull();
  } finally {
    other.close();
    await forgetSecureHost(kept.namespace);
  }
});

it("publishes namespace and authority on local and cross-tab notifications", async () => {
  const changes: unknown[] = [];
  const receive = (event: Event) => changes.push((event as CustomEvent).detail);
  window.addEventListener("mold:secure-pairings-changed", receive);
  const profile = original("scoped-notifications");
  const other = new BroadcastChannel("mold-secure-v3");
  try {
    await saveSecureHost(profile);
    expect(changes).toContainEqual({
      namespace: profile.namespace,
      authority: "replaced",
    });
    other.postMessage({ namespace: profile.namespace, authority: "unchanged" });
    await vi.waitFor(() =>
      expect(changes).toContainEqual({
        namespace: profile.namespace,
        authority: "unchanged",
      }),
    );
    await forgetSecureHost(profile.namespace);
    expect(changes).toContainEqual({
      namespace: profile.namespace,
      authority: "removed",
    });
  } finally {
    other.close();
    window.removeEventListener("mold:secure-pairings-changed", receive);
  }
});

it("retains only bounded Secure credential digests after forget and reload", async () => {
  const store = await import("./secureStore");
  const profile = {
    ...original("credential-history"),
    apiKey: "secure-history-secret",
  };
  await store.saveSecureHost(profile);
  expect(await store.isKnownSecureCredential(profile.apiKey)).toBe(true);
  await store.forgetSecureHost(profile.namespace);
  vi.resetModules();
  const reloaded = await import("./secureStore");
  expect(await reloaded.isKnownSecureCredential(profile.apiKey)).toBe(true);
  expect(await reloaded.isKnownSecureCredential("unrelated-regular-key")).toBe(
    false,
  );
  const digests = await stored("secure-credential-digests");
  expect(JSON.stringify(digests)).not.toContain(profile.apiKey);
  expect(digests).toEqual(
    expect.arrayContaining([expect.stringMatching(/^[a-f0-9]{64}$/)]),
  );
});

it("allows Regular-only plain HTTP clients without WebCrypto or a Secure history", async () => {
  const { IDBFactory } = await import("fake-indexeddb");
  const previousDatabase = globalThis.indexedDB;
  const previousCrypto = globalThis.crypto;
  try {
    vi.stubGlobal("indexedDB", new IDBFactory());
    vi.stubGlobal("crypto", {});
    vi.resetModules();
    const store = await import("./secureStore");
    expect(await store.isKnownSecureCredential("regular-only-key")).toBe(false);
  } finally {
    vi.stubGlobal("indexedDB", previousDatabase);
    vi.stubGlobal("crypto", previousCrypto);
  }
});
