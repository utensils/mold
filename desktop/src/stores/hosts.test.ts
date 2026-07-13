import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import type { SavedHost } from "../lib/ipc";

const appSettingsGet = vi.fn();
const appSettingsSet = vi.fn().mockResolvedValue(undefined);
const secretGet = vi.fn().mockResolvedValue(null);
const secretSet = vi.fn().mockResolvedValue(undefined);
const testRemoteHost = vi.fn();
const startLocalEngine = vi.fn();

vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: (...a: unknown[]) => appSettingsGet(...a),
    appSettingsSet: (...a: unknown[]) => appSettingsSet(...a),
    secretGet: (...a: unknown[]) => secretGet(...a),
    secretSet: (...a: unknown[]) => secretSet(...a),
    testRemoteHost: (...a: unknown[]) => testRemoteHost(...a),
    startLocalEngine: (...a: unknown[]) => startLocalEngine(...a),
  },
}));

const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", () => ({
  apiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
}));

import { useConnectionStore } from "./connection";
import { useHostsStore } from "./hosts";
import { useToastStore } from "./toasts";

function settings(overrides: Record<string, unknown> = {}) {
  return {
    mode: "local",
    remoteUrl: null,
    remoteApiKey: null,
    lastRoute: null,
    engineEnv: {},
    theme: "system",
    themeFamily: "mold",
    notifications: true,
    dockBadge: true,
    restoreLastRoute: false,
    runpodIncludeHfToken: false,
    runpodNetworkVolumeId: null,
    uiScalePercent: 100,
    updateChannel: "stable",
    savedHosts: [] as SavedHost[],
    connectedHostIds: [] as string[],
    generateTargetHost: null,
    saveRemoteOutputs: true,
    ...overrides,
  };
}

const hal: SavedHost = {
  id: "hal9000-7680",
  name: "hal9000",
  url: "http://hal9000:7680",
  lastUsedMs: 1,
};

/** Stateful settings mock: writes are visible to subsequent reads. */
function installSettings(initial: ReturnType<typeof settings>) {
  let current = initial;
  appSettingsGet.mockImplementation(() => Promise.resolve(current));
  appSettingsSet.mockImplementation((next: ReturnType<typeof settings>) => {
    current = next;
    return Promise.resolve(undefined);
  });
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  secretGet.mockResolvedValue(null);
  secretSet.mockResolvedValue(undefined);
  installSettings(settings());
  apiJsonTo.mockResolvedValue({ queue_depth: 0, queue_capacity: 8, version: null });
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
  conn.status = "ready";
});

describe("hosts store", () => {
  it("exposes the primary connection as This Mac", () => {
    const hosts = useHostsStore();
    expect(hosts.all).toHaveLength(1);
    expect(hosts.all[0]).toMatchObject({
      id: "local",
      kind: "local",
      primary: true,
      status: "ready",
      baseUrl: "http://127.0.0.1:49152",
    });
  });

  it("reconnects remembered extra hosts at boot with their own keys", async () => {
    installSettings(settings({ savedHosts: [hal], connectedHostIds: [hal.id] }));
    secretGet.mockResolvedValue("host-key");
    testRemoteHost.mockResolvedValue({ ok: true, version: "0.16.0", error: null });
    const hosts = useHostsStore();
    await hosts.init();
    expect(secretGet).toHaveBeenCalledWith("remote-api-key.hal9000-7680");
    const extra = hosts.all.find((h) => h.id === hal.id);
    expect(extra).toMatchObject({ status: "ready", apiKey: "host-key", primary: false });
  });

  it("marks an unreachable remembered host instead of blocking boot", async () => {
    installSettings(settings({ savedHosts: [hal], connectedHostIds: [hal.id] }));
    testRemoteHost.mockResolvedValue({ ok: false, version: null, error: "down" });
    // A host the probe rejected fails telemetry too — only telemetry for the
    // primary keeps succeeding.
    apiJsonTo.mockImplementation((target: { baseUrl: string }) =>
      target.baseUrl.includes("hal9000")
        ? Promise.reject(new Error("down"))
        : Promise.resolve({ queue_depth: 0, queue_capacity: 8, version: null }),
    );
    const hosts = useHostsStore();
    await hosts.init();
    expect(hosts.all.find((h) => h.id === hal.id)?.status).toBe("error");
    expect(useToastStore().items.some((t) => t.message.includes("hal9000"))).toBe(true);
  });

  it("connect() adds a host and persists it for the next boot", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: "0.16.0", error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", "key-1", "hal9000");
    expect(hosts.all.map((h) => h.id)).toContain("hal9000-7680");
    expect(secretSet).toHaveBeenCalledWith("remote-api-key.hal9000-7680", "key-1");
    const persisted = appSettingsSet.mock.lastCall?.[0] as ReturnType<typeof settings>;
    expect(persisted.connectedHostIds).toContain("hal9000-7680");
    expect(persisted.savedHosts[0]).toMatchObject({ id: "hal9000-7680", name: "hal9000" });
  });

  it("disconnect() removes the host and its persistence, keeping the saved entry", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    await hosts.disconnect("hal9000-7680");
    expect(hosts.all.map((h) => h.id)).not.toContain("hal9000-7680");
    const persisted = appSettingsSet.mock.lastCall?.[0] as ReturnType<typeof settings>;
    expect(persisted.connectedHostIds).not.toContain("hal9000-7680");
    expect(persisted.savedHosts.map((h: SavedHost) => h.id)).toContain("hal9000-7680");
  });

  it("resolveRoute(null) auto-routes to the least busy ready host", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    hosts.telemetry["local"] = { queueDepth: 4, queueCapacity: 8, version: null };
    hosts.telemetry["hal9000-7680"] = { queueDepth: 0, queueCapacity: 8, version: null };
    expect(hosts.resolveRoute(null)?.hostId).toBe("hal9000-7680");
  });

  it("resolveRoute honors an explicit pick and refuses unavailable hosts", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    expect(hosts.resolveRoute("local")?.hostId).toBe("local");
    const extra = hosts.extras.find((h) => h.id === "hal9000-7680")!;
    extra.status = "error";
    expect(hosts.resolveRoute("hal9000-7680")).toBeNull();
  });

  it("resolveRoute falls back to Auto for a stale pick whose host is gone", () => {
    const hosts = useHostsStore();
    // Only the primary exists; a persisted pick for a forgotten host must
    // route like Auto instead of wedging every generate.
    expect(hosts.resolveRoute("vanished-host")?.hostId).toBe("local");
  });

  it("hides a loopback extra that points at the same server as the primary", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("http://127.0.0.1:49152", null, null);
    // Same URL as the primary — one row, not two.
    expect(hosts.all).toHaveLength(1);
    expect(hosts.all[0]?.id).toBe("local");
  });

  it("persist keeps a previously discovered name across nameless reconnects", async () => {
    installSettings(
      settings({
        savedHosts: [hal],
      }),
    );
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("http://hal9000:7680", null, null); // no name passed
    const persisted = appSettingsSet.mock.lastCall?.[0] as ReturnType<typeof settings>;
    expect(persisted.savedHosts[0]).toMatchObject({ id: "hal9000-7680", name: "hal9000" });
  });

  it("adopt() lists an unreachable host as an errored extra, idempotently", () => {
    const hosts = useHostsStore();
    hosts.adopt("hal9000-7680", "http://hal9000:7680", "key", "hal9000");
    hosts.adopt("hal9000-7680", "http://hal9000:7680", "key", "hal9000");
    expect(hosts.extras).toHaveLength(1);
    const row = hosts.all.find((h) => h.id === "hal9000-7680");
    expect(row).toMatchObject({ status: "error", primary: false, apiKey: "key" });
    expect(row?.label).toBe("hal9000");
  });

  it("init() does not duplicate a host that was already adopted", async () => {
    installSettings(settings({ savedHosts: [hal], connectedHostIds: [hal.id] }));
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    hosts.adopt(hal.id, hal.url, null, null);
    await hosts.init();
    expect(hosts.extras.filter((h) => h.id === hal.id)).toHaveLength(1);
  });

  it("rename() updates the live label and the saved entry", async () => {
    installSettings(settings({ savedHosts: [hal], connectedHostIds: [hal.id] }));
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.init();
    await hosts.rename(hal.id, "  render box  ");
    expect(hosts.all.find((h) => h.id === hal.id)?.label).toBe("render box");
    const persisted = appSettingsSet.mock.lastCall?.[0] as ReturnType<typeof settings>;
    expect(persisted.savedHosts[0]).toMatchObject({ id: hal.id, name: "render box" });
  });

  it("labels a remote primary with its saved friendly name", async () => {
    const conn = useConnectionStore();
    conn.info = { mode: "remote", baseUrl: "http://hal9000:7680", apiKey: null };
    installSettings(settings({ savedHosts: [hal] }));
    const hosts = useHostsStore();
    await hosts.init();
    expect(hosts.primaryHost?.label).toBe("hal9000");
    await hosts.rename(hal.id, "render box");
    expect(hosts.primaryHost?.label).toBe("render box");
  });

  it("init() lists a host shadowed by the primary without re-probing it", async () => {
    // Review follow-up: the primary's id now lands in connectedHostIds, so
    // init used to burn a testRemoteHost probe on a server the app is
    // already connected to. The row still exists (so switching the primary
    // away mid-session keeps the host live) but stays hidden and unprobed.
    const conn = useConnectionStore();
    conn.info = { mode: "remote", baseUrl: "http://hal9000:7680", apiKey: null };
    installSettings(settings({ savedHosts: [hal], connectedHostIds: [hal.id] }));
    const hosts = useHostsStore();
    await hosts.init();
    expect(testRemoteHost).not.toHaveBeenCalled();
    expect(hosts.extras.find((h) => h.id === hal.id)?.status).toBe("ready");
    // Hidden while shadowed by the primary.
    expect(hosts.all.filter((h) => h.id === hal.id)).toHaveLength(1);
  });

  it("disconnect() clears a sticky generation target pointing at the removed host", async () => {
    installSettings(settings({ generateTargetHost: "hal9000-7680" }));
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    await hosts.disconnect("hal9000-7680");
    const persisted = appSettingsSet.mock.lastCall?.[0] as ReturnType<typeof settings>;
    expect(persisted.generateTargetHost).toBeNull();
  });

  it("rename() persists even when the saved-hosts entry was pruned", async () => {
    installSettings(settings({ savedHosts: [], connectedHostIds: [] }));
    const hosts = useHostsStore();
    hosts.adopt(hal.id, hal.url, null, null);
    await hosts.rename(hal.id, "render box");
    const persisted = appSettingsSet.mock.lastCall?.[0] as ReturnType<typeof settings>;
    expect(persisted.savedHosts[0]).toMatchObject({ id: hal.id, url: hal.url, name: "render box" });
  });

  it("demoteToExtra switches to built-in first, then keeps the host live as an extra", async () => {
    // Regression (Copilot review): connect() early-returns while the host is
    // still the primary, so the engine switch must happen before the re-add.
    const conn = useConnectionStore();
    conn.info = { mode: "remote", baseUrl: "http://hal9000:7680", apiKey: "key" };
    startLocalEngine.mockImplementation(() => {
      conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
      return Promise.resolve(conn.info);
    });
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.demoteToExtra(hosts.primaryHost!);
    expect(conn.mode).toBe("local");
    const row = hosts.all.find((h) => h.id === "hal9000-7680");
    expect(row).toMatchObject({ primary: false, status: "ready", apiKey: "key" });
  });

  it("refresh() pulls queue telemetry from every host", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    apiJsonTo.mockResolvedValue({ queue_depth: 2, queue_capacity: 8, version: "0.16.0" });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    await hosts.refresh();
    expect(hosts.telemetry["local"]?.queueDepth).toBe(2);
    expect(hosts.telemetry["hal9000-7680"]?.queueDepth).toBe(2);
  });
});
