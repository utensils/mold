import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import type { SavedHost } from "../lib/ipc";

const appSettingsGet = vi.fn();
const appSettingsSet = vi.fn().mockResolvedValue(undefined);
const secretGet = vi.fn().mockResolvedValue(null);
const secretSet = vi.fn().mockResolvedValue(undefined);
const testRemoteHost = vi.fn();
const startLocalEngine = vi.fn();
const ensureLocalServer = vi.fn();

vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: (...a: unknown[]) => appSettingsGet(...a),
    appSettingsSet: (...a: unknown[]) => appSettingsSet(...a),
    secretGet: (...a: unknown[]) => secretGet(...a),
    secretSet: (...a: unknown[]) => secretSet(...a),
    testRemoteHost: (...a: unknown[]) => testRemoteHost(...a),
    startLocalEngine: (...a: unknown[]) => startLocalEngine(...a),
    ensureLocalServer: (...a: unknown[]) => ensureLocalServer(...a),
  },
}));

const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
}));

import { useConnectionStore } from "./connection";
import { useDownloadsStore } from "./downloads";
import { useHostModelsStore } from "./hostModels";
import { useHostsStore } from "./hosts";
import { useToastStore } from "./toasts";
import type { ModelEntry } from "../lib/api/types";

function installedModel(name: string): ModelEntry {
  return {
    name,
    family: "flux",
    size_gb: 12,
    is_loaded: false,
    hf_repo: "",
    default_steps: 28,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "",
    downloaded: true,
  };
}

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
    navRailWidth: null,
    generateParamsWidth: null,
    ...overrides,
  };
}

const hal: SavedHost = {
  id: "hal9000-7680",
  name: "hal9000",
  url: "http://hal9000:7680",
  lastUsedMs: 1,
};

const studio: SavedHost = {
  id: "studio-local-7680",
  name: "studio",
  url: "http://studio.local:7680",
  lastUsedMs: 2,
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
  ensureLocalServer.mockResolvedValue({
    kind: "embedded",
    baseUrl: "http://127.0.0.1:49152",
    apiKey: "k",
    port: 49152,
  });
  installSettings(settings());
  apiJsonTo.mockResolvedValue({ queue_depth: 0, queue_capacity: 8, version: null });
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
  conn.status = "ready";
});

describe("hosts store", () => {
  it("exposes the primary connection as this device", () => {
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

  it("reconnects the whole remembered host set at boot, local primary first", async () => {
    installSettings(
      settings({
        savedHosts: [studio, hal],
        connectedHostIds: [hal.id, studio.id],
      }),
    );
    secretGet.mockImplementation((key: string) =>
      Promise.resolve(key.endsWith(studio.id) ? "studio-key" : "hal-key"),
    );
    testRemoteHost.mockResolvedValue({ ok: true, version: "0.17.1", error: null });

    const hosts = useHostsStore();
    await hosts.init();

    // The built-in engine is always the primary; every remembered host follows.
    expect(hosts.primaryHost).toMatchObject({ id: "local", primary: true });
    expect(hosts.all.map((host) => host.id)).toEqual(["local", hal.id, studio.id]);
    expect(hosts.all.find((h) => h.id === studio.id)).toMatchObject({
      status: "ready",
      apiKey: "studio-key",
      primary: false,
    });
    expect(testRemoteHost).toHaveBeenCalledWith(hal.url, "hal-key");
    expect(testRemoteHost).toHaveBeenCalledWith(studio.url, "studio-key");
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
    const downloads = useDownloadsStore();
    const abort = new AbortController();
    downloads.hostStates["hal9000-7680"] = {
      activeJobs: [],
      queued: [],
      history: [],
      label: "hal9000",
      target: { baseUrl: "http://hal9000:7680", apiKey: null },
      subscribed: true,
      abort,
      cancelling: [],
      ready: null,
    };
    await hosts.disconnect("hal9000-7680");
    expect(hosts.all.map((h) => h.id)).not.toContain("hal9000-7680");
    expect(abort.signal.aborted).toBe(true);
    expect(downloads.hostStates["hal9000-7680"]).toBeUndefined();
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

  it('resolveRoute("capable") routes to the CUDA host over an idler Metal host', async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    hosts.telemetry["local"] = {
      queueDepth: 0,
      queueCapacity: 8,
      version: null,
      gpuInfo: { name: "Apple M3 Max", vram_total_mb: 65536, vram_used_mb: 0, backend: "metal" },
    };
    hosts.telemetry["hal9000-7680"] = {
      queueDepth: 5,
      queueCapacity: 8,
      version: null,
      gpuInfo: {
        name: "NVIDIA GeForce RTX 4090",
        vram_total_mb: 24564,
        vram_used_mb: 0,
        backend: "cuda",
      },
    };
    expect(hosts.resolveRoute("capable")?.hostId).toBe("hal9000-7680");
  });

  it('resolveRoute("capable") infers the backend from the GPU name on older servers', async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    hosts.telemetry["local"] = { queueDepth: 0, queueCapacity: 8, version: null, gpuInfo: null };
    hosts.telemetry["hal9000-7680"] = {
      queueDepth: 5,
      queueCapacity: 8,
      version: null,
      // No `backend` field — pre-0.17 server; the name still says CUDA.
      gpuInfo: { name: "NVIDIA GeForce RTX 4090", vram_total_mb: 24564, vram_used_mb: 0 },
    };
    expect(hosts.resolveRoute("capable")?.hostId).toBe("hal9000-7680");
  });

  it('resolveRoute("capable") restricts to hosts that have the model', async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    hosts.telemetry["hal9000-7680"] = {
      queueDepth: 0,
      queueCapacity: 8,
      version: null,
      gpuInfo: { name: "NVIDIA RTX 4090", vram_total_mb: 24564, vram_used_mb: 0, backend: "cuda" },
    };
    const hostModels = useHostModelsStore();
    hostModels.byHost["local"] = {
      entries: [installedModel("z-image:q8")],
      fetchedAt: Date.now(),
      error: null,
    };
    hostModels.byHost["hal9000-7680"] = {
      entries: [installedModel("flux-dev:q8")],
      fetchedAt: Date.now(),
      error: null,
    };
    // The CUDA host doesn't have the model — the Metal-less local host does.
    expect(hosts.resolveRoute("capable", "z-image:q8")?.hostId).toBe("local");
  });

  it("resolveRoute(null) with a model restricts Auto to hosts that have it", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    hosts.telemetry["local"] = { queueDepth: 0, queueCapacity: 8, version: null };
    hosts.telemetry["hal9000-7680"] = { queueDepth: 3, queueCapacity: 8, version: null };
    const hostModels = useHostModelsStore();
    hostModels.byHost["hal9000-7680"] = {
      entries: [installedModel("z-image:q8")],
      fetchedAt: Date.now(),
      error: null,
    };
    // hal9000 is busier but the only host with the model.
    expect(hosts.resolveRoute(null, "z-image:q8")?.hostId).toBe("hal9000-7680");
  });

  it("resolveRoute(null) keeps least-busy routing when no host has the model", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    hosts.telemetry["local"] = { queueDepth: 0, queueCapacity: 8, version: null };
    hosts.telemetry["hal9000-7680"] = { queueDepth: 3, queueCapacity: 8, version: null };
    const hostModels = useHostModelsStore();
    hostModels.byHost["local"] = {
      entries: [installedModel("flux-dev:q8")],
      fetchedAt: Date.now(),
      error: null,
    };
    // Nobody reports the model — the router falls back to least busy, which
    // will auto-pull it there.
    expect(hosts.resolveRoute(null, "brand-new-model")?.hostId).toBe("local");
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

  it("init() does not duplicate a host already listed as an extra", async () => {
    installSettings(settings({ savedHosts: [hal], connectedHostIds: [hal.id] }));
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    // Pre-listed (e.g. connected earlier this session) — init must not re-add.
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
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

  it("names a remote extra by its server hostname when the user hasn't renamed it", async () => {
    installSettings(settings({ savedHosts: [{ ...hal, name: null }], connectedHostIds: [hal.id] }));
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    apiJsonTo.mockResolvedValue({
      queue_depth: 0,
      queue_capacity: 8,
      version: null,
      hostname: "hal9000",
    });
    const hosts = useHostsStore();
    await hosts.init();
    await hosts.refresh();
    expect(hosts.all.find((h) => h.id === hal.id)?.label).toBe("hal9000");
    // A user rename still wins over the server hostname.
    await hosts.rename(hal.id, "render box");
    expect(hosts.all.find((h) => h.id === hal.id)?.label).toBe("render box");
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
    // A live extra whose MRU row was pruned still deserves a sticky name.
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
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
    apiJsonTo.mockResolvedValue({
      queue_depth: 2,
      queue_capacity: 8,
      version: "0.16.0",
      models_loaded: ["flux2-klein:q4"],
      gpu_info: { name: "NVIDIA GeForce RTX 4090", vram_total_mb: 24564, vram_used_mb: 8192 },
    });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    await hosts.refresh();
    expect(hosts.telemetry["local"]?.queueDepth).toBe(2);
    expect(hosts.telemetry["hal9000-7680"]?.queueDepth).toBe(2);
    // The status-bar fallback data rides the same poll.
    expect(hosts.telemetry["hal9000-7680"]?.modelsLoaded).toEqual(["flux2-klein:q4"]);
    expect(hosts.telemetry["hal9000-7680"]?.gpuInfo?.vram_total_mb).toBe(24564);
  });

  it("connect() dedupes a server already connected under another address by instance id", async () => {
    testRemoteHost.mockResolvedValue({
      ok: true,
      version: null,
      error: null,
      instanceId: "uuid-1",
      hostname: "hal9000",
    });
    const hosts = useHostsStore();
    const first = await hosts.connect("http://hal9000:7680", null, null);
    // Same physical box, reached by IP this time — one row, not two.
    const second = await hosts.connect("http://192.168.1.114:7680", null, null);
    expect(second.id).toBe(first.id);
    expect(hosts.all.filter((h) => h.id !== "local").map((h) => h.id)).toEqual(["hal9000-7680"]);
  });

  it("connect() adopts a provided key onto the existing slug when deduping by instance id", async () => {
    testRemoteHost.mockResolvedValue({
      ok: true,
      version: null,
      error: null,
      instanceId: "uuid-1",
      hostname: "hal9000",
    });
    const hosts = useHostsStore();
    await hosts.connect("http://hal9000:7680", null, null);
    secretGet.mockResolvedValue(null); // survivor slug has no stored key yet
    await hosts.connect("http://192.168.1.114:7680", "ip-key", null);
    expect(secretSet).toHaveBeenCalledWith("remote-api-key.hal9000-7680", "ip-key");
  });

  it("connect() refuses a URL whose slug collides with the built-in engine id", async () => {
    const hosts = useHostsStore();
    await expect(hosts.connect("https://local", null, null)).rejects.toThrow(/built-in engine/);
    expect(testRemoteHost).not.toHaveBeenCalled();
  });

  it("refresh() stamps instance id + hostname and persists the id onto the saved host", async () => {
    installSettings(settings({ savedHosts: [hal], connectedHostIds: [hal.id] }));
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    apiJsonTo.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve({
        queue_depth: 0,
        queue_capacity: 8,
        version: "0.18.0",
        instance_id: target.baseUrl.includes("hal9000") ? "uuid-hal" : "uuid-local",
        hostname: target.baseUrl.includes("hal9000") ? "hal9000" : "this-mac",
      }),
    );
    const hosts = useHostsStore();
    await hosts.init();
    await hosts.refresh();
    expect(hosts.telemetry[hal.id]?.instanceId).toBe("uuid-hal");
    expect(hosts.telemetry[hal.id]?.hostname).toBe("hal9000");
    const persisted = appSettingsSet.mock.lastCall?.[0] as ReturnType<typeof settings>;
    expect(persisted.savedHosts.find((h: SavedHost) => h.id === hal.id)?.instanceId).toBe(
      "uuid-hal",
    );
  });

  it("refresh() collapses two saved slugs that report the same instance id", async () => {
    const ip: SavedHost = {
      id: "192-168-1-114-7680",
      name: null,
      url: "http://192.168.1.114:7680",
      lastUsedMs: 1,
    };
    installSettings(
      settings({
        savedHosts: [hal, ip],
        connectedHostIds: [hal.id, ip.id],
        generateTargetHost: ip.id,
      }),
    );
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    // The two remote slugs answer with the SAME instance id — one physical box;
    // the local engine keeps its own id.
    apiJsonTo.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve({
        queue_depth: 0,
        queue_capacity: 8,
        version: "0.18.0",
        instance_id: target.baseUrl.includes("127.0.0.1") ? "uuid-local" : "uuid-shared",
        hostname: "hal9000",
      }),
    );
    const hosts = useHostsStore();
    await hosts.init();
    await hosts.refresh();
    const persisted = appSettingsSet.mock.lastCall?.[0] as ReturnType<typeof settings>;
    // Survivor is hal (more recent lastUsedMs); the IP slug is dropped and the
    // sticky target re-homed onto the survivor.
    expect(persisted.savedHosts.map((h: SavedHost) => h.id)).toEqual([hal.id]);
    expect(persisted.connectedHostIds).toEqual([hal.id]);
    expect(persisted.generateTargetHost).toBe(hal.id);
  });
});
