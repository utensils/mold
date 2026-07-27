import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import type { AppSettings, SavedHost } from "../lib/ipc";

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
const listDevices = vi.fn();
const listQueue = vi.fn();
vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
}));
vi.mock("@studio/api/devices", () => ({
  listDevices: (...a: unknown[]) => listDevices(...a),
}));
vi.mock("@studio/api/queuePlan", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/queuePlan")>()),
  listQueue: (...a: unknown[]) => listQueue(...a),
}));

import { useAppPrefsStore } from "./appPrefs";
import { useConnectionStore } from "./connection";
import { useDownloadsStore } from "./downloads";
import { useHostModelsStore } from "./hostModels";
import { useHostsStore } from "./hosts";
import { useToastStore } from "./toasts";
import type { ModelEntry, ServerCapabilities } from "../lib/api/types";
import type { DeviceInfo } from "@studio/api/devices";

function device(ordinal: number, overrides: Partial<DeviceInfo> = {}): DeviceInfo {
  return {
    id: `cuda:${ordinal}`,
    backend: "cuda",
    ordinal,
    device_kind: "full_gpu",
    nvml_uuid: `GPU-${ordinal}`,
    physical_uuid: `GPU-${ordinal}`,
    mig_uuid: null,
    mig_parent_uuid: null,
    mig_profile: null,
    name: `GPU ${ordinal}`,
    pci_bus_id: null,
    compute_capability: "8.6",
    memory: {
      total_bytes: 24 * 1024 ** 3,
      used_bytes: 0,
      mold_used_bytes: 0,
      other_used_bytes: 0,
    },
    telemetry: {
      utilization_percent: 0,
      temperature_c: 30,
      power_w: 20,
    },
    desired_enabled: true,
    admin_state: "enabled",
    health: "healthy",
    activity: "idle",
    schedulable: true,
    unschedulable_reason: null,
    loaded_models: [],
    active_work_id: null,
    planned_work_ids: [],
    ...overrides,
  };
}

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

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
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
    sidebarCollapsed: false,
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
  listDevices.mockRejectedValue(new Error("legacy server"));
  listQueue.mockRejectedValue(new Error("legacy server"));
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

  it("tries every saved host at boot even when the legacy reconnect set is empty", async () => {
    installSettings(settings({ savedHosts: [hal], connectedHostIds: [] }));
    secretGet.mockResolvedValue("host-key");
    testRemoteHost.mockResolvedValue({ ok: true, version: "0.17.0", error: null });

    const hosts = useHostsStore();
    await hosts.init();

    expect(testRemoteHost).toHaveBeenCalledWith(hal.url, "host-key");
    expect(hosts.all.find((host) => host.id === hal.id)).toMatchObject({ status: "ready" });
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

  it("refresh() caches expansion capability for each ready host using its authenticated target", async () => {
    installSettings(settings({ savedHosts: [hal], connectedHostIds: [hal.id] }));
    secretGet.mockResolvedValue("host-key");
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    apiJsonTo.mockImplementation(
      (target: { baseUrl: string; apiKey: string | null }, path: string) => {
        if (path === "/api/capabilities") {
          return Promise.resolve({
            gallery: { can_delete: true },
            expand: target.baseUrl.includes("hal9000")
              ? { configured: true, model_present: null, backend: "api" }
              : { configured: true, model_present: true, backend: "local" },
          } satisfies ServerCapabilities);
        }
        return Promise.resolve({ queue_depth: 0, queue_capacity: 8, version: null });
      },
    );

    const hosts = useHostsStore();
    await hosts.init();
    await hosts.refresh();

    expect(hosts.capabilities.local?.expand).toEqual({
      configured: true,
      model_present: true,
      backend: "local",
    });
    expect(hosts.capabilities[hal.id]?.expand).toEqual({
      configured: true,
      model_present: null,
      backend: "api",
    });
    expect(apiJsonTo).toHaveBeenCalledWith(
      { baseUrl: hal.url, apiKey: "host-key" },
      "/api/capabilities",
    );
  });

  it("keeps an older host's missing expansion capability as unknown and refreshes it later", async () => {
    apiJsonTo.mockImplementation((_target: unknown, path: string) =>
      Promise.resolve(
        path === "/api/capabilities"
          ? { gallery: { can_delete: true } }
          : { queue_depth: 0, queue_capacity: 8, version: null },
      ),
    );
    const hosts = useHostsStore();
    await hosts.refresh();
    expect(hosts.capabilities.local?.expand).toBeUndefined();

    apiJsonTo.mockImplementation((_target: unknown, path: string) =>
      Promise.resolve(
        path === "/api/capabilities"
          ? {
              gallery: { can_delete: true },
              expand: { configured: true, model_present: false, backend: "local" },
            }
          : { queue_depth: 0, queue_capacity: 8, version: null },
      ),
    );
    await hosts.refresh();
    expect(hosts.capabilities.local?.expand?.model_present).toBe(false);
  });

  it("disconnect() removes the host capability cache", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    hosts.capabilities[hal.id] = {
      gallery: { can_delete: true },
      expand: { configured: true, model_present: null, backend: "api" },
    };

    await hosts.disconnect(hal.id);

    expect(hosts.capabilities[hal.id]).toBeUndefined();
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

  it('resolveRoute("capable") considers every healthy GPU on a host', async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    hosts.telemetry["local"] = {
      queueDepth: 0,
      queueCapacity: 8,
      version: null,
      gpuInfo: {
        name: "NVIDIA RTX A2000",
        vram_total_mb: 12288,
        vram_used_mb: 0,
        backend: "cuda",
      },
      gpuWorkers: [
        {
          ordinal: 0,
          name: "NVIDIA RTX A2000",
          vram_total_bytes: 12 * 1024 ** 3,
          vram_used_bytes: 0,
          state: "idle",
        },
        {
          ordinal: 1,
          name: "NVIDIA B200",
          vram_total_bytes: 80 * 1024 ** 3,
          vram_used_bytes: 0,
          state: "idle",
        },
      ],
    };
    hosts.telemetry["hal9000-7680"] = {
      queueDepth: 0,
      queueCapacity: 8,
      version: null,
      gpuInfo: {
        name: "NVIDIA RTX 4090",
        vram_total_mb: 24564,
        vram_used_mb: 0,
        backend: "cuda",
      },
    };

    expect(hosts.resolveRoute("capable")?.hostId).toBe("local");
  });

  it("uses /api/devices and does not route excluded current-server GPUs", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("http://hal9000:7680", null, null);
    hosts.telemetry.local = {
      queueDepth: 0,
      queueCapacity: 200,
      version: "0.20.0",
      gpuInfo: {
        backend: "cuda",
        name: "NVIDIA B200",
        vram_total_mb: 196608,
        vram_used_mb: 0,
      },
      gpuWorkers: [
        {
          ordinal: 0,
          name: "NVIDIA B200",
          vram_total_bytes: 192 * 1024 ** 3,
          vram_used_bytes: 0,
          state: "idle",
        },
      ],
      devices: [
        device(0, {
          admin_state: "startup_excluded",
          desired_enabled: false,
          schedulable: false,
          unschedulable_reason: "excluded by MOLD_GPUS",
        }),
      ],
    };
    hosts.telemetry["hal9000-7680"] = {
      queueDepth: 1,
      queueCapacity: 200,
      version: "0.20.0",
      gpuInfo: {
        backend: "cuda",
        name: "NVIDIA RTX 4090",
        vram_total_mb: 24576,
        vram_used_mb: 0,
      },
      gpuWorkers: null,
    };

    expect(hosts.resolveRoute(null)?.hostId).toBe("hal9000-7680");
    expect(hosts.resolveRoute("local")).toBeNull();
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

  it("keeps the last-known hostname label across a failed poll", async () => {
    installSettings(settings({ savedHosts: [{ ...hal, name: null }], connectedHostIds: [hal.id] }));
    testRemoteHost.mockResolvedValue({
      ok: true,
      version: null,
      error: null,
      instanceId: null,
      hostname: "hal9000",
    });
    const hosts = useHostsStore();
    await hosts.init();
    // The boot probe already reported the hostname — the row must not sit on
    // the raw URL until the first poll.
    expect(hosts.all.find((h) => h.id === hal.id)?.label).toBe("hal9000");
    apiJsonTo.mockResolvedValue({
      queue_depth: 0,
      queue_capacity: 8,
      version: null,
      hostname: "hal9000",
    });
    await hosts.refresh();
    expect(hosts.all.find((h) => h.id === hal.id)?.label).toBe("hal9000");
    // A wifi blip fails one poll: telemetry is dropped, but the label must
    // not flip back to the raw URL until the next successful poll.
    apiJsonTo.mockImplementation((target: { baseUrl: string }) =>
      target.baseUrl.includes("hal9000")
        ? Promise.reject(new Error("timeout"))
        : Promise.resolve({ queue_depth: 0, queue_capacity: 8, version: null }),
    );
    await hosts.refresh();
    const row = hosts.all.find((h) => h.id === hal.id);
    expect(row?.status).toBe("error");
    expect(row?.label).toBe("hal9000");
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

  it("refresh() pulls queue telemetry from every host", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    apiJsonTo.mockResolvedValue({
      queue_depth: 2,
      queue_capacity: 8,
      version: "0.16.0",
      models_loaded: ["flux2-klein:q4"],
      gpu_info: { name: "NVIDIA GeForce RTX 4090", vram_total_mb: 24564, vram_used_mb: 8192 },
      gpus: [
        {
          ordinal: 0,
          name: "NVIDIA GeForce RTX 4090",
          vram_total_bytes: 24564 * 1024 ** 2,
          vram_used_bytes: 8192 * 1024 ** 2,
          state: "idle",
        },
      ],
    });
    listDevices.mockResolvedValue({
      plan_version: 1,
      devices: [device(0)],
    });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    await hosts.refresh();
    expect(hosts.telemetry["local"]?.queueDepth).toBe(2);
    expect(hosts.telemetry["hal9000-7680"]?.queueDepth).toBe(2);
    // The status-bar fallback data rides the same poll.
    expect(hosts.telemetry["hal9000-7680"]?.modelsLoaded).toEqual(["flux2-klein:q4"]);
    expect(hosts.telemetry["hal9000-7680"]?.gpuInfo?.vram_total_mb).toBe(24564);
    expect(hosts.telemetry["hal9000-7680"]?.gpuWorkers).toHaveLength(1);
    expect(hosts.telemetry["hal9000-7680"]?.devices).toHaveLength(1);
  });

  it("ignores an older same-host telemetry refresh after a newer refresh settles", async () => {
    const older = deferred<Record<string, unknown>>();
    const newer = deferred<Record<string, unknown>>();
    let statusCall = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return statusCall++ === 0 ? older.promise : newer.promise;
      if (path === "/api/capabilities") return Promise.resolve({});
      return Promise.reject(new Error(`unexpected path ${path}`));
    });
    listDevices.mockResolvedValue({
      plan_version: 1,
      devices: [device(0)],
    });
    listQueue.mockResolvedValue({ entries: [], plan: null });
    const hosts = useHostsStore();

    const first = hosts.refresh();
    const second = hosts.refresh();
    newer.resolve({
      queue_depth: 9,
      queue_capacity: 16,
      version: "newer",
    });
    await second;
    older.resolve({
      queue_depth: 1,
      queue_capacity: 16,
      version: "older",
    });
    await first;

    expect(hosts.telemetry.local?.queueDepth).toBe(9);
    expect(hosts.telemetry.local?.version).toBe("newer");
  });

  it("always refreshes predicted completion so Auto can refine equal-depth hosts", async () => {
    const now = Date.now();
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    apiJsonTo.mockResolvedValue({
      queue_depth: 2,
      queue_capacity: 8,
      version: "0.20.2",
    });
    listQueue.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve({
        entries: [],
        plan: {
          plan_version: 1,
          state_version: 1,
          optimizer_state: "optimized",
          dirty_since_unix_ms: null,
          next_replan_at_unix_ms: null,
          work_items: [
            {
              work_id: "queued",
              parent_id: "queued",
              work_kind: "generation",
              priority_class: "user",
              queue_rank: 0,
              bypass_count: 0,
              estimate_confidence: "high",
              estimated_finish_unix_ms: target.baseUrl.includes("hal9000")
                ? now + 10_000
                : now + 20_000,
            },
          ],
        },
      }),
    );
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);

    await hosts.refresh();

    expect(listQueue).toHaveBeenCalledWith({
      baseUrl: "http://hal9000:7680",
      apiKey: null,
    });
    expect(hosts.resolveRoute(null)?.hostId).toBe("hal9000-7680");
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

  it("connect() revives a dead instance-id twin with the freshly validated address", async () => {
    // Boot-reconnect lists the saved host as an errored extra (its old DHCP
    // address no longer answers) carrying the persisted instance id.
    installSettings(
      settings({
        savedHosts: [{ ...hal, name: null, instanceId: "uuid-x" }],
        connectedHostIds: [hal.id],
      }),
    );
    testRemoteHost.mockResolvedValueOnce({ ok: false, version: null, error: "down" });
    apiJsonTo.mockImplementation((target: { baseUrl: string }) =>
      target.baseUrl.includes("hal9000")
        ? Promise.reject(new Error("down"))
        : Promise.resolve({ queue_depth: 0, queue_capacity: 8, version: null }),
    );
    const hosts = useHostsStore();
    await hosts.init();
    expect(hosts.all.find((h) => h.id === hal.id)?.status).toBe("error");

    // The user adds the box's NEW address; the probe proves it's the same box.
    testRemoteHost.mockResolvedValue({
      ok: true,
      version: null,
      error: null,
      instanceId: "uuid-x",
      hostname: "hal9000",
    });
    apiJsonTo.mockResolvedValue({ queue_depth: 0, queue_capacity: 8, version: null });
    const view = await hosts.connect("http://192.168.1.99:7680", null, null);

    // The twin's surviving slug adopts the validated URL instead of keeping
    // the dead one forever.
    expect(view.id).toBe(hal.id);
    expect(view.status).toBe("ready");
    expect(view.baseUrl).toBe("http://192.168.1.99:7680");
    const persisted = appSettingsSet.mock.lastCall?.[0] as ReturnType<typeof settings>;
    expect(persisted.savedHosts.find((h: SavedHost) => h.id === hal.id)?.url).toBe(
      "http://192.168.1.99:7680",
    );
  });

  it("connect() adopts a newly typed key onto its errored twin (key rotation)", async () => {
    // The server rotated its API key; the stored key fails the boot probe.
    installSettings(
      settings({ savedHosts: [{ ...hal, instanceId: "uuid-x" }], connectedHostIds: [hal.id] }),
    );
    secretGet.mockResolvedValue("stale-key");
    testRemoteHost.mockResolvedValueOnce({ ok: false, version: null, error: "401" });
    apiJsonTo.mockImplementation((target: { baseUrl: string }) =>
      target.baseUrl.includes("hal9000")
        ? Promise.reject(new Error("401"))
        : Promise.resolve({ queue_depth: 0, queue_capacity: 8, version: null }),
    );
    const hosts = useHostsStore();
    await hosts.init();
    expect(hosts.all.find((h) => h.id === hal.id)?.status).toBe("error");

    // Re-adding the host with the NEW key must persist it (a stale stored key
    // must not block adoption) and revive the live row in place.
    testRemoteHost.mockResolvedValue({
      ok: true,
      version: null,
      error: null,
      instanceId: "uuid-x",
      hostname: "hal9000",
    });
    apiJsonTo.mockResolvedValue({ queue_depth: 0, queue_capacity: 8, version: null });
    await hosts.connect(hal.url, "new-key", null);
    expect(secretSet).toHaveBeenCalledWith("remote-api-key.hal9000-7680", "new-key");
    const live = hosts.extras.find((h) => h.id === hal.id)!;
    expect(live.apiKey).toBe("new-key");
    expect(live.status).toBe("ready");
    expect(live.error).toBeNull();
  });

  it("connect() keeps two servers with the same instance id but different hostnames separate", async () => {
    // Two RunPod pods sharing one network volume (shared MOLD_HOME) report
    // the SAME instance uuid — the reported hostname tells them apart.
    testRemoteHost.mockImplementation((url: string) =>
      Promise.resolve({
        ok: true,
        version: null,
        error: null,
        instanceId: "uuid-shared",
        hostname: url.includes("pod-a") ? "pod-a" : "pod-b",
      }),
    );
    const hosts = useHostsStore();
    await hosts.connect("http://pod-a:7680", null, null);
    const second = await hosts.connect("http://pod-b:7680", null, null);
    expect(second.id).toBe("pod-b-7680");
    expect(hosts.all.filter((h) => h.kind === "remote").map((h) => h.id)).toEqual([
      "pod-a-7680",
      "pod-b-7680",
    ]);
  });

  it("reconcile does not clobber a settings write that lands during its secret IPC", async () => {
    const ip: SavedHost = {
      id: "192-168-1-114-7680",
      name: null,
      url: "http://192.168.1.114:7680",
      lastUsedMs: 1,
    };
    installSettings(settings({ savedHosts: [hal, ip], connectedHostIds: [] }));
    let releaseSecret!: () => void;
    const gate = new Promise<string | null>((resolve) => {
      releaseSecret = () => resolve(null);
    });
    secretGet.mockImplementationOnce(() => gate);
    const hosts = useHostsStore();
    const reconcile = hosts.reconcileSavedInstanceIds(
      new Map([
        [hal.id, "uuid-shared"],
        [ip.id, "uuid-shared"],
      ]),
    );
    // Reconcile is parked on the per-host secret round-trip...
    await vi.waitFor(() => expect(secretGet).toHaveBeenCalled());
    // ...while a different writer (a theme toggle) lands in between.
    await appSettingsSet({ ...(await appSettingsGet()), theme: "dark" });
    releaseSecret();
    await reconcile;
    const final = (await appSettingsGet()) as ReturnType<typeof settings>;
    // The merge landed AND the interleaved write survived.
    expect(final.savedHosts.map((h: SavedHost) => h.id)).toEqual([hal.id]);
    expect(final.theme).toBe("dark");
  });

  it("serializes this store's settings writers so neither clobbers the other", async () => {
    installSettings(settings({ savedHosts: [hal, studio], connectedHostIds: [hal.id, studio.id] }));
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.init();
    // Two concurrent read-modify-writes: unserialized, the rename's stale
    // snapshot would resurrect the just-disconnected host.
    await Promise.all([hosts.disconnect(studio.id), hosts.rename(hal.id, "render box")]);
    const final = (await appSettingsGet()) as ReturnType<typeof settings>;
    expect(final.connectedHostIds).not.toContain(studio.id);
    expect(final.savedHosts.find((h: SavedHost) => h.id === hal.id)?.name).toBe("render box");
  });

  it("reconcile keeps the preferred live row when saved aliases both reconnect", async () => {
    // Every saved address is attempted at boot. Reconcile still collapses
    // aliases to the more recently used slug once instance identity arrives.
    const ip: SavedHost = {
      id: "192-168-1-114-7680",
      name: null,
      url: "http://192.168.1.114:7680",
      lastUsedMs: 1,
    };
    installSettings(
      settings({
        savedHosts: [{ ...hal, lastUsedMs: 5, instanceId: "uuid-shared" }, ip],
        connectedHostIds: [ip.id],
      }),
    );
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    apiJsonTo.mockImplementation((target: { baseUrl: string }) => {
      const remote = !target.baseUrl.includes("127.0.0.1");
      return Promise.resolve({
        queue_depth: 0,
        queue_capacity: 8,
        version: null,
        instance_id: remote ? "uuid-shared" : "uuid-local",
        hostname: remote ? "hal9000" : "this-mac",
      });
    });
    const hosts = useHostsStore();
    await hosts.init();
    await hosts.refresh();
    const row = hosts.all.find((h) => h.id === hal.id);
    expect(row).toMatchObject({ status: "ready", baseUrl: hal.url });
    expect(hosts.all.some((h) => h.id === ip.id)).toBe(false);
    const persisted = appSettingsSet.mock.lastCall?.[0] as ReturnType<typeof settings>;
    expect(persisted.connectedHostIds).toEqual([hal.id]);
  });

  it("reconcile re-homes in-memory names and the appPrefs snapshot", async () => {
    const ip: SavedHost = {
      id: "192-168-1-114-7680",
      name: "Render box",
      url: "http://192.168.1.114:7680",
      lastUsedMs: 1,
    };
    installSettings(
      settings({
        savedHosts: [{ ...hal, name: null, lastUsedMs: 5 }, ip],
        connectedHostIds: [hal.id, ip.id],
        generateTargetHost: ip.id,
      }),
    );
    const prefs = useAppPrefsStore();
    prefs.settings = settings({ generateTargetHost: ip.id }) as unknown as AppSettings;
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    apiJsonTo.mockImplementation((target: { baseUrl: string }) =>
      Promise.resolve({
        queue_depth: 0,
        queue_capacity: 8,
        version: null,
        instance_id: target.baseUrl.includes("127.0.0.1") ? "uuid-local" : "uuid-shared",
        hostname: target.baseUrl.includes("127.0.0.1") ? "this-mac" : "hal9000",
      }),
    );
    const hosts = useHostsStore();
    await hosts.init();
    await hosts.refresh();
    // The sticky target is re-homed in the live prefs snapshot, not just on
    // disk — resolveRoute readers must keep honoring the pin immediately.
    expect(prefs.settings?.generateTargetHost).toBe(hal.id);
    // The loser's user-assigned name carries onto the surviving row now, not
    // only after a relaunch re-reads savedHosts.
    expect(hosts.all.find((h) => h.id === hal.id)?.label).toBe("Render box");
    expect(hosts.all.some((h) => h.id === ip.id)).toBe(false);
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
