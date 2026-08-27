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
const previewGenerationPlacement = vi.fn();
const previewChainPlacement = vi.fn();
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
vi.mock("@studio/api/generationPlacement", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/generationPlacement")>()),
  previewGenerationPlacement: (...a: unknown[]) => previewGenerationPlacement(...a),
  previewChainPlacement: (...a: unknown[]) => previewChainPlacement(...a),
}));

import { useAppPrefsStore } from "./appPrefs";
import { useConnectionStore } from "./connection";
import { useDownloadsStore } from "./downloads";
import { useHostModelsStore } from "./hostModels";
import { useHostsStore, type ExtraHost } from "./hosts";
import { useToastStore } from "./toasts";
import type {
  AutoChainRequest,
  GenerateRequest,
  ModelEntry,
  ServerCapabilities,
} from "../lib/api/types";
import type { DeviceInfo } from "@studio/api/devices";
import { ApiError } from "../lib/api/client";

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

function plannedPlacement(deviceId = "cuda:0") {
  return {
    version: 1,
    authoritative: true,
    state_version: 1,
    plan_version: 1,
    outcome: "planned",
    candidate: {
      device_id: deviceId,
      execution_fingerprint: "test",
      predicted_start_after_ms: 0,
      predicted_completion_after_ms: 100,
      setup_ms: 0,
      setup_kind: "warm",
      estimate_confidence: "high",
    },
  };
}

const placementRequest = {
  prompt: "",
  model: "flux-dev:q4",
  width: 512,
  height: 512,
  steps: 4,
  guidance: 3.5,
  batch_size: 1,
} as GenerateRequest;

/** A print is admitted durably and never previewed; a SEQUENCE is planned
 * before it is created, so the placement machinery below rides one. */
const sequenceRequest = {
  ...placementRequest,
  total_frames: 241,
  clip_frames: 97,
  motion_tail_frames: 17,
} as unknown as GenerateRequest;

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
  previewGenerationPlacement.mockResolvedValue(plannedPlacement());
  previewChainPlacement.mockResolvedValue(plannedPlacement());
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

  it("keeps explicitly disconnected saved hosts out of boot reconnects", async () => {
    installSettings(settings({ savedHosts: [hal], connectedHostIds: [] }));
    secretGet.mockResolvedValue("host-key");
    testRemoteHost.mockResolvedValue({ ok: true, version: "0.17.0", error: null });

    const hosts = useHostsStore();
    await hosts.init();

    expect(testRemoteHost).not.toHaveBeenCalledWith(hal.url, "host-key");
    expect(hosts.all.find((host) => host.id === hal.id)).toBeUndefined();
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

    // The built-in engine is always the primary; every connected host follows.
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

  it("keeps an unreachable remembered host reconnecting instead of falsely offline", async () => {
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
    expect(hosts.all.find((h) => h.id === hal.id)?.status).toBe("connecting");
    expect(useToastStore().items.some((t) => t.message.includes("hal9000"))).toBe(false);
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
    useHostModelsStore().byHost[hal.id] = {
      entries: [installedModel("old-model")],
      fetchedAt: Date.now(),
      error: null,
    };

    await hosts.disconnect(hal.id);

    expect(hosts.capabilities[hal.id]).toBeUndefined();
    expect(useHostModelsStore().byHost[hal.id]).toBeUndefined();
  });

  it("resolveRoute(null) auto-routes to the least busy ready host", async () => {
    testRemoteHost.mockResolvedValue({ ok: true, version: null, error: null });
    const hosts = useHostsStore();
    await hosts.connect("hal9000", null, null);
    hosts.telemetry["local"] = { queueDepth: 4, queueCapacity: 8, version: null };
    hosts.telemetry["hal9000-7680"] = { queueDepth: 0, queueCapacity: 8, version: null };
    expect(hosts.resolveRoute(null)?.hostId).toBe("hal9000-7680");
  });

  it("freezes the exact durable-media capability into the selected host route", () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "hal-instance",
    });
    hosts.telemetry[hal.id] = { queueDepth: 0, queueCapacity: 8, version: "0.25.0" };
    hosts.capabilities[hal.id] = {
      gallery: { can_delete: true },
      queue: {
        heterogeneous_batch_max_outputs: 64,
      },
      durable_media: {
        protocol_version: 2,
        encrypted_at_rest: true,
        generate_request_media: true,
        identity: true,
        h3_references: false,
        private_h3: false,
      },
    };

    expect(hosts.resolveRoute(hal.id)?.durableMedia).toEqual(
      hosts.capabilities[hal.id]?.durable_media,
    );
    expect(hosts.resolveRoute(hal.id)?.heterogeneousBatchMaxOutputs).toBe(64);
  });

  it("projects the pinned target and the placement route through one mapping", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "hal-instance",
    });
    hosts.telemetry[hal.id] = { queueDepth: 0, queueCapacity: 8, version: "0.25.0" };
    hosts.capabilities[hal.id] = {
      gallery: { can_delete: true },
      queue: { heterogeneous_batch_max_outputs: 64 },
      durable_media: {
        protocol_version: 2,
        encrypted_at_rest: true,
        generate_request_media: true,
        identity: true,
        h3_references: true,
        private_h3: true,
      },
    } as ServerCapabilities;

    // The pinned-target path and the placement path used to build this route
    // twice; one mapping is what keeps them from disagreeing about what a
    // machine can carry.
    const pinned = hosts.resolveRoute(hal.id);
    const placed = await hosts.resolveFeasibleRoute(hal.id, placementRequest);
    expect(placed).toEqual(pinned);
  });

  it("keeps a machine that read-refuses the durable contract out of Auto", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "hal-instance",
    });
    // This device is emptier but answered /api/capabilities with no durable
    // queue: it would refuse at submit, so Auto must rank hal instead.
    hosts.telemetry.local = { queueDepth: 0, queueCapacity: 8, version: "0.25.0" };
    hosts.telemetry[hal.id] = { queueDepth: 5, queueCapacity: 8, version: "0.25.0" };
    hosts.capabilities.local = { gallery: { can_delete: true } };
    hosts.capabilities[hal.id] = {
      gallery: { can_delete: true },
      queue: { heterogeneous_batch_max_outputs: 64 },
    };

    await expect(hosts.resolveFeasible(null, placementRequest)).resolves.toMatchObject({
      kind: "route",
      route: { hostId: hal.id },
      preview: null,
    });
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
  });

  it("skips placement preview for canonical v2 pinned and automatic routing", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "hal-instance",
    });
    hosts.telemetry.local = { queueDepth: 4, queueCapacity: 8, version: "0.25.0" };
    hosts.telemetry[hal.id] = { queueDepth: 0, queueCapacity: 8, version: "0.25.0" };
    const canonical: ServerCapabilities = {
      gallery: { can_delete: true },
      queue: {
        heterogeneous_batch_max_outputs: 64,
      },
    };
    hosts.capabilities.local = canonical;
    hosts.capabilities[hal.id] = canonical;

    await expect(hosts.resolveFeasible("local", placementRequest)).resolves.toMatchObject({
      kind: "route",
      route: { hostId: "local" },
      preview: null,
    });
    await expect(hosts.resolveFeasible(null, placementRequest)).resolves.toMatchObject({
      kind: "route",
      route: { hostId: hal.id },
      preview: null,
    });
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
  });

  it("returns a missing-model recovery for a canonical pinned host with a known absence", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "hal-instance",
    });
    hosts.telemetry[hal.id] = { queueDepth: 0, queueCapacity: 8, version: "0.25.0" };
    hosts.capabilities[hal.id] = {
      gallery: { can_delete: true },
      queue: {
        heterogeneous_batch_max_outputs: 64,
      },
    };
    useHostModelsStore().byHost[hal.id] = { entries: [], fetchedAt: Date.now(), error: null };

    await expect(hosts.resolveFeasible(hal.id, placementRequest)).resolves.toMatchObject({
      kind: "infeasible",
      perHost: [
        {
          hostId: hal.id,
          missingModel: { model: placementRequest.model },
        },
      ],
    });
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
  });

  it("allows Auto across differing profiles on the same Mold major version", () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "hal-instance",
    });
    hosts.telemetry.local = { queueDepth: 1, queueCapacity: 8, version: "0.23.1" };
    hosts.telemetry[hal.id] = { queueDepth: 0, queueCapacity: 8, version: "0.23.0" };
    const hostModels = useHostModelsStore();
    hostModels.byHost.local = {
      entries: [
        {
          ...installedModel(placementRequest.model),
          generation_profile: { profile_hash: "local-profile" } as never,
        },
      ],
      fetchedAt: Date.now(),
      error: null,
    };
    hostModels.byHost[hal.id] = {
      entries: [
        {
          ...installedModel(placementRequest.model),
          generation_profile: { profile_hash: "remote-profile" } as never,
        },
      ],
      fetchedAt: Date.now(),
      error: null,
    };

    expect(hosts.resolveRoute(null, placementRequest.model)?.hostId).toBe(hal.id);
  });

  it("requires an explicit machine when model owners use different Mold major versions", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "hal-instance",
    });
    hosts.telemetry.local = { queueDepth: 0, queueCapacity: 8, version: "0.23.1" };
    hosts.telemetry[hal.id] = { queueDepth: 0, queueCapacity: 8, version: "1.0.0" };
    const hostModels = useHostModelsStore();
    hostModels.byHost.local = {
      entries: [
        {
          ...installedModel(placementRequest.model),
          generation_profile: { profile_hash: "local-profile" } as never,
        },
      ],
      fetchedAt: Date.now(),
      error: null,
    };
    hostModels.byHost[hal.id] = {
      entries: [
        {
          ...installedModel(placementRequest.model),
          generation_profile: { profile_hash: "remote-profile" } as never,
        },
      ],
      fetchedAt: Date.now(),
      error: null,
    };

    expect(hosts.resolveRoute(null, placementRequest.model)).toBeNull();
    await expect(hosts.resolveFeasible(null, placementRequest)).resolves.toEqual({
      kind: "profile_mismatch",
      perHost: [
        expect.objectContaining({ hostId: "local", profileHash: "local-profile" }),
        expect.objectContaining({ hostId: hal.id, profileHash: "remote-profile" }),
      ],
    });
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
    expect(hosts.resolveRoute(hal.id, placementRequest.model)?.hostId).toBe(hal.id);
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

  it("refuses an advertised restricted request before placement", async () => {
    const hosts = useHostsStore();
    const restrictedModel = "hf:MiniMaxAI/MiniMaxH3";
    hosts.capabilities.local = {
      gallery: { can_delete: true },
      model_access: {
        restrictions: [
          {
            code: "minimax_h3_authorization_required",
            family: "minimax-h3",
            message: "MiniMax H3 is not activated.",
            license_url: "https://example.test/license",
            authorization_url: "https://example.test/authorize",
          },
        ],
      },
    };
    useHostModelsStore().byHost.local = {
      entries: [{ ...installedModel(restrictedModel), family: "minimax-h3" }],
      fetchedAt: Date.now(),
      error: null,
    };
    const request = { ...placementRequest, model: restrictedModel };

    expect(hosts.resolveRoute("local", restrictedModel)).toBeNull();
    await expect(hosts.resolveFeasible("local", request)).resolves.toEqual({
      kind: "infeasible",
      perHost: [
        expect.objectContaining({
          hostId: "local",
          reason: "MiniMax H3 is not activated.",
        }),
      ],
    });
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
  });

  it("stops placement when the live generation selection changes", async () => {
    const prefs = useAppPrefsStore();
    prefs.settings = settings({
      generateTargetHost: "local",
    }) as unknown as AppSettings;
    const gate = deferred<ReturnType<typeof plannedPlacement>>();
    previewChainPlacement.mockReturnValueOnce(gate.promise);
    const hosts = useHostsStore();

    const pending = hosts.resolveFeasibleRoute("local", sequenceRequest);
    await vi.waitFor(() => expect(previewChainPlacement).toHaveBeenCalledTimes(1));
    prefs.settings = settings({
      generateTargetHost: "hal9000-7680",
    }) as unknown as AppSettings;
    gate.resolve(plannedPlacement());

    await expect(pending).resolves.toBeNull();
    expect(previewChainPlacement).toHaveBeenCalledTimes(1);
  });

  it.each([
    [
      "URL",
      (host: ExtraHost) => {
        host.url = "http://replacement:7680";
      },
    ],
    [
      "API key",
      (host: ExtraHost) => {
        host.apiKey = "replacement-key";
      },
    ],
    [
      "instance identity",
      (host: ExtraHost) => {
        host.instanceId = "replacement-instance";
      },
    ],
  ])(
    "stops placement without probing a replacement when the host %s changes",
    async (_field, mutate) => {
      const prefs = useAppPrefsStore();
      prefs.settings = settings({
        generateTargetHost: hal.id,
      }) as unknown as AppSettings;
      const hosts = useHostsStore();
      const extra = {
        id: hal.id,
        label: "hal9000",
        url: hal.url,
        apiKey: "host-key",
        status: "ready" as const,
        error: null,
        instanceId: "instance-a",
      };
      hosts.extras.push(extra);
      const gate = deferred<ReturnType<typeof plannedPlacement>>();
      previewChainPlacement.mockReturnValueOnce(gate.promise);

      const pending = hosts.resolveFeasibleRoute(hal.id, sequenceRequest);
      await vi.waitFor(() => expect(previewChainPlacement).toHaveBeenCalledTimes(1));
      mutate(hosts.extras.find((host) => host.id === hal.id)!);
      gate.resolve(plannedPlacement());

      await expect(pending).resolves.toBeNull();
      expect(previewChainPlacement).toHaveBeenCalledTimes(1);
    },
  );

  it("reports a transient failure when availability changes during both attempts", async () => {
    const hosts = useHostsStore();
    const first = deferred<ReturnType<typeof plannedPlacement>>();
    const second = deferred<ReturnType<typeof plannedPlacement>>();
    previewChainPlacement.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);

    const pending = hosts.resolveFeasible("local", sequenceRequest);
    await vi.waitFor(() => expect(previewChainPlacement).toHaveBeenCalledTimes(1));
    useConnectionStore().status = "starting";
    first.resolve(plannedPlacement());
    await vi.waitFor(() => expect(previewChainPlacement).toHaveBeenCalledTimes(2));
    useConnectionStore().status = "ready";
    second.resolve(plannedPlacement());

    await expect(pending).resolves.toEqual({
      kind: "transient",
      perHost: [
        expect.objectContaining({
          hostId: "local",
          error: "routing state changed while placement was being checked",
        }),
      ],
    });
  });

  it("never reroutes an explicitly selected host after authoritative infeasibility", async () => {
    const prefs = useAppPrefsStore();
    prefs.settings = settings({
      generateTargetHost: hal.id,
    }) as unknown as AppSettings;
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "instance-a",
    });
    previewChainPlacement.mockResolvedValueOnce({
      ...plannedPlacement(),
      outcome: "infeasible",
      reason: "insufficient_vram",
      candidate: null,
    });

    await expect(hosts.resolveFeasibleRoute(hal.id, sequenceRequest)).resolves.toBeNull();
    expect(previewChainPlacement).toHaveBeenCalledTimes(1);
    expect(previewChainPlacement.mock.calls[0]?.[0]).toEqual({
      baseUrl: hal.url,
      apiKey: "host-key",
    });
  });

  it("retains authoritative reasons and structured repair components per host", async () => {
    const hosts = useHostsStore();
    previewChainPlacement.mockResolvedValueOnce({
      ...plannedPlacement(),
      outcome: "infeasible",
      reason: "model 'flux-dev:q4' is missing a component",
      candidate: null,
      missing_components: [
        {
          kind: "vae",
          name: "ae.safetensors",
          present: false,
          repair_model: "flux-dev:q4",
        },
      ],
    });

    await expect(hosts.resolveFeasible("local", sequenceRequest)).resolves.toEqual({
      kind: "infeasible",
      perHost: [
        expect.objectContaining({
          hostId: "local",
          label: expect.any(String),
          reason: "model 'flux-dev:q4' is missing a component",
          missingComponents: [
            {
              kind: "vae",
              name: "ae.safetensors",
              present: false,
              repair_model: "flux-dev:q4",
            },
          ],
        }),
      ],
    });
  });

  it("distinguishes temporary scheduler failure from a malformed response", async () => {
    const hosts = useHostsStore();
    previewChainPlacement.mockResolvedValueOnce({
      ...plannedPlacement(),
      authoritative: false,
      outcome: "temporarily_unavailable",
      reason: "scheduler snapshot changed",
      candidate: null,
    });
    await expect(hosts.resolveFeasible("local", sequenceRequest)).resolves.toEqual({
      kind: "transient",
      perHost: [
        expect.objectContaining({
          hostId: "local",
          error: "scheduler snapshot changed",
        }),
      ],
    });

    previewChainPlacement.mockResolvedValueOnce({});
    await expect(hosts.resolveFeasible("local", sequenceRequest)).resolves.toEqual({
      kind: "unreachable",
      perHost: [
        expect.objectContaining({
          hostId: "local",
          error: "returned an invalid authoritative placement-preview response",
        }),
      ],
    });
  });

  it("treats malformed infeasible recovery metadata as an invalid response", async () => {
    const hosts = useHostsStore();
    previewChainPlacement.mockResolvedValueOnce({
      ...plannedPlacement(),
      outcome: "infeasible",
      reason: "missing a component",
      candidate: null,
      missing_components: [
        {
          kind: "vae",
          name: "",
          present: false,
          repair_model: "flux-dev:q4",
        },
      ],
    });

    await expect(hosts.resolveFeasible("local", sequenceRequest)).resolves.toEqual({
      kind: "unreachable",
      perHost: [
        expect.objectContaining({
          kind: "unreachable",
          hostId: "local",
          error: "returned an invalid authoritative placement-preview response",
        }),
      ],
    });
  });

  it("retains infeasible, temporary, and HTTP failures from one multi-host probe", async () => {
    const hosts = useHostsStore();
    hosts.extras.push(
      {
        id: hal.id,
        label: "hal9000",
        url: hal.url,
        apiKey: "hal-key",
        status: "ready",
        error: null,
        instanceId: "hal-instance",
      },
      {
        id: studio.id,
        label: "studio",
        url: studio.url,
        apiKey: "studio-key",
        status: "ready",
        error: null,
        instanceId: "studio-instance",
      },
    );
    previewChainPlacement
      .mockResolvedValueOnce({
        ...plannedPlacement(),
        outcome: "infeasible",
        reason: "missing VAE",
        candidate: null,
      })
      .mockResolvedValueOnce({
        ...plannedPlacement(),
        authoritative: false,
        outcome: "temporarily_unavailable",
        reason: "scheduler snapshot changed",
        candidate: null,
      })
      .mockRejectedValueOnce(
        new ApiError("placement failed", 401, { error: "API key was rejected" }),
      );

    await expect(hosts.resolveFeasible(null, sequenceRequest)).resolves.toEqual({
      kind: "mixed",
      perHost: [
        expect.objectContaining({
          kind: "infeasible",
          hostId: "local",
          reason: "missing VAE",
        }),
        expect.objectContaining({
          kind: "transient",
          hostId: hal.id,
          error: "scheduler snapshot changed",
        }),
        expect.objectContaining({
          kind: "unreachable",
          hostId: studio.id,
          error: "placement preview returned HTTP 401 — API key was rejected",
        }),
      ],
    });
  });

  it("routes to a clean planned host before a faster host with pending downloads", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "hal-key",
      status: "ready",
      error: null,
      instanceId: "hal-instance",
    });
    previewChainPlacement
      .mockResolvedValueOnce({
        ...plannedPlacement("cuda:local"),
        candidate: {
          ...plannedPlacement("cuda:local").candidate,
          predicted_completion_after_ms: 10,
        },
        pending_downloads: [
          {
            kind: "text_encoder",
            name: "t5-v1_1-xxl-q8.gguf",
            repo: "mold/runtime-components",
            bytes: 5_100_000_000,
          },
        ],
      })
      .mockResolvedValueOnce({
        ...plannedPlacement("cuda:remote"),
        candidate: {
          ...plannedPlacement("cuda:remote").candidate,
          predicted_completion_after_ms: 500,
        },
      });

    await expect(hosts.resolveFeasible(null, sequenceRequest)).resolves.toMatchObject({
      kind: "route",
      route: {
        hostId: hal.id,
        target: { baseUrl: hal.url, apiKey: "hal-key" },
      },
    });
  });

  it("stops waiting on a hung Auto candidate after another host returns a planned route", async () => {
    vi.useFakeTimers();
    try {
      const hosts = useHostsStore();
      hosts.extras.push({
        id: hal.id,
        label: "hal9000",
        url: hal.url,
        apiKey: "hal-key",
        status: "ready",
        error: null,
        instanceId: "hal-instance",
      });
      let remoteSignal: AbortSignal | undefined;
      previewChainPlacement.mockImplementation(
        (
          target: { baseUrl: string },
          _request: unknown,
          _copies: number,
          options?: { signal?: AbortSignal },
        ) => {
          if (!target.baseUrl.includes("hal9000")) return Promise.resolve(plannedPlacement());
          remoteSignal = options?.signal;
          return new Promise(() => {});
        },
      );

      const pending = hosts.resolveFeasible(null, sequenceRequest);
      await vi.waitFor(() => expect(previewChainPlacement).toHaveBeenCalledTimes(2));
      await vi.advanceTimersByTimeAsync(250);

      await expect(pending).resolves.toMatchObject({
        kind: "route",
        route: { hostId: "local" },
      });
      expect(remoteSignal?.aborted).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("fails closed when the only Auto candidate never finishes planning", async () => {
    vi.useFakeTimers();
    try {
      let requestSignal: AbortSignal | undefined;
      previewChainPlacement.mockImplementation(
        (
          _target: unknown,
          _request: unknown,
          _copies: number,
          options?: { signal?: AbortSignal },
        ) => {
          requestSignal = options?.signal;
          return new Promise(() => {});
        },
      );
      const hosts = useHostsStore();

      const pending = hosts.resolveFeasible(null, sequenceRequest);
      await vi.waitFor(() => expect(previewChainPlacement).toHaveBeenCalledTimes(1));
      // Nothing can route yet, so the first deadline extends once instead of
      // reporting a machine that is still checking as one that did not answer.
      await vi.advanceTimersByTimeAsync(5_000);
      let settled = false;
      void pending.then(() => (settled = true));
      await Promise.resolve();
      expect(settled).toBe(false);
      await vi.advanceTimersByTimeAsync(15_000);

      await expect(pending).resolves.toEqual({
        kind: "unreachable",
        perHost: [
          expect.objectContaining({
            hostId: "local",
            error: expect.stringContaining("Auto placement timed out after 20 seconds"),
          }),
        ],
      });
      expect(requestSignal?.aborted).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("keeps waiting past the Auto deadline for the only machine that might still plan", async () => {
    vi.useFakeTimers();
    try {
      const hosts = useHostsStore();
      hosts.extras.push({
        id: hal.id,
        label: "hal9000",
        url: hal.url,
        apiKey: "hal-key",
        status: "ready",
        error: null,
        instanceId: "hal-instance",
      });
      const slow = deferred<ReturnType<typeof plannedPlacement>>();
      previewChainPlacement.mockImplementation((target: { baseUrl: string }) =>
        target.baseUrl.includes("hal9000")
          ? slow.promise
          : Promise.resolve({
              ...plannedPlacement(),
              outcome: "infeasible",
              candidate: null,
              reason: "model 'flux-dev:q4' has no concrete local artifacts",
              missing_components: [
                {
                  kind: "transformer",
                  name: "transformer",
                  present: false,
                  repair_model: "flux-dev:q4",
                },
              ],
            }),
      );

      const pending = hosts.resolveFeasible(null, sequenceRequest);
      await vi.waitFor(() => expect(previewChainPlacement).toHaveBeenCalledTimes(2));
      await vi.advanceTimersByTimeAsync(5_000);
      let settled = false;
      void pending.then(() => (settled = true));
      await Promise.resolve();
      expect(settled).toBe(false);

      slow.resolve(plannedPlacement("cuda:remote"));
      await expect(pending).resolves.toMatchObject({
        kind: "route",
        route: { hostId: hal.id },
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it("reports a missing model apart from a machine that simply cannot fit it", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "hal-key",
      status: "ready",
      error: null,
      instanceId: "hal-instance",
    });
    previewChainPlacement
      .mockResolvedValueOnce({
        ...plannedPlacement(),
        outcome: "infeasible",
        candidate: null,
        reason: "model 'flux-dev:q4' has no concrete local artifacts",
        missing_components: [
          {
            kind: "transformer",
            name: "transformer",
            present: false,
            repair_model: "flux-dev:q4",
          },
        ],
      })
      .mockResolvedValueOnce({
        ...plannedPlacement(),
        outcome: "infeasible",
        candidate: null,
        reason: "no device can host this generation: needs 48.0 GB",
      });

    const result = await hosts.resolveFeasible(null, sequenceRequest);
    expect(result).toMatchObject({ kind: "infeasible" });
    const perHost = (result as { perHost: Array<Record<string, unknown>> }).perHost;
    expect(perHost[0]).toMatchObject({
      hostId: "local",
      missingModel: { model: "flux-dev:q4" },
    });
    expect(perHost[1]).toMatchObject({ hostId: hal.id, missingModel: null });
  });

  it("waits for every Most capable probe instead of using Auto's early route window", async () => {
    vi.useFakeTimers();
    try {
      const hosts = useHostsStore();
      hosts.extras.push({
        id: hal.id,
        label: "hal9000",
        url: hal.url,
        apiKey: "hal-key",
        status: "ready",
        error: null,
        instanceId: "hal-instance",
      });
      const stronger = deferred<ReturnType<typeof plannedPlacement>>();
      previewChainPlacement.mockImplementation((target: { baseUrl: string }) =>
        target.baseUrl.includes("hal9000")
          ? stronger.promise
          : Promise.resolve({
              ...plannedPlacement("cuda:local"),
              candidate: {
                ...plannedPlacement("cuda:local").candidate,
                predicted_completion_after_ms: 10_000,
              },
            }),
      );

      const pending = hosts.resolveFeasible("capable", sequenceRequest);
      await vi.waitFor(() => expect(previewChainPlacement).toHaveBeenCalledTimes(2));
      await vi.advanceTimersByTimeAsync(5_000);
      let settled = false;
      void pending.then(() => (settled = true));
      await Promise.resolve();
      expect(settled).toBe(false);

      stronger.resolve({
        ...plannedPlacement("cuda:remote"),
        candidate: {
          ...plannedPlacement("cuda:remote").candidate,
          predicted_completion_after_ms: 10,
        },
      });
      await expect(pending).resolves.toMatchObject({
        kind: "route",
        route: { hostId: hal.id },
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it("retains the placement probe HTTP status and message", async () => {
    previewChainPlacement.mockRejectedValueOnce(
      new ApiError("placement failed", 401, { error: "API key was rejected" }),
    );
    const hosts = useHostsStore();
    await expect(hosts.resolveFeasible("local", sequenceRequest)).resolves.toEqual({
      kind: "unreachable",
      perHost: [
        expect.objectContaining({
          hostId: "local",
          error: "placement preview returned HTTP 401 — API key was rejected",
        }),
      ],
    });
  });

  it.each([401, 403, 426, 500])("reports placement HTTP %s as unreachable", async (status) => {
    previewChainPlacement.mockRejectedValueOnce(new ApiError("placement failed", status));
    const hosts = useHostsStore();
    await expect(hosts.resolveFeasibleRoute("local", sequenceRequest)).resolves.toBeNull();
  });

  it("never refuses a pinned origin whose capabilities are not yet read", async () => {
    // Unread is "unknown", never "missing": admission asks the host itself.
    useConnectionStore().status = "starting";
    previewGenerationPlacement.mockRejectedValueOnce(new ApiError("unsupported", 404));
    const hosts = useHostsStore();

    await expect(hosts.resolveFeasible("local", placementRequest)).resolves.toMatchObject({
      kind: "route",
      route: {
        hostId: "local",
        target: { baseUrl: "http://127.0.0.1:49152", apiKey: "k" },
      },
    });
  });

  it("keeps the exact selected host for an explicit unsupported chain preview", async () => {
    const prefs = useAppPrefsStore();
    prefs.settings = settings({
      generateTargetHost: hal.id,
    }) as unknown as AppSettings;
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "instance-a",
    });
    previewChainPlacement.mockResolvedValueOnce({
      ...plannedPlacement(),
      authoritative: false,
      outcome: "unsupported",
      candidate: null,
    });
    const request: AutoChainRequest = {
      model: "ltx2.3-dev:bf16",
      prompt: "",
      total_frames: 193,
      clip_frames: 97,
      motion_tail_frames: 17,
      width: 768,
      height: 512,
      steps: 20,
      guidance: 3.5,
    };

    await expect(hosts.resolveFeasibleRoute(hal.id, request, 3)).resolves.toMatchObject({
      hostId: hal.id,
      instanceId: "instance-a",
      target: { baseUrl: hal.url, apiKey: "host-key" },
    });
    expect(previewChainPlacement).toHaveBeenCalledWith(expect.anything(), expect.anything(), 3, {
      signal: expect.any(AbortSignal),
    });
  });

  it("previews an auto-expanded long video through the chain endpoint", async () => {
    const hosts = useHostsStore();
    const request: AutoChainRequest = {
      model: "ltx2.3-dev:bf16",
      prompt: "",
      total_frames: 193,
      clip_frames: 97,
      motion_tail_frames: 17,
      width: 768,
      height: 512,
      steps: 20,
      guidance: 3.5,
    };

    await expect(hosts.resolveFeasibleRoute("local", request)).resolves.toMatchObject({
      hostId: "local",
    });
    expect(previewChainPlacement).toHaveBeenCalledTimes(1);
    expect(previewGenerationPlacement).not.toHaveBeenCalled();
  });

  it("previews prepared Batch N as N one-image siblings without mutating the request", async () => {
    const hosts = useHostsStore();
    const request = { ...sequenceRequest, batch_size: 4 };

    await expect(hosts.resolveFeasibleRoute("local", request, 4)).resolves.toMatchObject({
      hostId: "local",
    });

    expect(previewChainPlacement).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ batch_size: 1 }),
      4,
      { signal: expect.any(AbortSignal) },
    );
    expect(request.batch_size).toBe(4);
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
    // A wifi blip fails one poll: the verified snapshot remains available and
    // is marked stale, rather than becoming a false offline state.
    apiJsonTo.mockImplementation((target: { baseUrl: string }) =>
      target.baseUrl.includes("hal9000")
        ? Promise.reject(new Error("timeout"))
        : Promise.resolve({ queue_depth: 0, queue_capacity: 8, version: null }),
    );
    await hosts.refresh();
    const row = hosts.all.find((h) => h.id === hal.id);
    expect(row).toMatchObject({ status: "ready", stale: true });
    expect(row?.label).toBe("hal9000");
    expect(hosts.telemetry[hal.id]).toMatchObject({
      queueDepth: 0,
      queueCapacity: 8,
      stale: true,
    });
  });

  it("retains verified telemetry and capabilities through repeated status failures, then recovers", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "instance-a",
    });
    let remoteStatus: "fresh" | "failed" | "recovered" = "fresh";
    apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string) => {
      if (path === "/api/capabilities") {
        return Promise.resolve({
          gallery: { can_delete: true },
          queue: { heterogeneous_batch: true },
        });
      }
      if (!target.baseUrl.includes("hal9000")) {
        return Promise.resolve({ queue_depth: 0, queue_capacity: 8, version: "local" });
      }
      if (remoteStatus === "failed") return Promise.reject(new Error("status timeout"));
      return Promise.resolve({
        queue_depth: remoteStatus === "recovered" ? 7 : 5,
        queue_capacity: 32,
        version: remoteStatus === "recovered" ? "recovered" : "fresh",
        instance_id: "instance-a",
        gpu_info: {
          name: "RTX 4090",
          vram_total_mb: 24_564,
          vram_used_mb: 1_024,
        },
      });
    });
    listQueue.mockResolvedValue({ entries: [], plan: null });

    await hosts.refresh();
    const capabilitySnapshot = hosts.capabilities[hal.id];
    expect(hosts.all.find((host) => host.id === hal.id)).toMatchObject({
      status: "ready",
      stale: false,
      queueDepth: 5,
    });

    remoteStatus = "failed";
    await hosts.refresh();
    await hosts.refresh();

    expect(hosts.all.find((host) => host.id === hal.id)).toMatchObject({
      status: "ready",
      stale: true,
      queueDepth: 5,
    });
    expect(hosts.telemetry[hal.id]?.gpuInfo?.name).toBe("RTX 4090");
    expect(hosts.capabilities[hal.id]).toBe(capabilitySnapshot);

    remoteStatus = "recovered";
    await hosts.refresh();
    expect(hosts.all.find((host) => host.id === hal.id)).toMatchObject({
      status: "ready",
      stale: false,
      queueDepth: 7,
      version: "recovered",
    });
  });

  it("fences a verified host on an authoritative credential failure and accepts a rotated key", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "stale-key",
      status: "ready",
      error: null,
      instanceId: "instance-a",
    });
    hosts.telemetry[hal.id] = {
      queueDepth: 2,
      queueCapacity: 8,
      version: "last-good",
      instanceId: "instance-a",
      stale: false,
    };
    hosts.capabilities[hal.id] = { gallery: { can_delete: true } };
    useHostModelsStore().byHost[hal.id] = {
      entries: [installedModel("old-model")],
      fetchedAt: Date.now(),
      error: null,
    };
    apiJsonTo.mockImplementation((target: { baseUrl: string; apiKey: string | null }) =>
      target.baseUrl.includes("hal9000") && target.apiKey === "stale-key"
        ? Promise.reject(new ApiError("API key was rejected", 401))
        : Promise.resolve({ queue_depth: 0, queue_capacity: 8, version: "local" }),
    );

    await hosts.refresh();

    expect(hosts.all.find((host) => host.id === hal.id)).toMatchObject({
      status: "connecting",
      stale: false,
    });
    expect(hosts.telemetry[hal.id]).toBeUndefined();
    expect(hosts.capabilities[hal.id]).toBeUndefined();
    expect(useHostModelsStore().byHost[hal.id]).toBeUndefined();

    testRemoteHost.mockResolvedValue({
      ok: true,
      version: "current",
      error: null,
      instanceId: "instance-a",
      hostname: "hal9000",
    });
    await hosts.connect(hal.url, "rotated-key", null);

    expect(secretSet).toHaveBeenCalledWith(`remote-api-key.${hal.id}`, "rotated-key");
    expect(hosts.extras.find((host) => host.id === hal.id)).toMatchObject({
      apiKey: "rotated-key",
      status: "connecting",
      error: null,
    });
    await hosts.refresh();
    expect(hosts.extras.find((host) => host.id === hal.id)?.status).toBe("ready");
  });

  it.each(["devices", "queue", "capabilities"] as const)(
    "retires all verified authority when the %s health probe rejects the credential",
    async (endpoint) => {
      const hosts = useHostsStore();
      hosts.extras.push({
        id: hal.id,
        label: "hal9000",
        url: hal.url,
        apiKey: "rejected-key",
        status: "ready",
        error: null,
        instanceId: "instance-a",
      });
      hosts.telemetry[hal.id] = {
        queueDepth: 2,
        queueCapacity: 8,
        version: "last-good",
        instanceId: "instance-a",
        stale: false,
      };
      hosts.capabilities[hal.id] = { gallery: { can_delete: true } };
      useHostModelsStore().byHost[hal.id] = {
        entries: [installedModel("old-model")],
        fetchedAt: Date.now(),
        error: null,
      };
      apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string) => {
        if (
          endpoint === "capabilities" &&
          target.baseUrl.includes("hal9000") &&
          path === "/api/capabilities"
        )
          return Promise.reject(new ApiError("API key was rejected", 403));
        return Promise.resolve({
          queue_depth: 1,
          queue_capacity: 8,
          version: "current",
          instance_id: target.baseUrl.includes("hal9000") ? "instance-a" : "local",
        });
      });
      listDevices.mockImplementation((target: { baseUrl: string }) =>
        endpoint === "devices" && target.baseUrl.includes("hal9000")
          ? Promise.reject(new ApiError("API key was rejected", 401))
          : Promise.resolve({ plan_version: 1, devices: [device(0)] }),
      );
      listQueue.mockImplementation((target: { baseUrl: string }) =>
        endpoint === "queue" && target.baseUrl.includes("hal9000")
          ? Promise.reject(new ApiError("API key was rejected", 401))
          : Promise.resolve({ entries: [], plan: null }),
      );

      await hosts.refresh();

      expect(hosts.all.find((host) => host.id === hal.id)).toMatchObject({
        status: "connecting",
        stale: false,
      });
      expect(hosts.telemetry[hal.id]).toBeUndefined();
      expect(hosts.capabilities[hal.id]).toBeUndefined();
      expect(useHostModelsStore().byHost[hal.id]).toBeUndefined();
    },
  );

  it("retires verified authority immediately when a host key changes", async () => {
    const nextStatus = deferred<Record<string, unknown>>();
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "old-key",
      status: "ready",
      error: null,
      instanceId: "instance-a",
    });
    hosts.telemetry[hal.id] = {
      queueDepth: 4,
      queueCapacity: 8,
      version: "old",
      instanceId: "instance-a",
      stale: false,
    };
    hosts.capabilities[hal.id] = { gallery: { can_delete: true } };
    useHostModelsStore().byHost[hal.id] = {
      entries: [installedModel("old-model")],
      fetchedAt: Date.now(),
      error: null,
    };
    testRemoteHost.mockResolvedValue({
      ok: true,
      version: "current",
      error: null,
      instanceId: "instance-a",
      hostname: "hal9000",
    });
    apiJsonTo.mockImplementation(
      (target: { baseUrl: string; apiKey: string | null }, path: string) =>
        target.baseUrl.includes("hal9000") && target.apiKey === "new-key" && path === "/api/status"
          ? nextStatus.promise
          : Promise.resolve({}),
    );

    const connected = await hosts.connect(hal.url, "new-key", null);

    expect(connected.status).toBe("connecting");
    expect(hosts.telemetry[hal.id]).toBeUndefined();
    expect(hosts.capabilities[hal.id]).toBeUndefined();
    expect(useHostModelsStore().byHost[hal.id]).toBeUndefined();

    nextStatus.resolve({
      queue_depth: 1,
      queue_capacity: 8,
      version: "new",
      instance_id: "instance-a",
    });
    await hosts.refresh();
    expect(hosts.all.find((host) => host.id === hal.id)?.status).toBe("ready");
  });

  it("adopts a validated replacement address for a verified stale instance twin", async () => {
    const nextStatus = deferred<Record<string, unknown>>();
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: "timeout",
      instanceId: "instance-a",
    });
    hosts.telemetry[hal.id] = {
      queueDepth: 2,
      queueCapacity: 8,
      version: "last-good",
      instanceId: "instance-a",
      hostname: "hal9000",
      stale: true,
    };
    hosts.capabilities[hal.id] = { gallery: { can_delete: true } };
    useHostModelsStore().byHost[hal.id] = {
      entries: [installedModel("old-model")],
      fetchedAt: Date.now(),
      error: null,
    };
    testRemoteHost.mockResolvedValue({
      ok: true,
      version: "current",
      error: null,
      instanceId: "instance-a",
      hostname: "hal9000",
    });
    apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string) =>
      target.baseUrl.includes("192.168.1.99") && path === "/api/status"
        ? nextStatus.promise
        : Promise.resolve({}),
    );

    const connected = await hosts.connect("http://192.168.1.99:7680", "host-key", null);

    expect(connected.id).toBe(hal.id);
    expect(connected.status).toBe("connecting");
    expect(connected.baseUrl).toBe("http://192.168.1.99:7680");
    expect(hosts.extras.find((host) => host.id === hal.id)?.url).toBe("http://192.168.1.99:7680");
    expect(hosts.telemetry[hal.id]).toBeUndefined();
    expect(hosts.capabilities[hal.id]).toBeUndefined();
    expect(useHostModelsStore().byHost[hal.id]).toBeUndefined();

    nextStatus.resolve({
      queue_depth: 0,
      queue_capacity: 8,
      version: "new",
      instance_id: "instance-a",
    });
    await hosts.refresh();
  });

  it("fences capabilities from a replaced server identity", async () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "instance-a",
    });
    hosts.telemetry[hal.id] = {
      queueDepth: 2,
      queueCapacity: 8,
      version: "old",
      instanceId: "instance-a",
    };
    hosts.capabilities[hal.id] = {
      gallery: { can_delete: true },
      model_access: { restrictions: [] },
    };
    useHostModelsStore().byHost[hal.id] = {
      entries: [installedModel("old-model")],
      fetchedAt: Date.now(),
      error: null,
    };
    apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string) => {
      if (path === "/api/capabilities" && target.baseUrl.includes("hal9000")) {
        return Promise.reject(new Error("capabilities unavailable"));
      }
      return Promise.resolve({
        queue_depth: 1,
        queue_capacity: 8,
        version: "new",
        instance_id: target.baseUrl.includes("hal9000") ? "instance-b" : "local",
      });
    });
    listQueue.mockRejectedValue(new Error("queue unavailable"));

    await hosts.refresh();

    expect(hosts.telemetry[hal.id]).toMatchObject({
      instanceId: "instance-b",
      predictedCompletionMs: null,
      stale: false,
    });
    expect(hosts.capabilities[hal.id]).toBeUndefined();
    expect(useHostModelsStore().byHost[hal.id]).toBeUndefined();
  });

  it("marks the local host offline only after native lifecycle authority confirms death", async () => {
    const hosts = useHostsStore();
    hosts.telemetry.local = {
      queueDepth: 4,
      queueCapacity: 8,
      version: "last-good",
      stale: false,
    };
    hosts.capabilities.local = { gallery: { can_delete: true } };
    ensureLocalServer.mockRejectedValueOnce(new Error("embedded process exited"));
    apiJsonTo.mockRejectedValueOnce(new Error("status timeout"));

    await hosts.refresh();
    await vi.waitFor(() => expect(useConnectionStore().localStatus).toBe("error"));

    expect(hosts.primaryHost?.status).toBe("error");
    expect(hosts.telemetry.local).toBeUndefined();
    expect(hosts.capabilities.local).toBeUndefined();
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
    expect(listQueue).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://127.0.0.1:49152" }),
      { limit: 8 },
    );
    expect(listQueue).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://hal9000:7680" }),
      { limit: 8 },
    );
    expect(
      listQueue.mock.calls.every(
        ([, page]) => (page as { limit?: number } | undefined)?.limit === 8,
      ),
    ).toBe(true);
  });

  it.each([undefined, null, 0, -1, 1.5, Number.NaN])(
    "keeps the legacy queue read when status capacity is %s",
    async (queueCapacity) => {
      apiJsonTo.mockResolvedValue({
        queue_depth: 0,
        queue_capacity: queueCapacity,
        version: "legacy",
      });
      listQueue.mockResolvedValue({ entries: [], plan: null });
      const hosts = useHostsStore();

      await hosts.refresh();

      expect(listQueue).toHaveBeenCalledWith({
        baseUrl: "http://127.0.0.1:49152",
        apiKey: "k",
      });
      expect(listQueue.mock.calls[0]).toHaveLength(1);
    },
  );

  it("keeps healthy status and last-good plan timing when the bounded queue page fails", async () => {
    const now = Date.now();
    apiJsonTo.mockResolvedValue({
      queue_depth: 1,
      queue_capacity: 8,
      version: "current",
    });
    listQueue.mockResolvedValueOnce({
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
            estimated_finish_unix_ms: now + 10_000,
          },
        ],
      },
    });
    const hosts = useHostsStore();
    await hosts.refresh();
    const lastGood = hosts.telemetry.local?.predictedCompletionMs;

    apiJsonTo.mockResolvedValue({
      queue_depth: 2,
      queue_capacity: 8,
      version: "still-current",
    });
    listQueue.mockRejectedValueOnce(new Error("queue unavailable"));
    await hosts.refresh();

    expect(hosts.primaryHost).toMatchObject({
      status: "ready",
      queueDepth: 2,
      version: "still-current",
      predictedCompletionMs: lastGood,
    });
  });

  it("coalesces overlapping refreshes into one request wave", async () => {
    const status = deferred<Record<string, unknown>>();
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") return status.promise;
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
    expect(apiJsonTo.mock.calls.filter(([, path]) => path === "/api/status")).toHaveLength(1);

    status.resolve({
      queue_depth: 9,
      queue_capacity: 16,
      version: "current",
    });
    await Promise.all([first, second]);

    expect(hosts.telemetry.local?.queueDepth).toBe(9);
    expect(hosts.telemetry.local?.version).toBe("current");
  });

  it("does not let a pre-disconnect refresh restore authority after reconnect", async () => {
    const oldStatus = deferred<Record<string, unknown>>();
    const newStatus = deferred<Record<string, unknown>>();
    let remoteCalls = 0;
    apiJsonTo.mockImplementation((target: { baseUrl: string }, path: string) => {
      if (path === "/api/capabilities") return Promise.resolve({ gallery: { can_delete: true } });
      if (!target.baseUrl.includes("hal9000")) {
        return Promise.resolve({
          queue_depth: 0,
          queue_capacity: 8,
          version: "local",
          instance_id: "local",
        });
      }
      remoteCalls += 1;
      return remoteCalls === 1 ? oldStatus.promise : newStatus.promise;
    });
    listDevices.mockResolvedValue({ plan_version: 1, devices: [device(0)] });
    listQueue.mockResolvedValue({ entries: [], plan: null });
    testRemoteHost.mockResolvedValue({
      ok: true,
      version: "new",
      error: null,
      instanceId: "instance-b",
      hostname: "hal9000",
    });
    const hosts = useHostsStore();
    hosts.extras.push({
      id: hal.id,
      label: "hal9000",
      url: hal.url,
      apiKey: "host-key",
      status: "ready",
      error: null,
      instanceId: "instance-a",
    });
    installSettings(settings({ savedHosts: [hal], connectedHostIds: [hal.id] }));

    const obsolete = hosts.refresh();
    await hosts.disconnect(hal.id);
    await hosts.connect(hal.url, "host-key", null);
    oldStatus.resolve({
      queue_depth: 9,
      queue_capacity: 16,
      version: "old",
      instance_id: "instance-a",
    });
    await obsolete;
    await Promise.resolve();

    expect(hosts.telemetry[hal.id]).toBeUndefined();
    expect(hosts.capabilities[hal.id]).toBeUndefined();

    newStatus.resolve({
      queue_depth: 1,
      queue_capacity: 8,
      version: "new",
      instance_id: "instance-b",
    });
    await hosts.refresh();
    expect(hosts.telemetry[hal.id]).toMatchObject({
      queueDepth: 1,
      version: "new",
      instanceId: "instance-b",
    });
  });

  it("runs a non-overlapping follow-up wave when host connection state changes", async () => {
    const firstStatus = deferred<Record<string, unknown>>();
    const secondStatus = deferred<Record<string, unknown>>();
    let statusCalls = 0;
    apiJsonTo.mockImplementation((_target: unknown, path: string) => {
      if (path === "/api/status") {
        statusCalls += 1;
        return statusCalls === 1 ? firstStatus.promise : secondStatus.promise;
      }
      if (path === "/api/capabilities") return Promise.resolve({});
      return Promise.reject(new Error(`unexpected path ${path}`));
    });
    listDevices.mockResolvedValue({ plan_version: 1, devices: [device(0)] });
    listQueue.mockResolvedValue({ entries: [], plan: null });
    const hosts = useHostsStore();

    const current = hosts.refresh();
    const latest = hosts.refreshAfterCurrent();
    expect(statusCalls).toBe(1);

    firstStatus.resolve({ queue_depth: 1, queue_capacity: 8, version: "first" });
    await current;
    await Promise.resolve();
    expect(statusCalls).toBe(2);

    secondStatus.resolve({ queue_depth: 2, queue_capacity: 8, version: "second" });
    await latest;
    expect(hosts.telemetry.local?.version).toBe("second");
  });

  it("schedules the next poll only after the current refresh settles", async () => {
    vi.useFakeTimers();
    try {
      const firstStatus = deferred<Record<string, unknown>>();
      const secondStatus = deferred<Record<string, unknown>>();
      let statusCalls = 0;
      apiJsonTo.mockImplementation((_target: unknown, path: string) => {
        if (path === "/api/status") {
          statusCalls += 1;
          return statusCalls === 1 ? firstStatus.promise : secondStatus.promise;
        }
        if (path === "/api/capabilities") return Promise.resolve({});
        return Promise.reject(new Error(`unexpected path ${path}`));
      });
      listDevices.mockResolvedValue({ plan_version: 1, devices: [device(0)] });
      listQueue.mockResolvedValue({ entries: [], plan: null });
      const hosts = useHostsStore();

      const current = hosts.refresh();
      hosts.startPolling();
      expect(statusCalls).toBe(1);
      await vi.advanceTimersByTimeAsync(30_000);
      expect(statusCalls).toBe(1);

      firstStatus.resolve({ queue_depth: 0, queue_capacity: 8, version: "first" });
      await current;
      await Promise.resolve();
      await vi.advanceTimersByTimeAsync(9_999);
      expect(statusCalls).toBe(1);
      await vi.advanceTimersByTimeAsync(1);
      expect(statusCalls).toBe(2);
      secondStatus.resolve({ queue_depth: 0, queue_capacity: 8, version: "second" });
      await Promise.resolve();
    } finally {
      useHostsStore().stopPolling();
      vi.useRealTimers();
    }
  });

  it("keeps polling after a refresh-level reconciliation failure", async () => {
    vi.useFakeTimers();
    try {
      const hosts = useHostsStore();
      const refresh = vi
        .spyOn(hosts, "refresh")
        .mockRejectedValueOnce(new Error("settings unavailable"))
        .mockResolvedValue(undefined);

      hosts.startPolling();
      await Promise.resolve();
      await Promise.resolve();
      expect(refresh).toHaveBeenCalledTimes(1);

      await vi.advanceTimersByTimeAsync(10_000);
      expect(refresh).toHaveBeenCalledTimes(2);
    } finally {
      useHostsStore().stopPolling();
      vi.useRealTimers();
    }
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

    expect(listQueue).toHaveBeenCalledWith(
      {
        baseUrl: "http://hal9000:7680",
        apiKey: null,
      },
      { limit: 8 },
    );
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
    expect(hosts.all.find((h) => h.id === hal.id)?.status).toBe("connecting");

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
    expect(view.status).toBe("connecting");
    expect(view.baseUrl).toBe("http://192.168.1.99:7680");
    const persisted = appSettingsSet.mock.lastCall?.[0] as ReturnType<typeof settings>;
    expect(persisted.savedHosts.find((h: SavedHost) => h.id === hal.id)?.url).toBe(
      "http://192.168.1.99:7680",
    );
    await hosts.refresh();
    expect(hosts.extras.find((host) => host.id === hal.id)?.status).toBe("ready");
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
    expect(hosts.all.find((h) => h.id === hal.id)?.status).toBe("connecting");

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
    expect(live.status).toBe("connecting");
    expect(live.error).toBeNull();
    await hosts.refresh();
    expect(live.status).toBe("ready");
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

  it("reconcile keeps the explicitly connected route when saved aliases collapse", async () => {
    // Only the IP alias is connected. Reconcile keeps that live route while
    // collapsing persistence to the preferred, more recently used slug.
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
    expect(row).toMatchObject({ status: "ready", baseUrl: ip.url });
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
