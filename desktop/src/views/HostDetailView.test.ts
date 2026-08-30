import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";

const { apiJsonTo, listDevices, setDeviceEnabled } = vi.hoisted(() => ({
  apiJsonTo: vi.fn(),
  listDevices: vi.fn(),
  setDeviceEnabled: vi.fn(),
}));
vi.mock("../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
  apiFetchTo: vi.fn(),
  apiJson: vi.fn(),
  apiFetch: vi.fn(),
  currentTarget: () => ({ baseUrl: "http://127.0.0.1:49152", apiKey: null }),
}));
vi.mock("@studio/api/devices", () => ({
  listDevices,
  setDeviceEnabled,
}));
const { listTrash, emptyTrash, fetchHostConfigKey, setHostConfigKey } = vi.hoisted(() => ({
  listTrash: vi.fn(),
  emptyTrash: vi.fn(),
  fetchHostConfigKey: vi.fn(),
  setHostConfigKey: vi.fn(),
}));
vi.mock("@studio/api/galleryOrganization", () => ({
  listTrash: (...a: unknown[]) => listTrash(...a),
  emptyTrash: (...a: unknown[]) => emptyTrash(...a),
}));
vi.mock("../lib/api/hostConfig", () => ({
  fetchHostConfigKey: (...a: unknown[]) => fetchHostConfigKey(...a),
  setHostConfigKey: (...a: unknown[]) => setHostConfigKey(...a),
}));

interface SseCall {
  path: string;
  options: {
    signal: AbortSignal;
    target?: { baseUrl: string; apiKey: string | null };
    onOpen?: () => void;
    onEvent: (event: string, data: string) => void;
  };
}
const sseCalls: SseCall[] = [];
vi.mock("../lib/api/sse", () => ({
  sseStream: (path: string, options: SseCall["options"]) => {
    sseCalls.push({ path, options });
    options.onOpen?.();
    return new Promise(() => {});
  },
}));

const forgetRemoteHost = vi.fn().mockResolvedValue([]);
const appSettingsGet = vi.fn();
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: (...a: unknown[]) => appSettingsGet(...a),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    secretGet: vi.fn().mockResolvedValue(null),
    secretSet: vi.fn().mockResolvedValue(undefined),
    forgetRemoteHost: (...a: unknown[]) => forgetRemoteHost(...a),
  },
}));
vi.mock("../lib/notify", () => ({ notifyPulled: vi.fn() }));

const unloadModel = vi.hoisted(() => vi.fn());
vi.mock("../lib/api/models", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/models")>()),
  unloadModel: (...a: unknown[]) => unloadModel(...a),
}));

import HostDetailView from "./HostDetailView.vue";
import { authenticatedMiniMaxH3Capabilities } from "@studio/lib/minimaxH3Inventory.testFixtures";
import { useComposerStore } from "../stores/composer";
import { useConnectionStore } from "../stores/connection";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import type { DeviceInfo } from "@studio/api/devices";
import type { ModelEntry, ServerCapabilities, ServerStatus } from "../lib/api/types";

const stub = { template: "<div />" };

const REMOTE_ID = "hal9000-7680";

const DEVICE: DeviceInfo = {
  id: "cuda:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  backend: "cuda",
  ordinal: 0,
  device_kind: "full_gpu",
  nvml_uuid: "GPU-a",
  physical_uuid: "GPU-a",
  mig_uuid: null,
  mig_parent_uuid: null,
  mig_profile: null,
  name: "NVIDIA RTX 3090",
  pci_bus_id: null,
  compute_capability: "8.6",
  memory: {
    total_bytes: 24_000_000_000,
    used_bytes: 4_000_000_000,
    mold_used_bytes: null,
    other_used_bytes: null,
  },
  telemetry: { utilization_percent: 10, temperature_c: null, power_w: null },
  desired_enabled: true,
  admin_state: "enabled",
  health: "healthy",
  activity: "idle",
  schedulable: true,
  unschedulable_reason: null,
  loaded_models: [],
  active_work_id: null,
  planned_work_ids: [],
};

interface DeviceFixture {
  devices: DeviceInfo[];
  capabilities?: ServerCapabilities;
}

interface WireQueueEntry {
  id: string;
  model: string;
  state: "queued" | "running";
  started_at_unix_ms: number;
  position: number;
  gpu?: number;
  seed_pinned?: boolean;
  metadata?: Record<string, unknown>;
}

function installApi(
  status: Partial<ServerStatus> = {},
  queueEntries: WireQueueEntry[] = [],
  models: ModelEntry[] = [model("flux-dev:q8", "flux"), model("z-image:q8", "z-image")],
) {
  apiJsonTo.mockImplementation((_target: unknown, path: string) => {
    if (path === "/api/status") {
      return Promise.resolve({
        version: "0.17.0",
        models_loaded: [],
        uptime_secs: 5,
        ...status,
      });
    }
    if (path === "/api/models") return Promise.resolve(models);
    if (path.startsWith("/api/queue")) return Promise.resolve({ entries: queueEntries });
    if (path === "/api/capabilities")
      return Promise.resolve({
        queue: { can_pause: true, can_cancel_all: true },
        events: { available: true },
      });
    return Promise.reject(new Error(`unexpected ${path}`));
  });
}

function model(name: string, family: string): ModelEntry {
  return {
    name,
    family,
    size_gb: 12.4,
    is_loaded: false,
    hf_repo: "x/y",
    default_steps: 4,
    default_guidance: 1,
    default_width: 1024,
    default_height: 1024,
    description: "",
    downloaded: true,
  };
}

let router: Router;
let mountedHosts: ReturnType<typeof useHostsStore>;
const mountedViews: Array<ReturnType<typeof mount>> = [];

async function mountView(
  path = `/hosts/${REMOTE_ID}`,
  entries: ModelEntry[] = [model("flux-dev:q8", "flux"), model("z-image:q8", "z-image")],
  deviceFixture?: DeviceFixture,
) {
  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      { path: "/hosts/:id", component: stub },
      { path: "/machines", component: stub },
      { path: "/machines/:id", component: stub },
      { path: "/settings", component: stub },
      { path: "/models", component: stub },
      { path: "/jobs", component: stub },
      { path: "/generate", component: stub },
    ],
  });
  router.push(path);
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  const hosts = useHostsStore();
  mountedHosts = hosts;
  hosts.extras.push({
    id: REMOTE_ID,
    label: "hal9000",
    url: "http://hal9000:7680",
    apiKey: "sekrit",
    status: "ready",
    error: null,
    instanceId: null,
  });
  hosts.telemetry[REMOTE_ID] = {
    queueDepth: 2,
    queueCapacity: 8,
    version: "0.17.0",
    modelsLoaded: ["flux-dev:q8"],
    gpuInfo: { name: "NVIDIA GeForce RTX 4090", vram_total_mb: 24_000, vram_used_mb: 6_000 },
    gpuWorkers: [
      {
        ordinal: 0,
        name: "NVIDIA GeForce RTX 4090",
        vram_total_bytes: 24_000_000_000,
        vram_used_bytes: 6_000_000_000,
        state: "generating",
      },
      {
        ordinal: 1,
        name: "NVIDIA B200",
        vram_total_bytes: 80_000_000_000,
        vram_used_bytes: 20_000_000_000,
        state: "idle",
      },
    ],
    instanceId: "0f7a2c31-instance-uuid",
    hostname: "hal9000",
    ...(deviceFixture ? { devices: deviceFixture.devices } : {}),
  };
  if (deviceFixture?.capabilities) {
    hosts.capabilities[REMOTE_ID] = deviceFixture.capabilities;
  }
  // fetchedAt now → the view's hostModels.refresh() skips these (not stale).
  const hostModels = useHostModelsStore();
  hostModels.byHost[REMOTE_ID] = {
    entries,
    fetchedAt: Date.now(),
    error: null,
  };
  const wrapper = mount(HostDetailView, {
    global: { plugins: [pinia, router] },
  });
  mountedViews.push(wrapper);
  await flushPromises();
  return wrapper;
}

/** The resources stream opened for the remote host (the newest matching call). */
function lastStream(path = "/api/resources/stream"): SseCall {
  const call = [...sseCalls].reverse().find((candidate) => candidate.path === path);
  if (!call) throw new Error("no sseStream call recorded");
  return call;
}

beforeEach(() => {
  vi.clearAllMocks();
  listTrash.mockResolvedValue([]);
  emptyTrash.mockResolvedValue({ purged: 0 });
  fetchHostConfigKey.mockResolvedValue({
    key: "gallery.trash_retention_days",
    value: 30,
    source: "default",
  });
  setHostConfigKey.mockResolvedValue(new Response(null, { status: 200 }));
  listDevices.mockRejectedValue(new Error("legacy server in tests"));
  setDeviceEnabled.mockResolvedValue(undefined);
  unloadModel.mockResolvedValue(undefined);
  sseCalls.length = 0;
  appSettingsGet.mockResolvedValue({
    savedHosts: [],
    connectedHostIds: [REMOTE_ID],
    generateTargetHost: null,
  });
  installApi();
});

afterEach(() => {
  for (const wrapper of mountedViews) wrapper.unmount();
  mountedViews.length = 0;
});

describe("HostDetailView GPU lifecycle controls", () => {
  function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((next) => {
      resolve = next;
    });
    return { promise, resolve };
  }

  const authoritative: ServerCapabilities = {
    gallery: { can_delete: true },
    devices: { available: true, lifecycle: true, restart_enable: false },
    dispatch: { active_mode: "v2", v2_authoritative: true, observes_v2_decisions: false },
  };
  const restartOnly: ServerCapabilities = {
    gallery: { can_delete: true },
    devices: { available: true, lifecycle: true, restart_enable: true },
    dispatch: {
      active_mode: "observe",
      v2_authoritative: false,
      observes_v2_decisions: true,
    },
  };

  it("allows live disable only for an authoritative Scheduler V2 host", async () => {
    const lifecycleDevices = [
      DEVICE,
      {
        ...DEVICE,
        id: "cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ordinal: 1,
        admin_state: "startup_excluded" as const,
        desired_enabled: false,
        schedulable: false,
      },
    ];
    listDevices.mockResolvedValue({ devices: lifecycleDevices, plan_version: 1 });
    const wrapper = await mountView(undefined, undefined, {
      devices: lifecycleDevices,
      capabilities: authoritative,
    });

    const live = wrapper.get("[data-test='device-toggle-0']");
    expect(live.attributes("disabled")).toBeUndefined();
    expect(live.text()).toBe("Disable");
    expect(wrapper.text()).toContain(
      "Disabling a busy GPU lets its current stage finish, then removes it from scheduling.",
    );
    expect(wrapper.get("[data-test='device-toggle-1']").attributes("disabled")).toBeDefined();

    await live.trigger("click");
    await flushPromises();
    expect(setDeviceEnabled).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "sekrit" },
      DEVICE.id,
      false,
    );
    expect(listDevices).toHaveBeenCalled();
  });

  it("does not let an older mutation clear a newer device's busy state", async () => {
    const secondDevice = {
      ...DEVICE,
      id: "cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      ordinal: 1,
    };
    const devices = [DEVICE, secondDevice];
    const first = deferred<void>();
    const second = deferred<void>();
    listDevices.mockResolvedValue({ devices, plan_version: 1 });
    setDeviceEnabled.mockImplementation((_target, id) =>
      id === DEVICE.id ? first.promise : second.promise,
    );
    const wrapper = await mountView(undefined, undefined, {
      devices,
      capabilities: authoritative,
    });
    const refresh = vi.spyOn(mountedHosts, "refresh").mockResolvedValue();

    await wrapper.get("[data-test='device-toggle-0']").trigger("click");
    await wrapper.get("[data-test='device-toggle-1']").trigger("click");
    expect(wrapper.get("[data-test='device-toggle-0']").attributes("disabled")).toBeDefined();
    expect(wrapper.get("[data-test='device-toggle-1']").attributes("disabled")).toBeDefined();

    first.resolve();
    await vi.waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
    expect(wrapper.get("[data-test='device-toggle-0']").attributes("disabled")).toBeUndefined();
    expect(wrapper.get("[data-test='device-toggle-1']").attributes("disabled")).toBeDefined();

    second.resolve();
    await vi.waitFor(() =>
      expect(wrapper.findComponent({ name: "DevicePanel" }).props("busyDeviceIds")).toEqual([]),
    );
    expect(wrapper.get("[data-test='device-toggle-1']").attributes("disabled")).toBeUndefined();
  });

  it("blocks disable in Observe mode but offers disabled GPUs restart-only recovery", async () => {
    const disabled = {
      ...DEVICE,
      desired_enabled: false,
      admin_state: "disabled" as const,
      schedulable: false,
    };
    const pendingRestart = {
      ...DEVICE,
      id: "cuda:cccccccccccccccccccccccccccccccc",
      ordinal: 2,
      restart_required: true,
      health: "unavailable" as const,
      schedulable: false,
      unschedulable_reason: "device_unavailable",
    };
    listDevices.mockResolvedValue({
      devices: [
        DEVICE,
        { ...disabled, id: "cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", ordinal: 1 },
        pendingRestart,
      ],
      plan_version: 1,
    });
    const wrapper = await mountView(undefined, undefined, {
      devices: [
        DEVICE,
        { ...disabled, id: "cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", ordinal: 1 },
        pendingRestart,
      ],
      capabilities: restartOnly,
    });

    const disable = wrapper.get("[data-test='device-toggle-0']");
    expect(disable.text()).toBe("Disable");
    expect(disable.attributes("disabled")).toBeDefined();
    await disable.trigger("click");
    expect(setDeviceEnabled).not.toHaveBeenCalled();

    const restartEnable = wrapper.get("[data-test='device-toggle-1']");
    expect(restartEnable.text()).toBe("Enable on restart");
    expect(restartEnable.attributes("disabled")).toBeUndefined();
    expect(wrapper.text()).toContain(
      "Live GPU controls require Scheduler V2. Disabled GPUs can be enabled for the next server restart.",
    );
    await restartEnable.trigger("click");
    await flushPromises();
    expect(setDeviceEnabled).toHaveBeenCalledWith(
      { baseUrl: "http://hal9000:7680", apiKey: "sekrit" },
      "cuda:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      true,
    );

    const pending = wrapper.get("[data-test='device-toggle-2']");
    expect(pending.text()).toBe("Enabled on restart");
    expect(pending.attributes("disabled")).toBeDefined();
    expect(wrapper.findAll("[data-test='device-card']")[2]!.text()).toContain("Restart required");
    expect(wrapper.findAll("[data-test='device-card']")[2]!.text()).not.toContain("unavailable");
  });

  it("treats older hosts with missing lifecycle capabilities as unsupported", async () => {
    listDevices.mockResolvedValue({ devices: [DEVICE], plan_version: 1 });
    const wrapper = await mountView(undefined, undefined, { devices: [DEVICE] });
    const button = wrapper.get("[data-test='device-toggle-0']");

    expect(button.text()).toBe("Disable");
    expect(button.attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("Live GPU controls are unavailable on this server.");
    await button.trigger("click");
    expect(setDeviceEnabled).not.toHaveBeenCalled();
  });
});

describe("HostDetailView header", () => {
  it("renders name, status, kind, url, version and instance id from the stores", async () => {
    const wrapper = await mountView();
    expect(wrapper.get("[data-test='host-title']").text()).toBe("hal9000");
    expect(wrapper.text()).toContain("REMOTE");
    expect(wrapper.get("[data-test='host-url']").text()).toBe("http://hal9000:7680");
    expect(wrapper.get("[data-test='host-version']").text()).toContain("0.17.0");
    const instance = wrapper.get("[data-test='host-instance-id']");
    expect(instance.text()).toContain("0f7a2c31-instance-uuid");
    // Remote hosts get the remote-only management actions.
    expect(wrapper.find("[data-test='rename-host']").exists()).toBe(true);
    expect(wrapper.find("[data-test='disconnect-host']").exists()).toBe(true);
    expect(wrapper.find("[data-test='forget-host']").exists()).toBe(true);
  });

  it("copies the full instance id on click even though the display truncates", async () => {
    const wrapper = await mountView();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });

    await wrapper.get("[data-test='host-instance-id']").trigger("click");
    expect(writeText).toHaveBeenCalledWith("0f7a2c31-instance-uuid");
  });

  it("reports a clipboard failure instead of silently rejecting", async () => {
    const wrapper = await mountView();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockRejectedValue(new Error("denied")) },
    });

    await wrapper.get("[data-test='host-instance-id']").trigger("click");
    await flushPromises();
    const { useToastStore } = await import("../stores/toasts");
    const toasts = useToastStore();
    expect(toasts.items.some((t) => t.kind === "error" && t.message.includes("copy"))).toBe(true);
  });

  it("renders the local primary without remote-only actions", async () => {
    const wrapper = await mountView("/hosts/local");
    expect(wrapper.text()).toContain("THIS DEVICE");
    expect(wrapper.find("[data-test='rename-host']").exists()).toBe(false);
    expect(wrapper.find("[data-test='disconnect-host']").exists()).toBe(false);
    expect(wrapper.find("[data-test='forget-host']").exists()).toBe(false);
  });

  it("renders a quiet empty state with a back link for an unknown id", async () => {
    const wrapper = await mountView("/hosts/no-such-host");
    expect(wrapper.find("[data-test='host-title']").exists()).toBe(false);
    const missing = wrapper.get("[data-test='host-missing']");
    expect(missing.text()).toContain("Host not found");
    expect(missing.find("[data-test='back-to-hosts']").attributes("href")).toBe("/machines");
    // No stream is opened for a host that doesn't exist.
    expect(sseCalls).toHaveLength(0);
  });
});

describe("HostDetailView telemetry", () => {
  it("subscribes to THIS host's resources stream and renders live frames", async () => {
    const wrapper = await mountView();
    const stream = lastStream();
    expect(stream.path).toBe("/api/resources/stream");
    expect(stream.options.target).toEqual({ baseUrl: "http://hal9000:7680", apiKey: "sekrit" });

    // Before any frame: every status-poll worker is visible.
    const fallbackCards = wrapper.findAll("[data-test='gpu-card']");
    expect(fallbackCards).toHaveLength(2);
    expect(fallbackCards[0]!.text()).toContain("NVIDIA GeForce RTX 4090");
    expect(fallbackCards[0]!.text()).toContain("6.0 GB/24.0 GB");
    expect(fallbackCards[1]!.text()).toContain("NVIDIA B200");
    expect(fallbackCards[1]!.text()).toContain("20.0 GB/80.0 GB");
    expect(wrapper.find("[data-test='cpu-card']").exists()).toBe(false);

    stream.options.onEvent(
      "snapshot",
      JSON.stringify({
        hostname: "hal9000",
        timestamp: 1,
        gpus: [
          {
            ordinal: 0,
            name: "NVIDIA GeForce RTX 4090",
            backend: "cuda",
            vram_total: 24_000_000_000,
            vram_used: 18_000_000_000,
            gpu_utilization: 97,
          },
        ],
        system_ram: { total: 64_000_000_000, used: 21_000_000_000 },
        cpu: { cores: 16, usage_percent: 43.2 },
      }),
    );
    await flushPromises();

    const gpuCard = wrapper.get("[data-test='gpu-card']");
    expect(gpuCard.text()).toContain("CUDA");
    expect(gpuCard.text()).toContain("18.0 GB/24.0 GB");
    expect(gpuCard.get("[data-test='gpu-utilization']").text()).toBe("97%");
    const cpuCard = wrapper.get("[data-test='cpu-card']");
    expect(cpuCard.text()).toContain("16 CORES");
    expect(cpuCard.text()).toContain("43%");
    expect(wrapper.get("[data-test='ram-card']").text()).toContain("21.0 GB/64.0 GB");
  });

  it("collapses VRAM and RAM into one Memory row on a unified-memory Metal host", async () => {
    const wrapper = await mountView();
    const stream = lastStream();
    stream.options.onEvent(
      "snapshot",
      JSON.stringify({
        hostname: "halcyon",
        timestamp: 1,
        gpus: [
          {
            ordinal: 0,
            name: "Apple Metal GPU",
            backend: "metal",
            vram_total: 51_500_000_000,
            vram_used: 46_900_000_000,
            gpu_utilization: null,
          },
        ],
        system_ram: { total: 51_500_000_000, used: 46_900_000_000 },
        cpu: { cores: 16, usage_percent: 44 },
      }),
    );
    await flushPromises();

    const gpuCard = wrapper.get("[data-test='gpu-card']");
    expect(gpuCard.text()).toContain("MEMORY");
    expect(gpuCard.text()).not.toContain("VRAM");
    expect(gpuCard.text()).toContain("46.9 GB/51.5 GB");
    // The standalone RAM row would repeat the same numbers — it stays hidden,
    // while CPU keeps its own row.
    expect(wrapper.find("[data-test='ram-card']").exists()).toBe(false);
    expect(wrapper.find("[data-test='cpu-card']").exists()).toBe(true);
  });

  it("aborts the resources stream on unmount", async () => {
    const wrapper = await mountView();
    const stream = lastStream();
    expect(stream.options.signal.aborted).toBe(false);
    wrapper.unmount();
    expect(stream.options.signal.aborted).toBe(true);
  });
});

describe("HostDetailView H3 placement", () => {
  it("renders the H3 capability panel below the primary instrument sections", async () => {
    const wrapper = await mountView();
    mountedHosts.capabilities[REMOTE_ID] =
      authenticatedMiniMaxH3Capabilities() as unknown as ServerCapabilities;
    await flushPromises();

    const html = wrapper.html();
    const h3At = html.indexOf('data-test="h3-inventory"');
    expect(h3At).toBeGreaterThan(-1);
    expect(html.indexOf("TELEMETRY")).toBeLessThan(h3At);
    expect(html.indexOf("INSTALLED MODELS")).toBeLessThan(h3At);
  });
});

describe("HostDetailView storage and queue", () => {
  it("shows the models-disk card from /api/status", async () => {
    installApi({ models_disk: { total_bytes: 2_000_000_000_000, free_bytes: 500_000_000_000 } });
    const wrapper = await mountView();
    const card = wrapper.get("[data-test='storage-card']");
    expect(card.text()).toContain("500.0 GB free of 2000.0 GB");
  });

  it("hides the storage card when the server predates models_disk", async () => {
    const wrapper = await mountView();
    expect(wrapper.find("[data-test='storage-card']").exists()).toBe(false);
  });

  it("shows queue depth/capacity and loaded-model chips", async () => {
    installApi({ queue_depth: 3, queue_capacity: 8 });
    const wrapper = await mountView();
    expect(wrapper.get("[data-test='queue-depth']").text()).toBe("3/8");
    const chips = wrapper.findAll("[data-test='loaded-model-name']");
    expect(chips.map((c) => c.text())).toEqual(["flux-dev:q8"]);
  });

  it("lists the host's server queue with state codes and ownership tags", async () => {
    installApi({ queue_depth: 2, queue_capacity: 8 }, [
      {
        id: "srv-1",
        model: "flux-dev:q8",
        state: "running",
        started_at_unix_ms: Date.now() - 90_000,
        position: 0,
        gpu: 0,
      },
      {
        id: "srv-2",
        model: "z-image:q8",
        state: "queued",
        started_at_unix_ms: Date.now(),
        position: 1,
      },
    ]);
    const wrapper = await mountView();
    const rows = wrapper.findAll("[data-test='host-queue-row']");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.text()).toContain("QUEUED #1");
    expect(rows[0]!.text()).toContain("z-image:q8");
    expect(rows[1]!.text()).toContain("RUNNING · GPU 0");
    expect(rows[1]!.text()).toContain("flux-dev:q8");
    // Elapsed wall-clock for the running row (~90s → "1m 30s").
    expect(rows[1]!.text()).toMatch(/1m \d+s/);
    // Neither entry belongs to this app's generation store.
    expect(rows[0]!.text()).toContain("OTHER CLIENT");
    expect(wrapper.find("[data-test='queue-empty']").exists()).toBe(false);
  });

  it("warms the VRAM meter from the polled queue even when the status depth lags", async () => {
    // status says an empty queue; the live queue poll disagrees (a job runs).
    installApi({ queue_depth: 0, queue_capacity: 8 }, [
      {
        id: "srv-1",
        model: "flux-dev:q8",
        state: "running",
        started_at_unix_ms: Date.now() - 5_000,
        position: 0,
        gpu: 0,
      },
    ]);
    const wrapper = await mountView();
    lastStream().options.onEvent(
      "snapshot",
      JSON.stringify({
        hostname: "hal9000",
        timestamp: 1,
        gpus: [
          {
            ordinal: 0,
            name: "NVIDIA GeForce RTX 4090",
            backend: "cuda",
            vram_total: 24_000_000_000,
            vram_used: 18_000_000_000,
            gpu_utilization: 97,
          },
        ],
        system_ram: { total: 64_000_000_000, used: 21_000_000_000 },
        cpu: { cores: 16, usage_percent: 43.2 },
      }),
    );
    await flushPromises();
    expect(wrapper.get("[data-test='gpu-card']").html()).toContain("bg-safelight");
  });

  it("shows an empty queue line and a PAUSED marker from the queue snapshot", async () => {
    installApi({ queue_paused: true } as Partial<ServerStatus>);
    const wrapper = await mountView();
    expect(wrapper.get("[data-test='queue-empty']").text()).toBe("Queue is empty.");
    expect(wrapper.get("[data-test='queue-paused']").text()).toBe("PAUSED");
  });
});

describe("HostDetailView stale status responses", () => {
  it("ignores a late /api/status response from a previously viewed host", async () => {
    // hal9000 responds slowly (TCP-retry limbo); the user navigates to the
    // local host before it resolves. The late response must not populate the
    // local host's page.
    let resolveHal: ((v: unknown) => void) | null = null;
    apiJsonTo.mockImplementation((target: unknown, path: string) => {
      const base = (target as { baseUrl: string }).baseUrl;
      if (path === "/api/models") return Promise.resolve([]);
      if (path === "/api/status") {
        if (base.includes("hal9000")) {
          return new Promise((resolve) => {
            resolveHal = resolve;
          });
        }
        return Promise.resolve({
          version: "0.17.0",
          models_loaded: [],
          uptime_secs: 5,
          models_disk: { total_bytes: 1_000_000_000_000, free_bytes: 900_000_000_000 },
        });
      }
      return Promise.reject(new Error(`unexpected ${path}`));
    });
    const wrapper = await mountView(); // /hosts/hal9000-7680 — status pending

    await router.push("/hosts/local");
    await flushPromises();
    expect(wrapper.get("[data-test='storage-card']").text()).toContain(
      "900.0 GB free of 1000.0 GB",
    );

    // Seconds later, hal9000's status finally arrives — for the OLD page.
    resolveHal!({
      version: "0.9.9",
      models_loaded: [],
      uptime_secs: 1,
      queue_depth: 9,
      queue_capacity: 9,
      models_disk: { total_bytes: 2_000_000_000_000, free_bytes: 500_000_000_000 },
    });
    await flushPromises();

    const card = wrapper.get("[data-test='storage-card']");
    expect(card.text()).toContain("900.0 GB free of 1000.0 GB");
    expect(card.text()).not.toContain("500.0 GB");
  });
});

describe("HostDetailView models", () => {
  it("lists this host's installed models with family", async () => {
    const wrapper = await mountView();
    const rows = wrapper.findAll("[data-test='model-row']");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.text()).toContain("flux-dev:q8");
    expect(rows[0]!.text()).toContain("flux");
    expect(rows[1]!.text()).toContain("z-image:q8");
  });

  it("keeps a downloaded runtime_available:false row in the host inventory (Models/repair surface, never the runtime filter)", async () => {
    const h3Nvfp4: ModelEntry = {
      ...model("minimax-h3-fl2va:comfy-pruned-nvfp4", "minimax-h3"),
      runtime_available: false,
    };
    installApi({}, [], [h3Nvfp4]);
    const wrapper = await mountView(`/hosts/${REMOTE_ID}`, [h3Nvfp4]);
    const rows = wrapper.findAll("[data-test='model-row']");
    expect(rows).toHaveLength(1);
    expect(rows[0]!.text()).toContain("MiniMax H3 FL2VA · NVFP4");
  });

  it("repeats model kind and mature-content classification in the host inventory", async () => {
    const matureLora = {
      ...model("cv:8001", "flux2"),
      display_name: "After Dark Portrait Adapter",
      kind: "lora",
      modality: "image",
      nsfw: true,
    };
    installApi({}, [], [matureLora]);
    const wrapper = await mountView(`/hosts/${REMOTE_ID}`, [matureLora]);
    const row = wrapper.get("[data-test='model-row']");

    expect(row.text()).toContain("LoRA");
    expect(row.text()).toContain("18+ NSFW");
    expect(row.get("[data-test='model-table-row']").attributes("aria-label")).toContain(
      "LoRA, 18+ NSFW",
    );
  });

  it("opens the shared model detail drawer from a model row", async () => {
    const wrapper = await mountView();
    await wrapper.get("[data-test='model-row'] [data-test='row-title']").trigger("click");
    await flushPromises();
    const drawer = wrapper.get("[data-test='catalog-detail-drawer']");
    expect(drawer.text()).toContain("flux-dev:q8");
  });

  it("closes an open drawer when navigating to a different host", async () => {
    const wrapper = await mountView();
    await wrapper.get("[data-test='model-row'] [data-test='row-title']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='catalog-detail-drawer']").exists()).toBe(true);

    // The reused view must not retarget the previous host's model (and its
    // Repair action) at the next host.
    await router.push("/hosts/local");
    await flushPromises();
    expect(wrapper.find("[data-test='catalog-detail-drawer']").exists()).toBe(false);
  });

  it("shows this host's active model-download progress", async () => {
    const wrapper = await mountView();
    const stream = lastStream("/api/downloads/stream");
    expect(stream.options.target).toEqual({ baseUrl: "http://hal9000:7680", apiKey: "sekrit" });

    stream.options.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: {
          active_jobs: [
            {
              id: "pull-1",
              model: "qwen-image:q4",
              status: "active",
              files_done: 1,
              files_total: 4,
              bytes_done: 2_500_000_000,
              bytes_total: 10_000_000_000,
            },
          ],
          queued: [],
          history: [],
        },
      }),
    );
    await flushPromises();

    const tray = wrapper.get("[data-test='host-downloads']");
    expect(tray.text()).toContain("qwen-image:q4");
    expect(tray.text()).toContain("2.5 GB / 10.0 GB");
    expect(tray.get("[role='progressbar']").attributes("aria-valuenow")).toBe("25");
  });

  it("keeps live components mounted across health polls and only reopens streams for credentials", async () => {
    installApi({ models_disk: { total_bytes: 2_000_000_000_000, free_bytes: 500_000_000_000 } });
    const wrapper = await mountView();
    expect(
      apiJsonTo.mock.calls.filter(
        (call) =>
          call[1] === "/api/models" &&
          (call[0] as { baseUrl: string }).baseUrl === "http://hal9000:7680",
      ),
    ).toHaveLength(1);

    const firstResourceStream = lastStream();
    const firstDeviceStream = lastStream("/api/events");
    firstResourceStream.options.onEvent(
      "snapshot",
      JSON.stringify({
        hostname: "hal9000",
        timestamp: 1,
        gpus: [
          {
            ordinal: 0,
            name: "NVIDIA GeForce RTX 4090",
            backend: "cuda",
            vram_total: 24_000_000_000,
            vram_used: 18_000_000_000,
            gpu_utilization: 97,
          },
        ],
        system_ram: { total: 64_000_000_000, used: 21_000_000_000 },
        cpu: { cores: 16, usage_percent: 43.2 },
      }),
    );
    await flushPromises();

    const gpuCard = wrapper.get("[data-test='gpu-card']").element;
    const cpuCard = wrapper.get("[data-test='cpu-card']").element;
    const ramCard = wrapper.get("[data-test='ram-card']").element;
    const storageCard = wrapper.get("[data-test='storage-card']").element;
    const resourceStreamCount = sseCalls.filter(
      (call) => call.path === "/api/resources/stream",
    ).length;
    const remote = useHostsStore().extras[0]!;
    remote.status = "error";
    await flushPromises();

    expect(wrapper.get("[data-test='gpu-card']").element).toBe(gpuCard);
    expect(wrapper.get("[data-test='cpu-card']").element).toBe(cpuCard);
    expect(wrapper.get("[data-test='ram-card']").element).toBe(ramCard);
    expect(wrapper.get("[data-test='storage-card']").element).toBe(storageCard);
    expect(sseCalls.filter((call) => call.path === "/api/resources/stream")).toHaveLength(
      resourceStreamCount,
    );

    remote.status = "ready";
    await flushPromises();
    expect(wrapper.get("[data-test='gpu-card']").element).toBe(gpuCard);
    expect(wrapper.get("[data-test='cpu-card']").element).toBe(cpuCard);
    expect(wrapper.get("[data-test='ram-card']").element).toBe(ramCard);
    expect(wrapper.get("[data-test='storage-card']").element).toBe(storageCard);
    expect(sseCalls.filter((call) => call.path === "/api/resources/stream")).toHaveLength(
      resourceStreamCount,
    );
    // The view status request and queue snapshot run at mount and ready-flip;
    // the device-event onOpen adds one authoritative queue/device snapshot.
    // The error flip in between must add none.
    expect(apiJsonTo.mock.calls.filter((call) => call[1] === "/api/status")).toHaveLength(5);

    remote.apiKey = "rotated-key";
    await flushPromises();

    expect(firstResourceStream.options.signal.aborted).toBe(true);
    expect(firstDeviceStream.options.signal.aborted).toBe(true);
    expect(lastStream().options.target).toEqual({
      baseUrl: "http://hal9000:7680",
      apiKey: "rotated-key",
    });
    expect(lastStream("/api/downloads/stream").options.target).toEqual({
      baseUrl: "http://hal9000:7680",
      apiKey: "rotated-key",
    });
    expect(lastStream("/api/events").options.target).toEqual({
      baseUrl: "http://hal9000:7680",
      apiKey: "rotated-key",
    });

    const queueReads = apiJsonTo.mock.calls.filter((call) =>
      call[1].startsWith("/api/queue"),
    ).length;
    lastStream("/api/events").options.onEvent(
      "message",
      JSON.stringify({ type: "device_state_changed" }),
    );
    await flushPromises();
    expect(apiJsonTo.mock.calls.filter((call) => call[1].startsWith("/api/queue"))).toHaveLength(
      queueReads + 1,
    );
  });
});

describe("HostDetailView queue drawer", () => {
  const runningEntry = (metadata?: Record<string, unknown>): WireQueueEntry => ({
    id: "srv-1",
    model: "qwen-image:bf16",
    state: "running",
    started_at_unix_ms: Date.now() - 60_000,
    position: 0,
    gpu: 3,
    ...(metadata ? { metadata } : {}),
  });

  const wireMetadata = {
    prompt: "a lighthouse at dusk",
    negative_prompt: null,
    model: "qwen-image:bf16",
    seed: 0,
    steps: 28,
    guidance: 3.5,
    width: 1328,
    height: 1328,
  };

  it("opens a queue row's info drawer and reuses its settings in Create", async () => {
    installApi({}, [runningEntry(wireMetadata)]);
    const wrapper = await mountView();
    await wrapper.get("[data-test='host-queue-row']").trigger("click");

    const drawer = wrapper.get("[data-test='queue-entry-drawer']");
    expect(drawer.text()).toContain("qwen-image:bf16");
    expect(drawer.get("[data-test='queue-detail-prompt']").text()).toBe("a lighthouse at dusk");
    expect(drawer.text()).toContain("1328×1328");
    // Seed 0 on the wire = not pinned.
    expect(drawer.text()).toContain("Random");
    expect(drawer.text()).toContain("Another client");

    await drawer.get("[data-test='queue-detail-reuse']").trigger("click");
    await flushPromises();
    const prefill = useComposerStore().prefill as { metadata: Record<string, unknown> };
    expect(prefill.metadata).toMatchObject({ prompt: "a lighthouse at dusk", seed: null });
    expect(useComposerStore().prefill).toMatchObject({
      queueSelection: { hostId: REMOTE_ID, jobId: "srv-1", running: true },
    });
    expect(router.currentRoute.value.path).toBe("/create");
    // Reusing settings closes the drawer.
    expect(wrapper.find("[data-test='queue-entry-drawer']").exists()).toBe(false);
  });

  it("keeps an explicitly pinned seed 0 instead of restoring it as random", async () => {
    installApi({}, [{ ...runningEntry({ ...wireMetadata }), seed_pinned: true }]);
    const wrapper = await mountView();
    await wrapper.get("[data-test='host-queue-row']").trigger("click");
    const drawer = wrapper.get("[data-test='queue-entry-drawer']");
    expect(drawer.text()).not.toContain("Random");

    await drawer.get("[data-test='queue-detail-reuse']").trigger("click");
    await flushPromises();
    const prefill = useComposerStore().prefill as { metadata: Record<string, unknown> };
    expect(prefill.metadata).toMatchObject({ seed: 0 });
  });

  it("still reports the job when the durable listing has not loaded its request", async () => {
    installApi({}, [runningEntry()]);
    const wrapper = await mountView();
    await wrapper.get("[data-test='host-queue-row']").trigger("click");

    const drawer = wrapper.get("[data-test='queue-entry-drawer']");
    // The gap is the wire's, not an old host's: never tell anyone to upgrade.
    expect(drawer.text()).not.toMatch(/upgrade/i);
    expect(drawer.get("[data-test='queue-detail-settings-notice']").text()).toMatch(
      /once this machine loads the job/i,
    );
    // Everything the durable projection DOES carry is still rendered.
    expect(drawer.text()).toContain("RUNNING · GPU 3");
    expect(drawer.get("[data-test='queue-detail-facts']").text()).toContain("Elapsed");
    expect(drawer.get("[data-test='queue-detail-reuse']").attributes("disabled")).toBeDefined();
  });
});

describe("HostDetailView loaded-chip unload", () => {
  it("unloads on the confirming second click and hides the chip until the poll confirms", async () => {
    const wrapper = await mountView();
    expect(wrapper.findAll("[data-test='loaded-model-chip']")).toHaveLength(1);

    // First click only arms the inline confirm.
    const chip = wrapper.get("[data-test='unload-chip']");
    await chip.trigger("click");
    await flushPromises();
    expect(unloadModel).not.toHaveBeenCalled();
    expect(chip.text()).toBe("Unload?");

    await chip.trigger("click");
    await flushPromises();
    expect(unloadModel).toHaveBeenCalledWith("flux-dev:q8", {
      baseUrl: "http://hal9000:7680",
      apiKey: "sekrit",
    });
    // Optimistically hidden — telemetry still lists it until the next poll.
    expect(wrapper.findAll("[data-test='loaded-model-chip']")).toHaveLength(0);
  });

  it("keeps the chip and surfaces an error toast when the unload fails", async () => {
    unloadModel.mockRejectedValueOnce(new Error("model is busy"));
    const wrapper = await mountView();
    const chip = wrapper.get("[data-test='unload-chip']");
    await chip.trigger("click");
    await chip.trigger("click");
    await flushPromises();
    expect(wrapper.findAll("[data-test='loaded-model-chip']")).toHaveLength(1);
  });
});

describe("HostDetailView layout", () => {
  it("uses the full workspace width instead of preserving a fixed desktop cap", async () => {
    const wrapper = await mountView();
    const content = wrapper.get("[data-test='host-detail-content']");
    expect(content.classes()).toContain("w-full");
    expect(content.classes().some((name) => name.startsWith("max-w-"))).toBe(false);
    expect(
      wrapper
        .get("[data-test='host-model-column']")
        .classes()
        .some((name) => name.includes("max-w-")),
    ).toBe(false);
  });

  it("shows uptime from /api/status in the telemetry header", async () => {
    installApi({ uptime_secs: 200_000 });
    const wrapper = await mountView();
    expect(wrapper.get("[data-test='host-uptime']").text()).toBe("UP 2d 7h");
  });

  it("renders the models-disk meter inside the telemetry panel, not a separate section", async () => {
    installApi({ models_disk: { total_bytes: 2_000_000_000_000, free_bytes: 500_000_000_000 } });
    const wrapper = await mountView();
    const panel = wrapper.get("[data-test='telemetry-panel']");
    expect(panel.find("[data-test='storage-card']").exists()).toBe(true);
    expect(panel.find("[data-test='gpu-card']").exists()).toBe(true);
  });

  it("labels resident models LOADED so they can't read as queued jobs", async () => {
    const wrapper = await mountView();
    expect(wrapper.get("[data-test='loaded-label']").text()).toBe("LOADED");
    const chips = wrapper.findAll("[data-test='loaded-model-name']");
    expect(chips.map((c) => c.text())).toEqual(["flux-dev:q8"]);
  });

  it("uses the installed model title for an opaque loaded-model id", async () => {
    const wrapper = await mountView();
    useHostModelsStore().byHost[REMOTE_ID]!.entries = [
      {
        ...model("cv:1759168", "sdxl"),
        display_name: "Juggernaut XL - Ragnarok",
      },
    ];
    useHostsStore().telemetry[REMOTE_ID]!.modelsLoaded = ["cv:1759168"];
    await flushPromises();

    expect(wrapper.get("[data-test='loaded-model-name']").text()).toBe("Juggernaut XL - Ragnarok");
    expect(wrapper.text()).not.toContain("cv:1759168");
  });

  it("places the downloads tray in its own card below the telemetry panel", async () => {
    const wrapper = await mountView();
    const stream = lastStream("/api/downloads/stream");
    stream.options.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: {
          active_jobs: [
            {
              id: "pull-1",
              model: "qwen-image:q4",
              status: "active",
              files_done: 1,
              files_total: 4,
              bytes_done: 2_500_000_000,
              bytes_total: 10_000_000_000,
            },
          ],
          queued: [],
          history: [],
        },
      }),
    );
    await flushPromises();
    const telemetry = wrapper.get("[data-test='telemetry-panel']").element;
    const tray = wrapper.get("[data-test='host-downloads']").element;
    expect(telemetry.compareDocumentPosition(tray) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("summarizes installed model count and total size in the models header", async () => {
    const wrapper = await mountView();
    expect(wrapper.get("[data-test='models-summary']").text()).toBe("2 · 24.8 GB");
  });
});

describe("HostDetailView forget", () => {
  it("confirms with blunt copy, then drops the host and returns to Machines", async () => {
    const wrapper = await mountView();
    await wrapper.get("[data-test='forget-host']").trigger("click");
    expect(forgetRemoteHost).not.toHaveBeenCalled();

    // The confirm dialog teleports to <body>; it carries the §11 copy.
    const dialog = document.querySelector("[data-test='confirm-dialog']");
    expect(dialog?.textContent).toContain("Forget studio?");
    expect(dialog?.textContent).toContain("Its API key is discarded.");

    (document.querySelector("[data-test='confirm-accept']") as HTMLButtonElement).click();
    await flushPromises();
    expect(forgetRemoteHost).toHaveBeenCalledWith(REMOTE_ID);
    expect(useHostsStore().extras).toHaveLength(0);
    expect(router.currentRoute.value.path).toBe("/machines");
    document.body.innerHTML = "";
  });
});

describe("HostDetailView storage (Library trash)", () => {
  const TARGET = { baseUrl: "http://hal9000:7680", apiKey: "sekrit" };
  const trashCapable: ServerCapabilities = {
    gallery: { can_delete: true, trash: { enabled: true, retention_days: 30 }, organize: true },
  };

  it("hides the card when the host's capabilities lack gallery.trash", async () => {
    const wrapper = await mountView(undefined, undefined, {
      devices: [],
      capabilities: { gallery: { can_delete: true } },
    });
    expect(wrapper.find("[data-test='host-storage']").exists()).toBe(false);
    expect(fetchHostConfigKey).not.toHaveBeenCalled();
    expect(listTrash).not.toHaveBeenCalled();
  });

  it("shows that host's retention and trash count, read from the host itself", async () => {
    fetchHostConfigKey.mockResolvedValue({
      key: "gallery.trash_retention_days",
      value: 7,
      source: "db",
    });
    listTrash.mockResolvedValue([
      { filename: "a.png" },
      { filename: "b.png" },
      { filename: "c.mp4" },
    ]);
    const wrapper = await mountView(undefined, undefined, {
      devices: [],
      capabilities: trashCapable,
    });
    expect(wrapper.find("[data-test='host-storage']").exists()).toBe(true);
    expect(fetchHostConfigKey).toHaveBeenCalledWith(
      TARGET,
      "gallery.trash_retention_days",
      expect.any(AbortSignal),
    );
    expect(listTrash).toHaveBeenCalledWith(TARGET, expect.any(AbortSignal));
    const select = wrapper.get<HTMLSelectElement>("[data-test='host-trash-retention']");
    expect(select.element.value).toBe("7");
    const labels = select.findAll("option").map((o) => o.text());
    expect(labels).toEqual(["1 day", "7 days", "30 days", "90 days", "1 year", "Forever"]);
    expect(wrapper.get("[data-test='host-trash-count']").text()).toContain("3");
  });

  it("keeps an off-menu retention visible instead of lying about it", async () => {
    fetchHostConfigKey.mockResolvedValue({
      key: "gallery.trash_retention_days",
      value: 14,
      source: "db",
    });
    const wrapper = await mountView(undefined, undefined, {
      devices: [],
      capabilities: trashCapable,
    });
    const select = wrapper.get<HTMLSelectElement>("[data-test='host-trash-retention']");
    expect(select.element.value).toBe("14");
    expect(select.findAll("option").map((o) => o.text())).toContain("14 days");
  });

  it("writes a new retention to THAT host's /api/config, never the primary's", async () => {
    // The host's own row reflects the write on the re-read that follows.
    setHostConfigKey.mockImplementation(async (_t: unknown, key: string, value: number) => {
      fetchHostConfigKey.mockResolvedValue({ key, value, source: "db" });
      return new Response(null, { status: 200 });
    });
    const wrapper = await mountView(undefined, undefined, {
      devices: [],
      capabilities: trashCapable,
    });
    const select = wrapper.get<HTMLSelectElement>("[data-test='host-trash-retention']");
    await select.setValue("0");
    await flushPromises();
    expect(setHostConfigKey).toHaveBeenCalledWith(TARGET, "gallery.trash_retention_days", 0);
    expect(select.element.value).toBe("0");
  });

  it("locks the select when the host's env var owns the value", async () => {
    fetchHostConfigKey.mockResolvedValue({
      key: "gallery.trash_retention_days",
      value: 90,
      source: "env",
      env_var: "MOLD_GALLERY_TRASH_RETENTION_DAYS",
    });
    const wrapper = await mountView(undefined, undefined, {
      devices: [],
      capabilities: trashCapable,
    });
    const select = wrapper.get<HTMLSelectElement>("[data-test='host-trash-retention']");
    expect(select.attributes("disabled")).toBeDefined();
    expect(select.attributes("title")).toContain("MOLD_GALLERY_TRASH_RETENTION_DAYS");
  });

  it("empties the trash only after the plain shared confirm naming host and count", async () => {
    listTrash.mockResolvedValue([{ filename: "a.png" }, { filename: "b.png" }]);
    emptyTrash.mockImplementation(async () => {
      listTrash.mockResolvedValue([]);
      return { purged: 2 };
    });
    const wrapper = await mountView(undefined, undefined, {
      devices: [],
      capabilities: trashCapable,
    });
    await wrapper.get("[data-test='host-empty-trash']").trigger("click");
    expect(emptyTrash).not.toHaveBeenCalled();

    const dialog = document.querySelector("[data-test='confirm-dialog']");
    expect(dialog?.textContent).toContain("Empty trash?");
    expect(dialog?.textContent).toContain(
      "Delete 2 prints in the trash on hal9000 forever? This can't be undone.",
    );
    // No typed-phrase gate anywhere (design amendment).
    expect(dialog?.querySelector("input")).toBeNull();

    (document.querySelector("[data-test='confirm-accept']") as HTMLButtonElement).click();
    await flushPromises();
    expect(emptyTrash).toHaveBeenCalledWith(TARGET);
    expect(wrapper.get("[data-test='host-trash-count']").text()).toContain("0");
    document.body.innerHTML = "";
  });

  it("disables Empty trash when the trash is empty", async () => {
    const wrapper = await mountView(undefined, undefined, {
      devices: [],
      capabilities: trashCapable,
    });
    expect(wrapper.get("[data-test='host-empty-trash']").attributes("disabled")).toBeDefined();
  });
});
