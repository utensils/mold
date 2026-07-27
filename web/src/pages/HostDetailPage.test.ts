import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick, reactive, ref } from "vue";
import type {
  HostCapabilities,
  HostStatus,
} from "../components/machines/hostClient";
import type {
  DownloadsListingWire,
  ModelInfoExtended,
  QueueEntry,
  ResourceSnapshot,
} from "../types";
import { addHost, getHost, updateHost } from "../lib/hostRegistry";
import HostDetailPage from "./HostDetailPage.vue";
import type { DeviceInfo, DeviceListResponse } from "@studio/api/devices";

// Mutable fixtures the hostClient mock reads at call time.
let poll: {
  status: ReturnType<typeof ref<HostStatus | null>>;
  devices: ReturnType<typeof ref<DeviceInfo[] | null>>;
  deviceState: ReturnType<typeof ref<DeviceListResponse | null>>;
  resources: ReturnType<typeof ref<ResourceSnapshot | null>>;
  online: ReturnType<typeof ref<boolean>>;
  lastSeen: ReturnType<typeof ref<number | null>>;
  error: ReturnType<typeof ref<string | null>>;
  loading: ReturnType<typeof ref<boolean>>;
  refresh: ReturnType<typeof vi.fn>;
  stop: ReturnType<typeof vi.fn>;
};
let caps: HostCapabilities;
let queueEntries: QueueEntry[];
let models: ModelInfoExtended[];
let downloadsListing: DownloadsListingWire;

const cancelQueueJob = vi.fn().mockResolvedValue(undefined);
const setQueueJobLane = vi.fn().mockResolvedValue(undefined);
const moveQueueJob = vi.fn().mockResolvedValue(undefined);
const pauseHostQueue = vi.fn().mockResolvedValue(undefined);
const resumeHostQueue = vi.fn().mockResolvedValue(undefined);
const cancelAllHostQueue = vi.fn().mockResolvedValue(undefined);
const routerPush = vi.fn().mockResolvedValue(undefined);
const requestConfirm = vi.hoisted(() => vi.fn(async () => true));
const requestText = vi.hoisted(() => vi.fn(async () => "Render box"));
const setDeviceEnabled = vi.hoisted(() =>
  vi.fn(async (_target: unknown, _id: string, enabled: boolean) =>
    makeDevice(0, {
      desired_enabled: enabled,
      admin_state: enabled ? "enabled" : "draining",
    }),
  ),
);

vi.mock("@studio/api/devices", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/devices")>()),
  setDeviceEnabled,
}));
const subscribeToDeviceSnapshots = vi.hoisted(() => vi.fn());
const hostCapabilitiesCall = vi.hoisted(() => vi.fn());
const hostQueueCall = vi.hoisted(() => vi.fn());

vi.mock("../lib/toasts", () => ({
  toast: vi.fn(),
  requestConfirm,
  requestText,
}));
vi.mock("../lib/deviceEvents", () => ({ subscribeToDeviceSnapshots }));

const routeHolder = reactive({ id: "origin" });

vi.mock("vue-router", () => ({
  useRoute: () => ({ params: routeHolder }),
  useRouter: () => ({ push: routerPush }),
  RouterLink: {
    name: "RouterLink",
    props: ["to"],
    template: "<a><slot /></a>",
  },
}));

vi.mock("../components/machines/hostClient", () => ({
  useHostPoll: () => poll,
  hostCapabilities: (...args: unknown[]) => hostCapabilitiesCall(...args),
  hostQueue: (...args: unknown[]) => hostQueueCall(...args),
  setHostDeviceEnabled: () =>
    Promise.resolve({ devices: poll.devices.value ?? [], plan_version: 0 }),
  hostModels: () => Promise.resolve(models),
  hostDownloads: () => Promise.resolve(downloadsListing),
  // Wrappers defer reading the vi.fns until call time (the factory is hoisted
  // above their declarations).
  cancelQueueJob: (...a: unknown[]) => cancelQueueJob(...a),
  setQueueJobLane: (...a: unknown[]) => setQueueJobLane(...a),
  moveQueueJob: (...a: unknown[]) => moveQueueJob(...a),
  pauseHostQueue: (...a: unknown[]) => pauseHostQueue(...a),
  resumeHostQueue: (...a: unknown[]) => resumeHostQueue(...a),
  cancelAllHostQueue: (...a: unknown[]) => cancelAllHostQueue(...a),
}));

function makeStatus(over: Partial<HostStatus> = {}): HostStatus {
  return {
    version: "0.16.0",
    models_loaded: [],
    busy: false,
    uptime_secs: 3_600 + 720,
    queue_depth: 2,
    gpus: [
      {
        ordinal: 0,
        name: "RTX 4090",
        vram_total_bytes: 24_000_000_000,
        vram_used_bytes: 12_000_000_000,
        state: "idle",
      },
    ],
    models_disk: { total_bytes: 994_000_000_000, free_bytes: 213_000_000_000 },
    ...over,
  };
}

function makeResources(over: Partial<ResourceSnapshot> = {}): ResourceSnapshot {
  return {
    hostname: "studio",
    timestamp: 0,
    system_ram: {
      total: 64_000_000_000,
      used: 16_000_000_000,
      used_by_mold: 8_000_000_000,
      used_by_other: 8_000_000_000,
    },
    cpu: { cores: 16, usage_percent: 32 },
    gpus: [
      {
        ordinal: 0,
        name: "RTX 4090",
        backend: "cuda",
        vram_total: 24_000_000_000,
        vram_used: 12_000_000_000,
        vram_used_by_mold: null,
        vram_used_by_other: null,
        gpu_utilization: 55,
      },
    ],
    ...over,
  };
}

function makeDevice(
  ordinal: number,
  overrides: Partial<DeviceInfo> = {},
): DeviceInfo {
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
      total_bytes: 24_000_000_000,
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

function queued(id: string, position: number): QueueEntry {
  return {
    id,
    model: `flux-${id}`,
    state: "queued",
    started_at_unix_ms: 0,
    position,
  };
}

let wrapper: ReturnType<typeof mount> | null = null;
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

async function mountDetail() {
  wrapper = mount(HostDetailPage);
  await flushPromises();
  await nextTick();
  return wrapper;
}

beforeEach(() => {
  localStorage.clear();
  const host = addHost({
    url: "192.168.1.20:7680",
    name: "Studio",
    apiKey: "secret",
  });
  routeHolder.id = host.id;
  caps = {
    queue: { can_reorder: false },
    devices: { available: true, lifecycle: true, restart_enable: false },
    dispatch: {
      active_mode: "v2",
      v2_authoritative: true,
      observes_v2_decisions: false,
    },
  };
  queueEntries = [];
  models = [];
  downloadsListing = { active: null, active_jobs: [], queued: [], history: [] };
  cancelQueueJob.mockClear();
  setQueueJobLane.mockClear();
  moveQueueJob.mockClear();
  pauseHostQueue.mockClear();
  resumeHostQueue.mockClear();
  cancelAllHostQueue.mockClear();
  setDeviceEnabled.mockClear();
  routerPush.mockClear();
  requestConfirm.mockReset().mockResolvedValue(true);
  requestText.mockReset().mockResolvedValue("Render box");
  subscribeToDeviceSnapshots.mockClear();
  hostCapabilitiesCall
    .mockReset()
    .mockImplementation(() => Promise.resolve(caps));
  hostQueueCall
    .mockReset()
    .mockImplementation(() => Promise.resolve({ entries: queueEntries }));
  poll = {
    status: ref<HostStatus | null>(makeStatus()),
    devices: ref<DeviceInfo[] | null>(null),
    deviceState: ref<DeviceListResponse | null>(null),
    resources: ref<ResourceSnapshot | null>(makeResources()),
    online: ref(true),
    lastSeen: ref<number | null>(Date.now()),
    error: ref<string | null>(null),
    loading: ref(false),
    refresh: vi.fn(),
    stop: vi.fn(),
  };
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

describe("HostDetailPage — telemetry", () => {
  it("subscribes with the viewed host key and refreshes authoritative state", async () => {
    const host = getHost(routeHolder.id)!;
    updateHost(routeHolder.id, { apiKey: "host-key" });
    await mountDetail();

    expect(subscribeToDeviceSnapshots).toHaveBeenCalledTimes(1);
    const [target, _signal, refresh] =
      subscribeToDeviceSnapshots.mock.calls[0]!;
    expect(target).toEqual({ baseUrl: host.url, apiKey: "host-key" });
    refresh();
    expect(poll.refresh).toHaveBeenCalledTimes(1);
  });

  it("starts queue refresh and SSE without waiting for a stalled capability probe", async () => {
    hostCapabilitiesCall.mockImplementation(() => new Promise(() => {}));

    wrapper = mount(HostDetailPage);
    await nextTick();
    await Promise.resolve();

    expect(hostQueueCall).toHaveBeenCalledTimes(1);
    expect(subscribeToDeviceSnapshots).toHaveBeenCalledTimes(1);
  });

  it("threads the active session signal through interval and SSE reloads", async () => {
    vi.useFakeTimers();
    try {
      wrapper = mount(HostDetailPage);
      await nextTick();
      await Promise.resolve();
      const sessionSignal = hostQueueCall.mock.calls[0]![1] as AbortSignal;
      const refresh = subscribeToDeviceSnapshots.mock
        .calls[0]![2] as () => void;

      hostQueueCall.mockClear();
      await vi.advanceTimersByTimeAsync(4_000);
      expect(hostQueueCall).toHaveBeenCalledWith(
        expect.objectContaining({ id: routeHolder.id }),
        sessionSignal,
      );

      hostQueueCall.mockClear();
      refresh();
      await Promise.resolve();
      expect(hostQueueCall).toHaveBeenCalledWith(
        expect.objectContaining({ id: routeHolder.id }),
        sessionSignal,
      );
      expect(sessionSignal.aborted).toBe(false);
    } finally {
      wrapper?.unmount();
      wrapper = null;
      vi.useRealTimers();
    }
  });

  it("ignores an older same-host queue response after a newer event refresh", async () => {
    const older = deferred<{ entries: QueueEntry[] }>();
    const newer = deferred<{ entries: QueueEntry[] }>();
    hostQueueCall
      .mockImplementationOnce(() => older.promise)
      .mockImplementationOnce(() => newer.promise);

    wrapper = mount(HostDetailPage);
    await nextTick();
    await Promise.resolve();
    const refresh = subscribeToDeviceSnapshots.mock.calls[0]![2] as () => void;
    refresh();
    await Promise.resolve();

    newer.resolve({ entries: [queued("newer", 0)] });
    await flushPromises();
    older.resolve({ entries: [queued("older", 0)] });
    await flushPromises();

    expect(wrapper.text()).toContain("flux-newer");
    expect(wrapper.text()).not.toContain("flux-older");
  });

  it("cancels and rebinds capabilities, queue polling, and SSE on route changes", async () => {
    const first = getHost(routeHolder.id)!;
    updateHost(first.id, { apiKey: "first-key" });
    const second = addHost({
      url: "192.168.1.21:7680",
      name: "Second",
      apiKey: "second-key",
    });
    await mountDetail();
    const firstSignal = subscribeToDeviceSnapshots.mock
      .calls[0]![1] as AbortSignal;
    const firstCapabilitySignal = hostCapabilitiesCall.mock
      .calls[0]![1] as AbortSignal;
    const firstQueueSignal = hostQueueCall.mock.calls[0]![1] as AbortSignal;

    routeHolder.id = second.id;
    await nextTick();
    await flushPromises();

    expect(firstSignal.aborted).toBe(true);
    expect(firstCapabilitySignal.aborted).toBe(true);
    expect(firstQueueSignal.aborted).toBe(true);
    expect(hostCapabilitiesCall).toHaveBeenLastCalledWith(
      expect.objectContaining({
        id: second.id,
        url: second.url,
        apiKey: "second-key",
      }),
      expect.any(AbortSignal),
    );
    expect(hostQueueCall).toHaveBeenLastCalledWith(
      expect.objectContaining({ id: second.id, apiKey: "second-key" }),
      expect.any(AbortSignal),
    );
    expect(subscribeToDeviceSnapshots).toHaveBeenCalledTimes(2);
    expect(subscribeToDeviceSnapshots.mock.calls[1]![0]).toEqual({
      baseUrl: second.url,
      apiKey: "second-key",
    });
  });

  it("cancels and rebinds the same host when its URL or API key rotates", async () => {
    const first = getHost(routeHolder.id)!;
    updateHost(first.id, { apiKey: "old-key" });
    await mountDetail();
    const firstSignal = subscribeToDeviceSnapshots.mock
      .calls[0]![1] as AbortSignal;
    const firstCapabilitySignal = hostCapabilitiesCall.mock
      .calls[0]![1] as AbortSignal;
    const firstQueueSignal = hostQueueCall.mock.calls[0]![1] as AbortSignal;

    updateHost(first.id, {
      url: "http://render-box.internal:7680",
      apiKey: "rotated-key",
    });
    await nextTick();
    await flushPromises();

    expect(firstSignal.aborted).toBe(true);
    expect(firstCapabilitySignal.aborted).toBe(true);
    expect(firstQueueSignal.aborted).toBe(true);
    expect(hostCapabilitiesCall).toHaveBeenLastCalledWith(
      expect.objectContaining({
        id: first.id,
        url: "http://render-box.internal:7680",
        apiKey: "rotated-key",
      }),
      expect.any(AbortSignal),
    );
    expect(hostQueueCall).toHaveBeenLastCalledWith(
      expect.objectContaining({
        id: first.id,
        url: "http://render-box.internal:7680",
        apiKey: "rotated-key",
      }),
      expect.any(AbortSignal),
    );
    expect(subscribeToDeviceSnapshots).toHaveBeenCalledTimes(2);
    expect(subscribeToDeviceSnapshots.mock.calls[1]![0]).toEqual({
      baseUrl: "http://render-box.internal:7680",
      apiKey: "rotated-key",
    });
  });

  it("aborts the active host session and SSE subscription on unmount", async () => {
    const w = await mountDetail();
    const capabilitySignal = hostCapabilitiesCall.mock
      .calls[0]![1] as AbortSignal;
    const queueSignal = hostQueueCall.mock.calls[0]![1] as AbortSignal;
    const sseSignal = subscribeToDeviceSnapshots.mock
      .calls[0]![1] as AbortSignal;

    w.unmount();
    wrapper = null;

    expect(capabilitySignal.aborted).toBe(true);
    expect(queueSignal.aborted).toBe(true);
    expect(sseSignal.aborted).toBe(true);
  });

  it("maps the resource snapshot into the telemetry card", async () => {
    const w = await mountDetail();
    expect(w.get('[data-test="machine-detail-title"]').text()).toBe("Studio");
    expect(w.get('[data-test="telemetry-gpu"]').text()).toContain(
      "RTX 4090 · cuda",
    );
    expect(w.get('[data-test="telemetry-load"]').text()).toBe("55%");
    expect(w.get('[data-test="telemetry-mem"]').text()).toBe("50%");
    expect(w.get('[data-test="telemetry-cpu"]').text()).toBe("32%");
    expect(w.get('[data-test="telemetry-ram"]').text()).toBe("25%");
    expect(w.get('[data-test="telemetry-queue"]').text()).toBe("2");
    expect(w.get('[data-test="telemetry-uptime"]').text()).toBe("1h 12m");
    expect(w.get('[data-test="telemetry-storage"]').text()).toContain(
      "free of",
    );
  });

  it("renders em-dash fallbacks for metrics the host does not expose", async () => {
    poll.status.value = {
      version: "0.16.0",
      models_loaded: [],
      busy: false,
      uptime_secs: 5,
    };
    poll.resources.value = null;
    const w = await mountDetail();
    expect(w.get('[data-test="telemetry-gpu"]').text()).toBe("—");
    expect(w.get('[data-test="telemetry-load"]').text()).toBe("—");
    expect(w.get('[data-test="telemetry-mem"]').text()).toBe("—");
    expect(w.get('[data-test="telemetry-cpu"]').text()).toBe("—");
    expect(w.get('[data-test="telemetry-ram"]').text()).toBe("—");
    expect(w.get('[data-test="telemetry-queue"]').text()).toBe("—");
  });

  it("shows a not-found card for an unknown host id", async () => {
    routeHolder.id = "ghost-host";
    const w = await mountDetail();
    expect(w.find('[data-test="detail-not-found"]').exists()).toBe(true);
  });
});

describe("HostDetailPage — queue", () => {
  it("wires advertised pause and cancel-all controls to the viewed host", async () => {
    caps = {
      queue: { can_pause: true, can_cancel_all: true, can_reorder: false },
    };
    queueEntries = [queued("a", 0)];
    const w = await mountDetail();

    await w.get('[data-test="pause-toggle"]').trigger("click");
    await flushPromises();
    expect(pauseHostQueue).toHaveBeenCalledWith(
      expect.objectContaining({ id: routeHolder.id }),
    );

    await w.get('[data-test="cancel-all"]').trigger("click");
    await flushPromises();
    expect(cancelAllHostQueue).toHaveBeenCalledWith(
      expect.objectContaining({ id: routeHolder.id }),
    );
  });

  it("hides reorder controls when the host lacks can_reorder", async () => {
    caps = { queue: { can_reorder: false } };
    queueEntries = [queued("a", 0), queued("b", 1)];
    const w = await mountDetail();
    expect(w.findAll('[data-test="queue-row"]')).toHaveLength(2);
    expect(w.find('[data-test="queue-up"]').exists()).toBe(false);
    expect(w.find('[data-test="queue-down"]').exists()).toBe(false);
  });

  it("shows reorder controls when the host advertises can_reorder", async () => {
    caps = { queue: { can_reorder: true } };
    queueEntries = [queued("a", 0), queued("b", 1)];
    const w = await mountDetail();
    expect(w.find('[data-test="queue-up"]').exists()).toBe(true);
    expect(w.find('[data-test="queue-down"]').exists()).toBe(true);
  });

  it("reorders a job through the host client to an absolute position", async () => {
    caps = { queue: { can_reorder: true } };
    queueEntries = [queued("a", 0), queued("b", 1)];
    const w = await mountDetail();
    // Second row's "up" moves it to position 0.
    const downButtons = w.findAll('[data-test="queue-down"]');
    await downButtons[0]!.trigger("click");
    expect(moveQueueJob).toHaveBeenCalledWith(
      expect.objectContaining({ id: routeHolder.id }),
      "a",
      1,
    );
  });

  it("cancels a queued job through the host client", async () => {
    queueEntries = [queued("a", 0)];
    const w = await mountDetail();
    await w.get('[data-test="queue-cancel"]').trigger("click");
    expect(cancelQueueJob).toHaveBeenCalledWith(
      expect.objectContaining({ id: routeHolder.id }),
      "a",
    );
  });

  it("renders a per-GPU lane selector only when the host has multiple GPUs", async () => {
    queueEntries = [queued("a", 0)];
    poll.status.value = makeStatus({
      gpus: [
        {
          ordinal: 0,
          name: "RTX 4090",
          vram_total_bytes: 24_000_000_000,
          vram_used_bytes: 0,
          state: "idle",
        },
        {
          ordinal: 1,
          name: "RTX 4090",
          vram_total_bytes: 24_000_000_000,
          vram_used_bytes: 0,
          state: "idle",
        },
      ],
    });
    const w = await mountDetail();
    expect(w.find('[data-test="queue-lane"]').exists()).toBe(true);
  });

  it("preserves non-contiguous schedulable ordinals in queue lane options", async () => {
    queueEntries = [queued("a", 0)];
    poll.devices.value = [makeDevice(1), makeDevice(3)];

    const w = await mountDetail();
    const options = w
      .get('[data-test="queue-lane"]')
      .findAll("option")
      .map((option) => option.attributes("value"));

    expect(options).toEqual(["", "1", "3"]);
  });

  it("does not build queue lanes from explicitly non-routable workers", async () => {
    queueEntries = [queued("a", 0)];
    poll.status.value = makeStatus({
      gpu_info: {
        name: "excluded GPU",
        vram_total_mb: 24576,
        vram_used_mb: 0,
      },
      gpus: [
        {
          ordinal: 0,
          name: "excluded GPU 0",
          vram_total_bytes: 24_000_000_000,
          vram_used_bytes: 0,
          state: "idle",
        },
        {
          ordinal: 1,
          name: "excluded GPU 1",
          vram_total_bytes: 24_000_000_000,
          vram_used_bytes: 0,
          state: "idle",
        },
      ],
    });
    poll.devices.value = [
      makeDevice(0, {
        admin_state: "startup_excluded",
        desired_enabled: false,
        schedulable: false,
      }),
      makeDevice(1, {
        admin_state: "disabled",
        desired_enabled: false,
        schedulable: false,
      }),
    ];

    const w = await mountDetail();

    expect(w.find('[data-test="queue-lane"]').exists()).toBe(false);
  });

  it("renders all eight devices and their lifecycle states", async () => {
    poll.devices.value = Array.from({ length: 8 }, (_, ordinal) =>
      makeDevice(ordinal, {
        desired_enabled: ordinal < 6,
        admin_state:
          ordinal === 5
            ? "draining"
            : ordinal === 6
              ? "disabled"
              : ordinal === 7
                ? "starting"
                : "enabled",
        schedulable: ordinal < 5,
      }),
    );

    const w = await mountDetail();

    expect(w.findAll('[data-test="device-card"]')).toHaveLength(8);
    expect(w.text()).toContain("Finishing current work");
    expect(w.text()).toContain("disabled");
    expect(w.text()).toContain("Starting");
  });

  it("mutates the owning host with its API key and refreshes device state", async () => {
    poll.devices.value = [makeDevice(0)];
    const w = await mountDetail();

    await w.get('[data-test="device-toggle-0"]').trigger("click");
    await flushPromises();

    expect(setDeviceEnabled).toHaveBeenCalledWith(
      expect.objectContaining({
        baseUrl: expect.any(String),
        apiKey: "secret",
      }),
      "cuda:0",
      false,
    );
    expect(poll.refresh).toHaveBeenCalled();
  });

  it("gates legacy live controls and offers only restart recovery", async () => {
    caps = {
      queue: { can_reorder: false },
      devices: { available: true, lifecycle: false, restart_enable: true },
      dispatch: {
        active_mode: "legacy",
        v2_authoritative: false,
        observes_v2_decisions: false,
      },
    };
    poll.devices.value = [
      makeDevice(0),
      makeDevice(1, {
        desired_enabled: false,
        admin_state: "disabled",
        schedulable: false,
      }),
      makeDevice(2, {
        desired_enabled: true,
        restart_required: true,
        health: "unavailable",
        schedulable: false,
        unschedulable_reason: "device_unavailable",
      }),
    ];
    const w = await mountDetail();
    expect(
      w.get('[data-test="device-toggle-0"]').attributes("disabled"),
    ).toBeDefined();
    const recovery = w.get('[data-test="device-toggle-1"]');
    expect(recovery.text()).toBe("Enable on restart");
    await recovery.trigger("click");
    await flushPromises();
    expect(setDeviceEnabled).toHaveBeenCalledWith(
      expect.any(Object),
      "cuda:1",
      true,
    );
    const pending = w.get('[data-test="device-toggle-2"]');
    expect(pending.text()).toBe("Enabled on restart");
    expect(pending.attributes("disabled")).toBeDefined();
    expect(w.findAll('[data-test="device-card"]')[2]!.text()).toContain(
      "Restart required",
    );
    expect(w.findAll('[data-test="device-card"]')[2]!.text()).not.toContain(
      "unavailable",
    );
  });
});

describe("HostDetailPage — saved host actions", () => {
  it("renames and forgets a remote machine", async () => {
    const w = await mountDetail();
    await w.get('[data-test="detail-rename"]').trigger("click");
    await flushPromises();
    expect(w.get('[data-test="machine-detail-title"]').text()).toBe(
      "Render box",
    );
    expect(getHost(routeHolder.id)?.name).toBe("Render box");

    await w.get('[data-test="detail-forget"]').trigger("click");
    await flushPromises();
    expect(getHost(routeHolder.id)).toBeNull();
    expect(routerPush).toHaveBeenCalledWith("/machines");
  });
});

describe("HostDetailPage — models", () => {
  it("lists installed models with a loaded badge", async () => {
    models = [
      {
        name: "flux-dev:q8",
        family: "flux",
        size_gb: 12,
        is_loaded: true,
        last_used: null,
        hf_repo: "bfl/flux",
        downloaded: true,
        default_steps: 4,
        default_guidance: 0,
        default_width: 1024,
        default_height: 1024,
        description: "",
      },
      {
        name: "sdxl-base",
        family: "sdxl",
        size_gb: 6,
        is_loaded: false,
        last_used: null,
        hf_repo: "sdxl",
        downloaded: false,
        default_steps: 30,
        default_guidance: 7,
        default_width: 1024,
        default_height: 1024,
        description: "",
      },
    ];
    const w = await mountDetail();
    // Only the downloaded model is installed.
    expect(w.findAll('[data-test="model-row"]')).toHaveLength(1);
    expect(w.find('[data-test="model-loaded"]').exists()).toBe(true);
  });
});
