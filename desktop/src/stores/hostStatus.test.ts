import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import StatusBar from "../components/shell/StatusBar.vue";
import { useHostStatusStore } from "./hostStatus";
import { useConnectionStore } from "./connection";
import { useGenerationStore } from "./generation";
import { useHostsStore } from "./hosts";
import { useAppPrefsStore } from "./appPrefs";
import { useToastStore } from "./toasts";
import { ipc } from "../lib/ipc";
import { PLATFORM_UI } from "../lib/platform";

const apiJson = vi.fn();
const apiJsonTo = vi.fn().mockRejectedValue(new Error("unused"));
vi.mock("../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiJson: (...a: unknown[]) => apiJson(...a),
  apiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
  apiFetch: vi.fn().mockRejectedValue(new Error("unused")),
  apiFetchTo: vi.fn().mockRejectedValue(new Error("unused")),
  apiHeaders: () => ({}),
  conditionalApiJsonTo: vi.fn().mockRejectedValue(new Error("unused")),
  currentTarget: () => ({ baseUrl: "http://127.0.0.1:49152", apiKey: null }),
}));

vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn().mockResolvedValue({}),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    ensureLocalServer: vi.fn().mockResolvedValue({
      kind: "embedded",
      baseUrl: "http://127.0.0.1:49152",
      apiKey: null,
      port: 49152,
    }),
    startLocalEngine: vi
      .fn()
      .mockResolvedValue({ mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null }),
    stopLocalEngine: vi.fn().mockResolvedValue(null),
    getConnection: vi.fn().mockResolvedValue(null),
    testRemoteHost: vi.fn().mockResolvedValue(null),
    secretGet: vi.fn().mockResolvedValue(null),
  },
}));

interface StreamCall {
  path: string;
  target?: { baseUrl: string } | null;
  onEvent?: ((event: string, data: string) => void) | undefined;
}
const streamCalls: StreamCall[] = [];
vi.mock("../lib/api/sse", () => ({
  sseStream: (
    path: string,
    opts: {
      target?: { baseUrl: string } | null;
      onEvent?: (event: string, data: string) => void;
    },
  ) => {
    streamCalls.push({ path, target: opts.target ?? null, onEvent: opts.onEvent });
    return Promise.resolve();
  },
}));

/** The primary connection App.vue waits for before it starts the store. */
function setupPinia() {
  const pinia = createPinia();
  setActivePinia(pinia);
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  return pinia;
}

let started: ReturnType<typeof useHostStatusStore> | null = null;

/** App.vue calls `start()` once the primary connection is ready. */
async function startHostStatus() {
  const store = useHostStatusStore();
  started = store;
  store.start();
  await flushPromises();
  return store;
}

function addRemoteHost() {
  const hosts = useHostsStore();
  hosts.extras.push({
    id: "hal9000-7680",
    label: "hal9000",
    url: "http://hal9000:7680",
    apiKey: null,
    status: "ready",
    error: null,
    instanceId: null,
  });
  hosts.telemetry["hal9000-7680"] = {
    queueDepth: 2,
    queueCapacity: 200,
    version: "0.16.0",
    modelsLoaded: ["flux2-klein:q4"],
    gpuInfo: { name: "NVIDIA GeForce RTX 4090", vram_total_mb: 24564, vram_used_mb: 8192 },
  };
  return hosts;
}

function runningJobOn(hostId: string, hostLabel: string) {
  useGenerationStore().jobs.push({
    clientId: 1,
    status: "denoising",
    hostId,
    hostLabel,
    submittedAtUnixMs: Date.now(),
  } as never);
}

const MULTI_GPU_SNAPSHOT = JSON.stringify({
  hostname: "plato",
  timestamp: 0,
  gpus: [
    {
      ordinal: 0,
      name: "NVIDIA L40S",
      backend: "cuda",
      vram_total: 46_100_000_000,
      vram_used: 400_000_000,
    },
    {
      ordinal: 1,
      name: "NVIDIA L40S",
      backend: "cuda",
      vram_total: 46_100_000_000,
      vram_used: 41_500_000_000,
    },
  ],
  system_ram: {
    total: 512_000_000_000,
    used: 455_900_000_000,
    used_by_mold: 0,
    used_by_other: 0,
  },
});

const SINGLE_GPU_SNAPSHOT = JSON.stringify({
  hostname: "halcyon",
  timestamp: 0,
  gpus: [
    {
      ordinal: 0,
      name: "Apple M3 Ultra",
      backend: "metal",
      vram_total: 196_600_000_000,
      vram_used: 32_800_000_000,
    },
  ],
  system_ram: {
    total: 196_600_000_000,
    used: 64_000_000_000,
    used_by_mold: 0,
    used_by_other: 0,
  },
});

/** Feed the display host's resources stream one snapshot frame. */
function emitSnapshot(json: string) {
  streamCalls.at(-1)?.onEvent?.("snapshot", json);
}

beforeEach(() => {
  vi.clearAllMocks();
  streamCalls.length = 0;
  apiJson.mockResolvedValue({
    version: "0.16.0",
    models_loaded: [],
    uptime_secs: 1,
    queue_depth: 0,
    queue_capacity: 200,
  });
});

afterEach(() => {
  started?.stop();
  started = null;
});

describe("hostStatus host-aware display", () => {
  it("shows the primary engine when nothing is generating remotely", async () => {
    setupPinia();
    const store = await startHostStatus();
    expect(store.displayHost?.id).toBe("local");
    expect(store.displayingRemote).toBe(false);
    expect(store.queue).toEqual({ modelsLoaded: [], depth: 0, capacity: 200 });
  });

  it("shows the host selected in the Create header while idle", async () => {
    setupPinia();
    addRemoteHost();
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    const store = await startHostStatus();
    expect(store.displayHost?.label).toBe("hal9000");
    expect(store.queue).toEqual({
      modelsLoaded: ["flux2-klein:q4"],
      depth: 2,
      capacity: 200,
    });
    expect(streamCalls.at(-1)?.target?.baseUrl).toBe("http://hal9000:7680");
  });

  it("reports an offline selected host instead of presenting missing telemetry as status", async () => {
    setupPinia();
    const hosts = addRemoteHost();
    const remote = hosts.extras.find((host) => host.id === "hal9000-7680")!;
    remote.label = "Parity Fixture";
    remote.status = "error";
    remote.error = "connection refused";
    delete hosts.telemetry[remote.id];
    useAppPrefsStore().settings = { generateTargetHost: remote.id } as never;

    const store = await startHostStatus();

    expect(store.displayHost?.label).toBe("Parity Fixture");
    expect(store.connection).toBe("error");
    expect(store.sentence).toBe("Machine is offline.");
    expect(store.sentence).not.toContain("GPU telemetry");
    // An unreachable host is never asked for resources.
    expect(streamCalls).toHaveLength(0);
  });

  it("describes a local engine failure as an engine lifecycle state", async () => {
    setupPinia();
    useConnectionStore().status = "error";

    const store = await startHostStatus();

    expect(store.displayingRemote).toBe(false);
    expect(store.connection).toBe("error");
    expect(store.sentence).toBe("The engine hit an error.");
    expect(store.sentence).not.toContain("offline");
  });

  it("keeps a concrete Create host selection while another host has a live job", async () => {
    setupPinia();
    addRemoteHost();
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "plato-7680",
      label: "plato",
      url: "http://plato:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    hosts.telemetry["plato-7680"] = {
      queueDepth: 1,
      queueCapacity: 200,
      version: "0.16.0",
      modelsLoaded: ["qwen-image:bf16"],
      gpuInfo: { name: "NVIDIA L40S", vram_total_mb: 46068, vram_used_mb: 12288 },
    };
    useAppPrefsStore().settings = { generateTargetHost: "plato-7680" } as never;
    runningJobOn("hal9000-7680", "hal9000");

    const store = await startHostStatus();

    expect(store.displayHost?.label).toBe("plato");
    expect(store.queue).toEqual({
      modelsLoaded: ["qwen-image:bf16"],
      depth: 1,
      capacity: 200,
    });
  });

  it("follows a live remote job: chip, queue, and models come from that host", async () => {
    setupPinia();
    addRemoteHost();
    runningJobOn("hal9000-7680", "hal9000");

    const store = await startHostStatus();

    expect(store.displayHost?.label).toBe("hal9000");
    expect(store.displayingRemote).toBe(true);
    expect(store.queue).toEqual({
      modelsLoaded: ["flux2-klein:q4"],
      depth: 2,
      capacity: 200,
    });
  });

  it("re-targets the resources stream at the display host", async () => {
    setupPinia();
    addRemoteHost();
    runningJobOn("hal9000-7680", "hal9000");

    await startHostStatus();

    const last = streamCalls.at(-1);
    expect(last?.path).toBe("/api/resources/stream");
    expect(last?.target?.baseUrl).toBe("http://hal9000:7680");
  });

  it("reverts to the primary when the last routed job settles", async () => {
    setupPinia();
    addRemoteHost();
    const generation = useGenerationStore();
    runningJobOn("hal9000-7680", "hal9000");

    const store = await startHostStatus();
    expect(store.displayHost?.label).toBe("hal9000");

    generation.jobs[0]!.status = "complete" as never;
    // App.vue reopens the stream whenever `${displayHost.id}:${connection}` changes.
    store.startResourceStream();
    await flushPromises();

    expect(store.displayHost?.id).toBe("local");
    expect(store.queue.depth).toBe(0);
    expect(store.queue.capacity).toBe(200);
    // The resources stream re-targets the primary (no explicit target).
    expect(streamCalls.at(-1)?.target ?? null).toBeNull();
  });

  it("keeps polling the PRIMARY status for engine recovery while displaying a remote host", async () => {
    setupPinia();
    addRemoteHost();
    runningJobOn("hal9000-7680", "hal9000");

    const store = await startHostStatus();

    expect(store.displayingRemote).toBe(true);
    // The recovery poll is apiJson (currentTarget = primary), never apiJsonTo.
    expect(apiJson).toHaveBeenCalledWith("/api/status");
    expect(apiJsonTo).not.toHaveBeenCalled();
  });

  it("restarts the embedded engine after consecutive primary status failures", async () => {
    setupPinia();
    apiJson.mockRejectedValue(new Error("connection refused"));

    const store = await startHostStatus();
    expect(useToastStore().items).toHaveLength(0);

    await store.refreshStatus();
    await flushPromises();

    expect(vi.mocked(ipc).startLocalEngine).toHaveBeenCalled();
    expect(useToastStore().items.map((t) => t.message)).toContain("Engine restarted");
  });

  it("falls back to the status gpu_info for a primary whose resources stream is silent", async () => {
    setupPinia();
    apiJson.mockResolvedValue({
      version: "0.16.0",
      models_loaded: [],
      uptime_secs: 1,
      queue_depth: 0,
      queue_capacity: 200,
      gpu_info: { name: "Apple M3 Ultra", vram_total_mb: 196608, vram_used_mb: 32768 },
    });

    const store = await startHostStatus();

    expect(store.snapshot).toBeNull();
    expect(store.gpus.map((gpu) => gpu.name)).toEqual(["Apple M3 Ultra"]);
    expect(store.vramTotal).toBe(196_608_000_000);
  });

  it("reports one VRAM entry per GPU from a multi-GPU snapshot", async () => {
    setupPinia();
    const store = await startHostStatus();
    emitSnapshot(MULTI_GPU_SNAPSHOT);

    expect(store.gpus).toHaveLength(2);
    expect(store.gpus.map((gpu) => gpu.ordinal)).toEqual([0, 1]);
    expect(store.gpus[0]!.vram_used).toBe(400_000_000);
    expect(store.gpus[1]!.vram_used).toBe(41_500_000_000);
  });

  it("falls back to every status worker when a remote resource stream is silent", async () => {
    setupPinia();
    const hosts = addRemoteHost();
    hosts.telemetry["hal9000-7680"]!.gpuWorkers = [
      {
        ordinal: 0,
        name: "NVIDIA RTX 3090",
        vram_total_bytes: 24_000_000_000,
        vram_used_bytes: 8_000_000_000,
        state: "generating",
      },
      {
        ordinal: 1,
        name: "NVIDIA B200",
        vram_total_bytes: 80_000_000_000,
        vram_used_bytes: 20_000_000_000,
        state: "idle",
      },
    ];
    runningJobOn("hal9000-7680", "hal9000");

    const store = await startHostStatus();

    expect(store.snapshot).toBeNull();
    expect(store.gpus.map((gpu) => gpu.name)).toEqual(["NVIDIA RTX 3090", "NVIDIA B200"]);
  });

  it("aggregates VRAM across all GPUs", async () => {
    setupPinia();
    const store = await startHostStatus();
    emitSnapshot(MULTI_GPU_SNAPSHOT);

    expect(store.vramUsed).toBe(41_900_000_000);
    expect(store.vramTotal).toBe(92_200_000_000);
    // (0.4 + 41.5) / (46.1 + 46.1) ≈ 45% — not GPU 0's near-zero usage.
    expect(Math.round(store.vramPct)).toBe(45);
    expect(store.vramCritical).toBe(false);
  });

  it("keeps a single entry for single-GPU hosts", async () => {
    setupPinia();
    const store = await startHostStatus();
    emitSnapshot(SINGLE_GPU_SNAPSHOT);

    expect(store.gpus).toHaveLength(1);
    expect(store.gpus[0]!.name).toBe("Apple M3 Ultra");
  });

  it("falls back to telemetry VRAM when the remote resources stream is silent", async () => {
    setupPinia();
    addRemoteHost();
    runningJobOn("hal9000-7680", "hal9000");

    const store = await startHostStatus();

    expect(store.gpus.map((gpu) => gpu.name)).toEqual(["NVIDIA GeForce RTX 4090"]);
    expect(store.vramTotal).toBe(24_564_000_000);
  });
});

describe("hostStatus host-memory pressure", () => {
  const RAM_SNAPSHOT = JSON.stringify({
    hostname: "plato",
    timestamp: 0,
    gpus: [],
    system_ram: {
      total: 64_000_000_000,
      used: 60_000_000_000,
      used_by_mold: 0,
      used_by_other: 0,
    },
  });

  async function startWithRam(hostMemory: unknown) {
    apiJson.mockResolvedValue({
      version: "0.16.0",
      models_loaded: [],
      uptime_secs: 1,
      queue_depth: 0,
      queue_capacity: 200,
      ...(hostMemory === undefined ? {} : { host_memory: hostMemory }),
    });
    setupPinia();
    const store = await startHostStatus();
    emitSnapshot(RAM_SNAPSHOT);
    return store;
  }

  // `/api/status` is the fresher source and the only one polled while neither
  // Machines nor a queued print is holding the cross-host queue poll open.
  it("reads RAM pressure from the status snapshot with no queue polling active", async () => {
    const store = await startWithRam({
      total_bytes: 64_000_000_000,
      available_bytes: 4_000_000_000,
      headroom_bytes: 0,
      safety_floor_bytes: 4_000_000_000,
    });
    expect(store.hostMemory?.headroom_bytes).toBe(0);
    expect(store.hostMemoryPressure).toBe("critical");
  });

  it("warns inside one safety floor of the wall", async () => {
    const store = await startWithRam({
      total_bytes: 64_000_000_000,
      available_bytes: 6_000_000_000,
      headroom_bytes: 3_000_000_000,
      safety_floor_bytes: 4_000_000_000,
    });
    expect(store.hostMemoryPressure).toBe("warn");
  });

  // The status poll is primary-only by design, so its ledger describes this
  // Mac. Painting it onto a remote's RAM row would report the wrong machine.
  it("never paints the primary's ledger onto a remote display host", async () => {
    apiJson.mockResolvedValue({
      version: "0.16.0",
      models_loaded: [],
      uptime_secs: 1,
      queue_depth: 0,
      queue_capacity: 200,
      host_memory: {
        total_bytes: 64_000_000_000,
        available_bytes: 1_000_000_000,
        headroom_bytes: 0,
        safety_floor_bytes: 4_000_000_000,
      },
    });
    setupPinia();
    addRemoteHost();
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    const store = await startHostStatus();
    emitSnapshot(RAM_SNAPSHOT);

    expect(store.displayingRemote).toBe(true);
    expect(store.hostMemory).toBeNull();
    expect(store.hostMemoryPressure).toBe("ok");
  });

  it("stays neutral when neither source reports host memory", async () => {
    const store = await startWithRam(undefined);
    expect(store.hostMemory).toBeNull();
    expect(store.hostMemoryPressure).toBe("ok");
  });
});

describe("StatusBar", () => {
  const stub = { template: "<div />" };

  function makeRouter(): Router {
    return createRouter({
      history: createMemoryHistory(),
      routes: ["/create", "/queue", "/models", "/machines", "/machines/:id"].map((path) => ({
        path,
        component: stub,
      })),
    });
  }

  async function mountStatusBar() {
    const router = makeRouter();
    router.push("/create");
    await router.isReady();
    const pinia = setupPinia();
    return mount(StatusBar, { global: { plugins: [pinia, router] } });
  }

  it("renders the machine, queue, and vram readouts", async () => {
    const wrapper = await mountStatusBar();
    await startHostStatus();
    emitSnapshot(SINGLE_GPU_SNAPSHOT);
    await flushPromises();

    expect(wrapper.find("[data-test='status-bar']").exists()).toBe(true);
    expect(wrapper.get("[data-test='status-machine']").text()).toBe(PLATFORM_UI.deviceLabel);
    expect(wrapper.get("[data-test='status-queue']").text()).toBe("nothing waiting");
    expect(wrapper.get("[data-test='status-vram']").text()).toBe("vram 32.8 GB / 196.6 GB");
    expect(wrapper.get("[data-test='status-ram']").text()).toBe("ram 64.0 GB");
  });

  it("names an offline machine beside a stopped dot", async () => {
    const wrapper = await mountStatusBar();
    const hosts = addRemoteHost();
    const remote = hosts.extras.find((host) => host.id === "hal9000-7680")!;
    remote.label = "Parity Fixture";
    remote.status = "error";
    delete hosts.telemetry[remote.id];
    useAppPrefsStore().settings = { generateTargetHost: remote.id } as never;
    await startHostStatus();
    await flushPromises();

    const machine = wrapper.get("[data-test='status-machine']");
    expect(machine.text()).toContain("Parity Fixture · offline");
    expect(machine.get("span").classes()).toContain("bg-error");
    expect(machine.attributes("title")).toBe("Machine is offline.");
  });

  it("says a starting engine is connecting", async () => {
    const wrapper = await mountStatusBar();
    useConnectionStore().status = "starting";
    await startHostStatus();
    await flushPromises();

    expect(wrapper.get("[data-test='status-machine']").text()).toContain("· connecting");
  });

  it("counts the work in flight in the queue readout", async () => {
    const wrapper = await mountStatusBar();
    runningJobOn("local", "This device");
    await startHostStatus();
    await flushPromises();

    expect(wrapper.get("[data-test='status-queue']").text()).toBe("1 image being made");
  });

  it("turns the RAM readout red when the host has no headroom left", async () => {
    apiJson.mockResolvedValue({
      version: "0.16.0",
      models_loaded: [],
      uptime_secs: 1,
      queue_depth: 0,
      queue_capacity: 200,
      host_memory: {
        total_bytes: 64_000_000_000,
        available_bytes: 4_000_000_000,
        headroom_bytes: 0,
        safety_floor_bytes: 4_000_000_000,
      },
    });
    const wrapper = await mountStatusBar();
    await startHostStatus();
    emitSnapshot(SINGLE_GPU_SNAPSHOT);
    await flushPromises();

    expect(wrapper.get("[data-test='status-ram']").classes()).toContain("text-error");
  });

  it("warns on the RAM readout inside one safety floor of the wall", async () => {
    apiJson.mockResolvedValue({
      version: "0.16.0",
      models_loaded: [],
      uptime_secs: 1,
      queue_depth: 0,
      queue_capacity: 200,
      host_memory: {
        total_bytes: 64_000_000_000,
        available_bytes: 6_000_000_000,
        headroom_bytes: 3_000_000_000,
        safety_floor_bytes: 4_000_000_000,
      },
    });
    const wrapper = await mountStatusBar();
    await startHostStatus();
    emitSnapshot(SINGLE_GPU_SNAPSHOT);
    await flushPromises();

    const ram = wrapper.get("[data-test='status-ram']");
    expect(ram.classes()).toContain("text-warning");
    expect(ram.classes()).not.toContain("text-error");
  });

  it("keeps the RAM readout neutral when no source reports host memory", async () => {
    const wrapper = await mountStatusBar();
    await startHostStatus();
    emitSnapshot(SINGLE_GPU_SNAPSHOT);
    await flushPromises();

    const ram = wrapper.get("[data-test='status-ram']");
    expect(ram.classes()).not.toContain("text-error");
    expect(ram.classes()).not.toContain("text-warning");
  });
});
