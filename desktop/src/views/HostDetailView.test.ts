import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";

const apiJsonTo = vi.fn();
vi.mock("../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiJsonTo: (...a: unknown[]) => apiJsonTo(...a),
  apiFetchTo: vi.fn(),
  apiJson: vi.fn(),
  apiFetch: vi.fn(),
  currentTarget: () => ({ baseUrl: "http://127.0.0.1:49152", apiKey: null }),
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

import HostDetailView from "./HostDetailView.vue";
import { useConnectionStore } from "../stores/connection";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import type { ModelEntry, ServerStatus } from "../lib/api/types";

const stub = { template: "<div />" };

const REMOTE_ID = "hal9000-7680";

function installApi(status: Partial<ServerStatus> = {}) {
  apiJsonTo.mockImplementation((_target: unknown, path: string) => {
    if (path === "/api/status") {
      return Promise.resolve({
        version: "0.17.0",
        models_loaded: [],
        uptime_secs: 5,
        ...status,
      });
    }
    if (path === "/api/models")
      return Promise.resolve([model("flux-dev:q8", "flux"), model("z-image:q8", "z-image")]);
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

async function mountView(path = `/hosts/${REMOTE_ID}`) {
  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      { path: "/hosts/:id", component: stub },
      { path: "/settings", component: stub },
      { path: "/models", component: stub },
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
    instanceId: "0f7a2c31-instance-uuid",
    hostname: "hal9000",
  };
  // fetchedAt now → the view's hostModels.refresh() skips these (not stale).
  const hostModels = useHostModelsStore();
  hostModels.byHost[REMOTE_ID] = {
    entries: [model("flux-dev:q8", "flux"), model("z-image:q8", "z-image")],
    fetchedAt: Date.now(),
    error: null,
  };
  const wrapper = mount(HostDetailView, {
    global: { plugins: [pinia, router] },
  });
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
  sseCalls.length = 0;
  appSettingsGet.mockResolvedValue({
    savedHosts: [],
    connectedHostIds: [REMOTE_ID],
    generateTargetHost: null,
  });
  installApi();
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
    expect(instance.attributes("title")).toBe("0f7a2c31-instance-uuid");
    // Remote hosts get the remote-only management actions.
    expect(wrapper.find("[data-test='rename-host']").exists()).toBe(true);
    expect(wrapper.find("[data-test='disconnect-host']").exists()).toBe(true);
    expect(wrapper.find("[data-test='forget-host']").exists()).toBe(true);
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
    expect(missing.find("[data-test='back-to-hosts']").attributes("href")).toBe("/settings");
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

    // Before any frame: the status-poll gpu_info fallback (MB → decimal GB).
    expect(wrapper.get("[data-test='gpu-card']").text()).toContain("NVIDIA GeForce RTX 4090");
    expect(wrapper.get("[data-test='gpu-card']").text()).toContain("6.0 GB/24.0 GB");
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

  it("aborts the resources stream on unmount", async () => {
    const wrapper = await mountView();
    const stream = lastStream();
    expect(stream.options.signal.aborted).toBe(false);
    wrapper.unmount();
    expect(stream.options.signal.aborted).toBe(true);
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
    const chips = wrapper.findAll("[data-test='loaded-model-chip']");
    expect(chips.map((c) => c.text())).toEqual(["flux-dev:q8"]);
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
});

describe("HostDetailView forget", () => {
  it("requires a confirming second click, then drops the host and returns to Hosts", async () => {
    const wrapper = await mountView();
    const btn = wrapper.get("[data-test='forget-host']");
    await btn.trigger("click");
    expect(forgetRemoteHost).not.toHaveBeenCalled();
    expect(btn.text()).toBe("Forget?");
    await btn.trigger("click");
    await flushPromises();
    expect(forgetRemoteHost).toHaveBeenCalledWith(REMOTE_ID);
    expect(useHostsStore().extras).toHaveLength(0);
    expect(router.currentRoute.value.path).toBe("/settings");
  });
});
