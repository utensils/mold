import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";

const appSettingsGet = vi.fn();
const discoverServers = vi.fn();
const secretGet = vi.fn().mockResolvedValue(null);
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: (...a: unknown[]) => appSettingsGet(...a),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    secretGet: (...a: unknown[]) => secretGet(...a),
    secretSet: vi.fn().mockResolvedValue(undefined),
    discoverServers: (...a: unknown[]) => discoverServers(...(a as [])),
    testRemoteHost: vi
      .fn()
      .mockResolvedValue({ ok: true, version: "1", error: null, instanceId: null, hostname: null }),
    forgetRemoteHost: vi.fn().mockResolvedValue([]),
  },
}));
vi.mock("../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiJsonTo: vi.fn().mockResolvedValue({ queue_depth: 0, queue_capacity: 8, version: null }),
  apiFetchTo: vi.fn(),
  currentTarget: () => ({ baseUrl: "http://127.0.0.1:49152", apiKey: null }),
}));
vi.mock("../lib/api/sse", () => ({ sseStream: vi.fn() }));
vi.mock("../lib/notify", () => ({ notifyGenerated: vi.fn(), notifyGenerationFailed: vi.fn() }));

import MachinesView from "./MachinesView.vue";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";

const stub = { template: "<div />" };
let router: Router;

async function mountView() {
  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      { path: "/machines", component: stub },
      { path: "/machines/runpod", component: stub },
      { path: "/machines/:id", component: stub },
    ],
  });
  router.push("/machines");
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  const hosts = useHostsStore();
  hosts.extras.push({
    id: "hal9000-7680",
    label: "hal9000",
    url: "http://hal9000:7680",
    apiKey: "sekrit",
    status: "ready",
    error: null,
    instanceId: "uuid-remote",
  });
  hosts.telemetry.local = {
    queueDepth: 1,
    queueCapacity: 8,
    version: "1",
    gpuInfo: {
      name: "Apple M3 Max",
      vram_total_mb: 48_000,
      vram_used_mb: 30_000,
      backend: "metal",
    },
  };
  hosts.telemetry["hal9000-7680"] = {
    queueDepth: 0,
    queueCapacity: 8,
    version: "1",
    gpuInfo: {
      name: "NVIDIA GeForce RTX 4090",
      vram_total_mb: 24_000,
      vram_used_mb: 10_000,
      backend: "cuda",
    },
    gpuWorkers: [
      {
        ordinal: 0,
        name: "NVIDIA GeForce RTX 4090",
        vram_total_bytes: 24_000_000_000,
        vram_used_bytes: 10_000_000_000,
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
  };
  const wrapper = mount(MachinesView, { global: { plugins: [pinia, router] } });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  vi.clearAllMocks();
  secretGet.mockResolvedValue(null);
  appSettingsGet.mockResolvedValue({
    savedHosts: [
      { id: "okra-7680", name: "okra", url: "http://okra:7680", lastUsedMs: 1, instanceId: null },
    ],
    connectedHostIds: [],
    generateTargetHost: null,
  });
  discoverServers.mockResolvedValue([
    {
      name: "studio-7680",
      url: "http://192.168.1.20:7680",
      host: "192.168.1.20",
      port: 7680,
      version: "1",
      authRequired: false,
      isThisMachine: false,
    },
  ]);
});

describe("MachinesView overview", () => {
  it("renders This device first, then connected remote cards", async () => {
    const wrapper = await mountView();
    expect(wrapper.find("[data-test='this-device-card']").exists()).toBe(true);
    const remotes = wrapper.findAll("[data-test='host-card']");
    expect(remotes).toHaveLength(1);
    expect(remotes[0]!.text()).toContain("hal9000");
    // This device shows its memory meter and hardware line.
    const device = wrapper.get("[data-test='this-device-card']");
    expect(device.text()).toContain("Apple M3 Max");
    expect(device.text()).toContain("Memory");
    expect(remotes[0]!.text()).toContain("NVIDIA GeForce RTX 4090 + NVIDIA B200");
    expect(remotes[0]!.text()).toContain("30.0 GB / 104.0 GB");
  });

  it("never offers Forget on the This device card (it has no saved entry)", async () => {
    const wrapper = await mountView();
    expect(wrapper.get("[data-test='this-device-card']").text()).not.toContain("Forget");
  });

  it("routes the RunPod offer card to the provisioning view", async () => {
    const wrapper = await mountView();
    await wrapper.get("[data-test='start-pod']").trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/machines/runpod");
  });

  it("opens a host detail when a machine card is clicked", async () => {
    const wrapper = await mountView();
    await wrapper.get("[data-test='host-card']").trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/machines/hal9000-7680");
  });

  it("lists remembered (offline) hosts with a Connect action", async () => {
    const wrapper = await mountView();
    const remembered = wrapper.get("[data-test='remembered-host']");
    expect(remembered.text()).toContain("okra");
    expect(remembered.find("[data-test='remembered-connect']").exists()).toBe(true);
  });

  it("lists hosts discovered on the network with a Connect action", async () => {
    const wrapper = await mountView();
    const discovered = wrapper.get("[data-test='discovered-host']");
    expect(discovered.text()).toContain("studio-7680");
    expect(discovered.find("[data-test='discovered-add']").exists()).toBe(true);
  });

  it("opens the connect-a-machine modal from Add machine", async () => {
    const wrapper = await mountView();
    expect(wrapper.find("[data-test='connect-type-remote']").exists()).toBe(false);
    await wrapper.get("[data-test='add-machine']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='connect-type-remote']").exists()).toBe(true);
  });

  it("connecting a discovered box finds a stored key under its instance-id twin's slug", async () => {
    // Remembered by hostname (studio-7680) but advertised by IP: the key is
    // stored under the remembered slug, not the advertised one.
    appSettingsGet.mockResolvedValue({
      savedHosts: [
        {
          id: "studio-7680",
          name: "studio",
          url: "http://studio:7680",
          lastUsedMs: 1,
          instanceId: "uuid-studio",
        },
      ],
      connectedHostIds: [],
      generateTargetHost: null,
    });
    discoverServers.mockResolvedValue([
      {
        name: "studio-7680",
        url: "http://192.168.1.20:7680",
        host: "192.168.1.20",
        port: 7680,
        version: "1",
        authRequired: true,
        isThisMachine: false,
        instanceId: "uuid-studio",
      },
    ]);
    secretGet.mockImplementation((name: unknown) =>
      Promise.resolve(name === "remote-api-key.studio-7680" ? "twin-key" : null),
    );
    const wrapper = await mountView();
    await wrapper.get("[data-test='discovered-add']").trigger("click");
    await flushPromises();
    expect(secretGet).toHaveBeenCalledWith("remote-api-key.192-168-1-20-7680");
    expect(secretGet).toHaveBeenCalledWith("remote-api-key.studio-7680");
  });
});
