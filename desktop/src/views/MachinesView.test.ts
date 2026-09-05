import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";

const appSettingsGet = vi.fn();
const discoverServers = vi.fn();
const runpodOverview = vi.fn();
const runpodStop = vi.fn().mockResolvedValue(undefined);
const secretGet = vi.fn().mockResolvedValue(null);
const testRemoteHost = vi.fn().mockResolvedValue({
  ok: true,
  version: "1",
  error: null,
  instanceId: null,
  hostname: null,
});
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: (...a: unknown[]) => appSettingsGet(...a),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    secretGet: (...a: unknown[]) => secretGet(...a),
    secretSet: vi.fn().mockResolvedValue(undefined),
    discoverServers: (...a: unknown[]) => discoverServers(...(a as [])),
    runpodOverview: (...a: unknown[]) => runpodOverview(...(a as [])),
    runpodStop: (...a: unknown[]) => runpodStop(...a),
    testRemoteHost: (...a: unknown[]) => testRemoteHost(...a),
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
import { useContextMenuStore } from "../stores/contextMenu";

const stub = { template: "<div />" };
let router: Router;

function addRunPodHost(hosts: ReturnType<typeof useHostsStore>) {
  hosts.extras.push({
    id: "pod-123-7680-proxy-runpod-net",
    label: "mold-runpod",
    url: "https://pod-123-7680.proxy.runpod.net",
    apiKey: null,
    status: "ready",
    error: null,
    instanceId: "uuid-runpod",
  });
}

/** The text of the FIRST render — captured synchronously, before any await
 *  lets Vue's scheduler flush what `onMounted` changed. That frame is what a
 *  user sees on arrival, and it is the only place a start-up flash lives. */
let firstFrame = "";

async function mountView(
  setup?: (hosts: ReturnType<typeof useHostsStore>) => void,
  path = "/machines",
) {
  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      {
        path: "/machines",
        name: "machines",
        component: stub,
        children: [
          { path: "runpod", name: "runpod", component: stub },
          { path: ":id", name: "host-detail", component: stub },
        ],
      },
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
  setup?.(hosts);
  const wrapper = mount(MachinesView, { global: { plugins: [pinia, router] } });
  firstFrame = wrapper.text();
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
  runpodOverview.mockResolvedValue({
    configured: true,
    credentialSource: "app",
    account: null,
    pods: [
      {
        id: "pod-123",
        name: "mold-runpod",
        desiredStatus: "RUNNING",
        imageName: "mold:latest",
        gpuCount: 1,
        costPerHr: 0.74,
        uptimeSeconds: 120,
        memoryInGb: 26,
        vcpuCount: 8,
        volumeInGb: 40,
        machine: {
          gpuDisplayName: "NVIDIA GeForce RTX 4090",
          gpuTypeId: "NVIDIA GeForce RTX 4090",
          dataCenterId: "US-TX-3",
          location: "Texas",
        },
        gpu: null,
        networkVolumeId: null,
        networkVolume: null,
      },
    ],
    gpus: [],
    datacenters: [],
    networkVolumes: [],
  });
});

describe("MachinesView overview", () => {
  it("tells the user an unreachable machine is reconnecting on its own", async () => {
    const wrapper = await mountView((hosts) => {
      const remote = hosts.extras.find((h) => h.id === "hal9000-7680")!;
      remote.status = "error";
      remote.error = "connection refused";
    });

    // Exactly one: the errored remote. The reachable This-device card and the
    // ready rows say nothing.
    const notes = wrapper.findAll("[data-test='host-reconnecting']");
    expect(notes).toHaveLength(1);
    expect(notes[0]!.text()).toBe("reconnecting…");
    expect(notes[0]!.classes()).toContain("text-warning");
  });

  it("shows a stale verified machine as reconnecting without discarding its telemetry", async () => {
    const wrapper = await mountView((hosts) => {
      hosts.telemetry["hal9000-7680"]!.stale = true;
    });

    expect(wrapper.get("[data-test='host-reconnecting']").text()).toBe("reconnecting…");
    expect(wrapper.get("[data-test='host-card']").text()).toContain("NVIDIA B200");
    expect(wrapper.get("[data-test='host-card']").text()).toContain("nothing waiting");
  });

  it("offers common context-menu actions for connected, remembered, and discovered hosts", async () => {
    const wrapper = await mountView();

    await wrapper.get("[data-test='host-card']").trigger("contextmenu");
    expect(
      useContextMenuStore().entries.flatMap((entry) => ("separator" in entry ? [] : [entry.label])),
    ).toEqual([
      "Open details",
      "Make images here",
      "Copy address",
      "Open web UI",
      "Disconnect",
      "Forget…",
    ]);

    await wrapper.get("[data-test='remembered-host']").trigger("contextmenu");
    expect(
      useContextMenuStore().entries.flatMap((entry) => ("separator" in entry ? [] : [entry.label])),
    ).toEqual(["Connect", "Copy address", "Forget…"]);

    await wrapper.get("[data-test='discovered-host']").trigger("contextmenu");
    expect(
      useContextMenuStore().entries.flatMap((entry) => ("separator" in entry ? [] : [entry.label])),
    ).toEqual(["Connect", "Copy address"]);
    wrapper.unmount();
  });

  it("lands on the machine the shell talks about and marks it selected", async () => {
    const wrapper = await mountView();
    // No pin and nothing running: this device.
    expect(router.currentRoute.value.path).toBe("/machines/local");
    expect(wrapper.get("[data-test='this-device-card']").attributes("aria-current")).toBe("true");
    expect(wrapper.get("[data-test='host-card']").attributes("aria-current")).toBeUndefined();
  });

  it("badges the machine that makes images and offers the rest the job", async () => {
    appSettingsGet.mockResolvedValue({
      savedHosts: [],
      connectedHostIds: [],
      generateTargetHost: "hal9000-7680",
    });
    const wrapper = await mountView();
    const { useAppPrefsStore } = await import("../stores/appPrefs");
    await useAppPrefsStore().init();
    await flushPromises();
    expect(wrapper.get("[data-test='host-card']").find("[data-test='target-badge']").text()).toBe(
      "making images here",
    );
    expect(
      wrapper.get("[data-test='this-device-card']").find("[data-test='target-badge']").exists(),
    ).toBe(false);
  });

  it("renders This device first, then connected remote cards", async () => {
    const wrapper = await mountView();
    expect(wrapper.find("[data-test='this-device-card']").exists()).toBe(true);
    const remotes = wrapper.findAll("[data-test='host-card']");
    expect(remotes).toHaveLength(1);
    expect(remotes[0]!.text()).toContain("hal9000");
    // This device shows its memory meter and hardware line.
    const device = wrapper.get("[data-test='this-device-card']");
    expect(device.text()).toContain("Apple M3 Max");
    // The meter above says what it measures, so the reading is bare numbers.
    expect(device.text()).not.toContain("Memory");
    expect(remotes[0]!.text()).toContain("NVIDIA GeForce RTX 4090 + NVIDIA B200");
    expect(remotes[0]!.text()).toContain("30.0 / 104.0 GB");
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

  it("opens the connected machine detail when its RunPod row is clicked", async () => {
    const wrapper = await mountView(addRunPodHost);

    const pod = wrapper.get("[data-test='runpod-running']");
    const open = pod.get("[data-test='runpod-open']");
    expect(open.attributes("aria-label")).toBe("Open mold-runpod machine details");
    expect(pod.text()).toContain("mold-runpod");
    expect(pod.find("[data-test='machine-chevron']").exists()).toBe(true);

    await open.trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/machines/pod-123-7680-proxy-runpod-net");
  });

  it("lists a connected pod once — as its pod row, never also as a machine card", async () => {
    const wrapper = await mountView(addRunPodHost);

    // The pod row already IS the machine row for a rented GPU: its menu is
    // that host's own, and only it can carry the running cost and Stop.
    const machines = wrapper.findAll("[data-test='host-card']");
    expect(machines.map((card) => card.text())).toHaveLength(1);
    expect(machines.every((card) => !card.text().includes("mold-runpod"))).toBe(true);
    expect(wrapper.findAll("[data-test='runpod-running']")).toHaveLength(1);
  });

  it("marks the pod row when the pod is the machine making images", async () => {
    appSettingsGet.mockResolvedValue({
      savedHosts: [],
      connectedHostIds: [],
      generateTargetHost: "pod-123-7680-proxy-runpod-net",
    });
    const wrapper = await mountView(addRunPodHost);
    const { useAppPrefsStore } = await import("../stores/appPrefs");
    await useAppPrefsStore().init();
    await flushPromises();

    const pod = wrapper.get("[data-test='runpod-running']");
    expect(pod.get("[data-test='target-badge']").text()).toBe("making images here");
  });

  it("does not claim the network is empty before the first scan answers", async () => {
    // `scan()` runs in onMounted — AFTER the first render — so a `scanning`
    // that starts false published "No other mold servers found" in the very
    // frame the user arrives on, before anything had looked.
    discoverServers.mockResolvedValue([]);
    const wrapper = await mountView();
    expect(firstFrame).not.toContain("No other mold servers found");
    // …and it does say so once the scan has actually come back empty.
    expect(wrapper.text()).toContain("No other mold servers found");
  });

  it("connects to an unconnected RunPod before opening its machine detail", async () => {
    const wrapper = await mountView();

    await wrapper.get("[data-test='runpod-open']").trigger("click");
    await flushPromises();

    expect(testRemoteHost).toHaveBeenCalledWith("https://pod-123-7680.proxy.runpod.net", null);
    expect(router.currentRoute.value.path).toBe("/machines/pod-123-7680-proxy-runpod-net");
  });

  it("offers machine and RunPod context actions on a running pod", async () => {
    const wrapper = await mountView(addRunPodHost);

    await wrapper.get("[data-test='runpod-running']").trigger("contextmenu");
    expect(
      useContextMenuStore().entries.flatMap((entry) => ("separator" in entry ? [] : [entry.label])),
    ).toEqual([
      "Open details",
      "Make images here",
      "Copy address",
      "Open web UI",
      "Disconnect",
      "Forget…",
      "Manage RunPod",
      "Stop it",
    ]);
  });

  it("does not offer Stop for a pod backed by a network volume", async () => {
    const overview = await runpodOverview();
    overview.pods[0].networkVolumeId = "network-volume-1";
    overview.pods[0].networkVolume = {
      id: "network-volume-1",
      name: "models",
      dataCenterId: "US-TX-3",
      size: 100,
    };
    runpodOverview.mockResolvedValue(overview);
    const wrapper = await mountView(addRunPodHost);

    await wrapper.get("[data-test='runpod-running']").trigger("contextmenu");
    expect(
      useContextMenuStore().entries.flatMap((entry) => ("separator" in entry ? [] : [entry.label])),
    ).not.toContain("Stop it");
  });

  it("stops a pod without also opening its machine detail", async () => {
    const wrapper = await mountView(addRunPodHost);

    await wrapper.get("[data-test='pod-cost-stop']").trigger("click");
    await flushPromises();
    expect(runpodStop).toHaveBeenCalledWith("pod-123");
    expect(router.currentRoute.value.path).not.toContain("pod-123");
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

  it("prompts for a key before connecting an authenticated discovered host", async () => {
    discoverServers.mockResolvedValue([
      {
        name: "locked-7680",
        url: "http://192.168.1.30:7680",
        host: "192.168.1.30",
        port: 7680,
        version: "1",
        authRequired: true,
        isThisMachine: false,
      },
    ]);
    const wrapper = await mountView();

    await wrapper.get("[data-test='discovered-add']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='connect-discovered-selected']").text()).toContain(
      "locked-7680",
    );
    expect(wrapper.find("[data-test='connect-key']").exists()).toBe(true);
    expect(wrapper.find("[data-test='connect-error']").exists()).toBe(false);

    await wrapper.get("[data-test='connect-key']").setValue("peer-secret");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();
    expect(testRemoteHost).toHaveBeenCalledWith("http://192.168.1.30:7680", "peer-secret");
  });

  it("hides the app's own embedded server from On your network but keeps standalone same-machine servers", async () => {
    discoverServers.mockResolvedValue([
      {
        name: "halcyon-7680",
        url: "http://192.168.1.142:7680",
        host: "192.168.1.142",
        port: 7680,
        version: "1",
        authRequired: true,
        isThisMachine: true,
        instanceId: "uuid-local",
      },
      {
        name: "halcyon-7681",
        url: "http://192.168.1.142:7681",
        host: "192.168.1.142",
        port: 7681,
        version: "1",
        authRequired: false,
        isThisMachine: true,
        instanceId: "uuid-standalone",
      },
      {
        // A copied MOLD_HOME on another box shares the primary's UUID; it is
        // NOT this machine and must stay discoverable.
        name: "clone-7680",
        url: "http://192.168.1.99:7680",
        host: "192.168.1.99",
        port: 7680,
        version: "1",
        authRequired: false,
        isThisMachine: false,
        instanceId: "uuid-local",
      },
    ]);
    const wrapper = await mountView((hosts) => {
      hosts.telemetry.local!.instanceId = "uuid-local";
    });

    const rows = wrapper.findAll("[data-test='discovered-host']");
    const text = rows.map((row) => row.text()).join(" ");
    expect(text).not.toContain("halcyon-7680");
    expect(text).toContain("halcyon-7681");
    expect(text).toContain("clone-7680");
  });

  it("opens the connect-a-machine dialog from the palette's ?connect=1 and cleans the address", async () => {
    const wrapper = await mountView(undefined, "/machines?connect=1");
    expect(wrapper.find("[data-test='connect-address']").exists()).toBe(true);
    expect(router.currentRoute.value.query.connect).toBeUndefined();
  });

  it("opens the connect-a-machine modal from Add machine", async () => {
    const wrapper = await mountView();
    expect(wrapper.find("[data-test='connect-address']").exists()).toBe(false);
    await wrapper.get("[data-test='add-machine']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='connect-address']").exists()).toBe(true);
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
    expect(testRemoteHost).toHaveBeenCalledWith("http://192.168.1.20:7680", "twin-key");
  });
});
