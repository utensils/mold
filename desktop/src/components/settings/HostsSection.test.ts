import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import HostsSection from "./HostsSection.vue";
import type { DiscoveredHost, SavedHost } from "../../lib/ipc";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";

const discoverServers = vi.fn<() => Promise<DiscoveredHost[]>>();
const testRemoteHost = vi.fn().mockResolvedValue({
  ok: true,
  version: "0.18.0",
  error: null,
  instanceId: null,
  hostname: null,
});
const secretGet = vi.fn().mockResolvedValue(null);
const secretSet = vi.fn().mockResolvedValue(undefined);
const forgetRemoteHost = vi.fn().mockResolvedValue([]);
const appSettingsSet = vi.fn().mockResolvedValue(undefined);
let savedHosts: SavedHost[] = [];

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: () =>
      Promise.resolve({ savedHosts, connectedHostIds: [], generateTargetHost: null }),
    appSettingsSet: (...a: unknown[]) => appSettingsSet(...a),
    secretGet: (...a: unknown[]) => secretGet(...a),
    secretSet: (...a: unknown[]) => secretSet(...a),
    discoverServers: (...a: unknown[]) => discoverServers(...(a as [])),
    testRemoteHost: (...a: unknown[]) => testRemoteHost(...a),
    forgetRemoteHost: (...a: unknown[]) => forgetRemoteHost(...a),
  },
}));

// The hosts store's refresh poll hits /api/status; keep it inert.
vi.mock("../../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/api/client")>()),
  apiJsonTo: vi.fn().mockResolvedValue({ queue_depth: 0, queue_capacity: 8, version: null }),
}));

vi.mock("./ConfigSettingRow.vue", () => ({ default: { template: "<div />" } }));

function host(overrides: Partial<DiscoveredHost> = {}): DiscoveredHost {
  return {
    name: "hal9000-7680",
    url: "http://192.168.1.10:7680",
    host: "192.168.1.10",
    port: 7680,
    version: "0.18.0",
    authRequired: false,
    isThisMachine: false,
    ...overrides,
  };
}

async function mountSection() {
  setActivePinia(createPinia());
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  const wrapper = mount(HostsSection);
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  discoverServers.mockReset();
  discoverServers.mockResolvedValue([]);
  testRemoteHost.mockClear();
  secretGet.mockClear();
  secretGet.mockResolvedValue(null);
  secretSet.mockClear();
  forgetRemoteHost.mockClear();
  appSettingsSet.mockClear();
  savedHosts = [];
});

describe("HostsSection this-device card", () => {
  it("reveals and copies the desktop local-server API key", async () => {
    const wrapper = await mountSection();
    const conn = useConnectionStore();
    conn.localInfo = {
      kind: "embedded",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "desktop-secret",
      port: 7680,
    };
    conn.localStatus = "ready";
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });

    await wrapper.vm.$nextTick();
    expect(wrapper.text()).not.toContain("desktop-secret");
    await wrapper.get('[data-test="reveal-local-api-key"]').trigger("click");
    expect(wrapper.text()).toContain("desktop-secret");
    await wrapper.get('[data-test="copy-local-api-key"]').trigger("click");
    expect(writeText).toHaveBeenCalledWith("desktop-secret");
  });

  it("no longer advertises the Keychain", async () => {
    const wrapper = await mountSection();
    expect(wrapper.text()).not.toContain("KEYCHAIN");
  });
});

describe("HostsSection discovery", () => {
  it("auto-scans on mount and renders discovered hosts with badges", async () => {
    discoverServers.mockResolvedValue([
      host({ name: "hal9000-7680", isThisMachine: true }),
      host({ name: "studio-7680", url: "http://192.168.1.20:7680", authRequired: true }),
    ]);
    const wrapper = await mountSection();
    expect(discoverServers).toHaveBeenCalledTimes(1);
    const text = wrapper.text();
    expect(text).toContain("hal9000-7680");
    expect(text).toContain("studio-7680");
    expect(text).toContain("THIS DEVICE");
    expect(text).toContain("KEY");
  });

  it("hides a discovered host whose instance id is already connected", async () => {
    discoverServers.mockResolvedValue([
      host({ name: "hal-by-ip", url: "http://192.168.1.10:7680", instanceId: "uuid-1" }),
      host({ name: "okra-7680", url: "http://okra:7680", instanceId: "uuid-2" }),
    ]);
    const wrapper = await mountSection();
    const hosts = useHostsStore();
    // hal9000 is connected (by hostname) and reports instance id uuid-1.
    hosts.extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: "uuid-1",
    });
    await wrapper.vm.$nextTick();
    const rows = wrapper.findAll('[data-test="discovered-host"]');
    // Only okra remains — the IP advertisement of the connected box is hidden.
    expect(rows).toHaveLength(1);
    expect(rows[0]!.text()).toContain("okra-7680");
  });

  it("Scan again re-runs discovery", async () => {
    const wrapper = await mountSection();
    const scanBtn = wrapper.findAll("button").find((b) => b.text() === "Scan again");
    await scanBtn!.trigger("click");
    await flushPromises();
    expect(discoverServers).toHaveBeenCalledTimes(2);
  });

  it("shows an empty message when nothing else is found", async () => {
    const wrapper = await mountSection();
    expect(wrapper.text()).toContain("No other mold servers found on your network");
  });
});

describe("HostsSection add flow", () => {
  it("Add host connects the typed URL and key", async () => {
    testRemoteHost.mockResolvedValue({
      ok: true,
      version: "0.18.0",
      error: null,
      instanceId: null,
      hostname: "hal9000",
    });
    const wrapper = await mountSection();
    await wrapper.get('[data-test="add-host-url"]').setValue("hal9000");
    await wrapper.get('[data-test="add-host-key"]').setValue("secret-key");
    await wrapper.get('[data-test="add-host"]').trigger("click");
    await flushPromises();
    // hosts.connect normalizes the URL before probing.
    expect(testRemoteHost).toHaveBeenCalledWith("http://hal9000:7680", "secret-key");
    expect(secretSet).toHaveBeenCalledWith("remote-api-key.hal9000-7680", "secret-key");
    // The connected host now appears in the Connected list.
    expect(wrapper.text()).toContain("hal9000");
  });

  it("adding a discovered keyless host connects it in one click", async () => {
    discoverServers.mockResolvedValue([host({ name: "okra-7680", url: "http://okra:7680" })]);
    const wrapper = await mountSection();
    await wrapper.get('[data-test="discovered-host-add"]').trigger("click");
    await flushPromises();
    expect(testRemoteHost).toHaveBeenCalledWith("http://okra:7680", null);
  });
});

describe("HostsSection remembered hosts", () => {
  const saved: SavedHost = {
    id: "hal9000-7680",
    name: "hal9000",
    url: "http://hal9000:7680",
    lastUsedMs: Date.now() - 3_600_000,
  };

  it("lists remembered hosts that aren't connected", async () => {
    savedHosts = [saved];
    const wrapper = await mountSection();
    const text = wrapper.text();
    expect(text).toContain("Remembered");
    expect(text).toContain("hal9000");
    expect(text).toContain("1h ago");
  });

  it("Connect uses the host's own stored API key", async () => {
    savedHosts = [saved];
    secretGet.mockResolvedValue("per-host-key");
    const wrapper = await mountSection();
    const connectBtn = wrapper
      .findAll("[data-test='saved-host-connect']")
      .find((b) => b.text() === "Connect");
    await connectBtn!.trigger("click");
    await flushPromises();
    expect(secretGet).toHaveBeenCalledWith("remote-api-key.hal9000-7680");
    expect(testRemoteHost).toHaveBeenCalledWith("http://hal9000:7680", "per-host-key");
  });

  it("Forget is a two-step confirm that drops the host", async () => {
    savedHosts = [saved];
    const wrapper = await mountSection();
    const forgetBtn = wrapper.get("[data-test='saved-host-forget']");
    await forgetBtn.trigger("click");
    expect(forgetRemoteHost).not.toHaveBeenCalled();
    expect(forgetBtn.text()).toContain("Forget?");
    await forgetBtn.trigger("click");
    await flushPromises();
    expect(forgetRemoteHost).toHaveBeenCalledWith("hal9000-7680");
  });
});
