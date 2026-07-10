import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import EngineSection from "./EngineSection.vue";
import type { DiscoveredHost } from "../../lib/ipc";
import { useConnectionStore } from "../../stores/connection";

const discoverServers = vi.fn<() => Promise<DiscoveredHost[]>>();
const testRemoteHost = vi.fn().mockResolvedValue({ ok: true, version: "0.14.0", error: null });

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: () => Promise.resolve({ remoteUrl: "", remoteApiKey: null, mode: "local" }),
    secretGet: () => Promise.resolve(null),
    discoverServers: (...a: unknown[]) => discoverServers(...(a as [])),
    testRemoteHost: (...a: unknown[]) => testRemoteHost(...a),
    getConnection: () => Promise.resolve({ mode: "off", baseUrl: null, apiKey: null }),
  },
}));

// ConfigSettingRow reaches for the settings config; stub it to a no-op.
vi.mock("./ConfigSettingRow.vue", () => ({ default: { template: "<div />" } }));

function host(overrides: Partial<DiscoveredHost> = {}): DiscoveredHost {
  return {
    name: "hal9000-7680",
    url: "http://192.168.1.10:7680",
    host: "192.168.1.10",
    port: 7680,
    version: "0.14.0",
    authRequired: false,
    isThisMachine: false,
    ...overrides,
  };
}

async function mountSection() {
  setActivePinia(createPinia());
  const wrapper = mount(EngineSection);
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  discoverServers.mockReset();
  testRemoteHost.mockClear();
});

describe("EngineSection discovery", () => {
  it("renders exactly one selected engine mode", async () => {
    discoverServers.mockResolvedValue([]);
    const wrapper = await mountSection();
    const connection = useConnectionStore();
    connection.info = { mode: "external", baseUrl: "http://127.0.0.1:7680", apiKey: null };
    await wrapper.vm.$nextTick();

    const choices = wrapper.findAll('[role="radio"]');
    expect(choices).toHaveLength(2);
    expect(choices.map((choice) => choice.attributes("aria-checked"))).toEqual(["true", "false"]);

    connection.info = { mode: "remote", baseUrl: "http://hal9000:7680", apiKey: null };
    await wrapper.vm.$nextTick();
    expect(choices.map((choice) => choice.attributes("aria-checked"))).toEqual(["false", "true"]);
  });

  it("auto-scans on mount and renders discovered hosts", async () => {
    discoverServers.mockResolvedValue([
      host({ name: "hal9000-7680", isThisMachine: true }),
      host({ name: "studio-7680", url: "http://192.168.1.20:7680", authRequired: true }),
    ]);
    const wrapper = await mountSection();
    expect(discoverServers).toHaveBeenCalledTimes(1);
    const text = wrapper.text();
    expect(text).toContain("hal9000-7680");
    expect(text).toContain("studio-7680");
    expect(text).toContain("THIS MAC");
    expect(text).toContain("KEY");
  });

  it("shows an empty message when nothing is found", async () => {
    discoverServers.mockResolvedValue([]);
    const wrapper = await mountSection();
    expect(wrapper.text()).toContain("No mold servers found on your network");
  });

  it("Use on a keyless host fills the URL and tests the connection", async () => {
    discoverServers.mockResolvedValue([host()]);
    const wrapper = await mountSection();
    const useBtn = wrapper.findAll("button").find((b) => b.text() === "Use");
    expect(useBtn).toBeDefined();
    await useBtn!.trigger("click");
    await flushPromises();
    expect(testRemoteHost).toHaveBeenCalledWith("http://192.168.1.10:7680", null);
    const urlInput = wrapper.get<HTMLInputElement>("#remote-host");
    expect(urlInput.element.value).toBe("http://192.168.1.10:7680");
  });

  it("Use on an auth-required host fills the URL but waits for a key", async () => {
    discoverServers.mockResolvedValue([host({ authRequired: true })]);
    const wrapper = await mountSection();
    const useBtn = wrapper.findAll("button").find((b) => b.text() === "Use");
    await useBtn!.trigger("click");
    await flushPromises();
    // No connection test yet — the key field takes focus first.
    expect(testRemoteHost).not.toHaveBeenCalled();
    const urlInput = wrapper.get<HTMLInputElement>("#remote-host");
    expect(urlInput.element.value).toBe("http://192.168.1.10:7680");
  });

  it("Scan again re-runs discovery", async () => {
    discoverServers.mockResolvedValue([]);
    const wrapper = await mountSection();
    const scanBtn = wrapper.findAll("button").find((b) => b.text() === "Scan again");
    await scanBtn!.trigger("click");
    await flushPromises();
    expect(discoverServers).toHaveBeenCalledTimes(2);
  });
});
