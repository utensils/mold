import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import EngineSection from "./EngineSection.vue";
import type { DiscoveredHost, SavedHost } from "../../lib/ipc";
import { useConnectionStore } from "../../stores/connection";

const discoverServers = vi.fn<() => Promise<DiscoveredHost[]>>();
const testRemoteHost = vi.fn().mockResolvedValue({ ok: true, version: "0.14.0", error: null });
const setRemoteHost = vi
  .fn()
  .mockResolvedValue({ mode: "remote", baseUrl: "http://hal9000:7680", apiKey: null });
const secretGet = vi.fn().mockResolvedValue(null);
const forgetRemoteHost = vi.fn().mockResolvedValue([]);
let savedHosts: SavedHost[] = [];

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: () =>
      Promise.resolve({ remoteUrl: "", remoteApiKey: null, mode: "local", savedHosts }),
    secretGet: (...a: unknown[]) => secretGet(...a),
    discoverServers: (...a: unknown[]) => discoverServers(...(a as [])),
    testRemoteHost: (...a: unknown[]) => testRemoteHost(...a),
    setRemoteHost: (...a: unknown[]) => setRemoteHost(...a),
    forgetRemoteHost: (...a: unknown[]) => forgetRemoteHost(...a),
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
  setRemoteHost.mockClear();
  secretGet.mockClear();
  secretGet.mockResolvedValue(null);
  forgetRemoteHost.mockClear();
  savedHosts = [];
});

describe("EngineSection discovery", () => {
  it("renders exactly one selected engine mode", async () => {
    discoverServers.mockResolvedValue([]);
    const wrapper = await mountSection();
    const connection = useConnectionStore();
    connection.info = { mode: "external", baseUrl: "http://127.0.0.1:7680", apiKey: null };
    await wrapper.vm.$nextTick();

    const group = wrapper.get('[role="group"][aria-label="Engine"]');
    const choices = group.findAll('[data-test="engine-mode-choice"]');
    expect(choices).toHaveLength(2);
    expect(choices.map((choice) => choice.attributes("aria-pressed"))).toEqual(["true", "false"]);
    expect(choices.map((choice) => choice.attributes("tabindex"))).toEqual([undefined, undefined]);

    connection.info = { mode: "remote", baseUrl: "http://hal9000:7680", apiKey: null };
    await wrapper.vm.$nextTick();
    expect(choices.map((choice) => choice.attributes("aria-pressed"))).toEqual(["false", "true"]);
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

  it("reveals and copies the desktop local-server API key", async () => {
    discoverServers.mockResolvedValue([]);
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

describe("EngineSection recent hosts", () => {
  const saved: SavedHost = {
    id: "hal9000-7680",
    name: "hal9000",
    url: "http://hal9000:7680",
    lastUsedMs: Date.now() - 3_600_000,
  };

  it("renders remembered hosts with their last-used time", async () => {
    savedHosts = [saved];
    discoverServers.mockResolvedValue([]);
    const wrapper = await mountSection();
    const text = wrapper.text();
    expect(text).toContain("Recent hosts");
    expect(text).toContain("hal9000");
    expect(text).toContain("1h ago");
  });

  it("hides the section when nothing is remembered", async () => {
    discoverServers.mockResolvedValue([]);
    const wrapper = await mountSection();
    expect(wrapper.text()).not.toContain("Recent hosts");
  });

  it("Connect uses the host's own stored API key", async () => {
    savedHosts = [saved];
    secretGet.mockResolvedValue("per-host-key");
    discoverServers.mockResolvedValue([]);
    const wrapper = await mountSection();
    const connectBtn = wrapper
      .findAll("[data-test='saved-host-connect']")
      .find((b) => b.text() === "Connect");
    await connectBtn!.trigger("click");
    await flushPromises();
    expect(secretGet).toHaveBeenCalledWith("remote-api-key.hal9000-7680");
    expect(setRemoteHost).toHaveBeenCalledWith("http://hal9000:7680", "per-host-key", "hal9000");
  });

  it("Forget is a two-step confirm that drops the host", async () => {
    savedHosts = [saved];
    discoverServers.mockResolvedValue([]);
    const wrapper = await mountSection();
    const forgetBtn = wrapper.get("[data-test='saved-host-forget']");
    await forgetBtn.trigger("click");
    expect(forgetRemoteHost).not.toHaveBeenCalled();
    expect(forgetBtn.text()).toContain("Forget?");
    await forgetBtn.trigger("click");
    await flushPromises();
    expect(forgetRemoteHost).toHaveBeenCalledWith("hal9000-7680");
  });

  it("no longer advertises the Keychain", async () => {
    discoverServers.mockResolvedValue([]);
    const wrapper = await mountSection();
    expect(wrapper.text()).not.toContain("KEYCHAIN");
  });
});
