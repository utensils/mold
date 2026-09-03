/**
 * The generation-host chip. It used to live in the Create header; the
 * redesign moved it into the inspector's "Where it runs" row, so these
 * routing invariants now mount the component directly — the chip is the
 * authority for the persisted `generateTargetHost` contract wherever it is
 * rendered.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import HostChip from "./HostChip.vue";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    appSettingsGet: vi.fn().mockResolvedValue({}),
  },
}));

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => (document.body.innerHTML = ""));

function readyLocal() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
  conn.status = "ready";
  useHostsStore().initialized = true;
}

function addRemote(id = "hal9000-7680", label = "hal9000") {
  useHostsStore().extras.push({
    id,
    label,
    url: `http://${label}:7680`,
    apiKey: null,
    status: "ready",
    error: null,
    instanceId: null,
  });
}

describe("HostChip", () => {
  it("does not open a routing menu with a single host", async () => {
    readyLocal();
    const wrapper = mount(HostChip, { attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });

  it("toggles the routing menu open and closed from the chip", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: null } as never;
    addRemote();
    const wrapper = mount(HostChip, { attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(true);
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });

  it("lists Auto, Most capable, and every host; picking one persists and closes", async () => {
    readyLocal();
    const prefs = useAppPrefsStore();
    prefs.settings = { generateTargetHost: null } as never;
    const update = vi.spyOn(prefs, "update").mockResolvedValue(undefined as never);
    addRemote();
    const wrapper = mount(HostChip, { attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.find("[data-test='host-option-auto']").exists()).toBe(true);
    expect(wrapper.find("[data-test='host-option-capable']").exists()).toBe(true);
    await wrapper.get("[data-test='host-option-hal9000-7680']").trigger("click");
    await flushPromises();
    expect(update).toHaveBeenCalledWith({ generateTargetHost: "hal9000-7680" });
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });

  it("maps Auto back to null in the persisted setting", async () => {
    readyLocal();
    const prefs = useAppPrefsStore();
    prefs.settings = { generateTargetHost: "hal9000-7680" } as never;
    const update = vi.spyOn(prefs, "update").mockResolvedValue(undefined as never);
    addRemote();
    const wrapper = mount(HostChip, { attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    await wrapper.get("[data-test='host-option-auto']").trigger("click");
    expect(update).toHaveBeenCalledWith({ generateTargetHost: null });
  });

  it("shows a stale persisted pick as Auto when the host is gone", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: "ghost-7680" } as never;
    addRemote();
    const wrapper = mount(HostChip, { attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.get("[data-test='host-option-auto']").attributes("aria-checked")).toBe("true");
  });

  it("names the sticky pick on the chip", () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    addRemote();
    const wrapper = mount(HostChip);
    expect(wrapper.get("[data-test='host-chip']").text()).toContain("hal9000");
  });

  it("closes the menu on Escape", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: null } as never;
    addRemote();
    const wrapper = mount(HostChip, { attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await flushPromises();
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });
});
