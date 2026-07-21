import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import CreateHeader from "./CreateHeader.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
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

function form(): GenerateForm {
  return reactive({ ...newGenerateForm(), family: "flux" });
}

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

describe("CreateHeader", () => {
  it("renders the live summary of shape, dimensions, and steps", () => {
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() } });
    expect(wrapper.get(".ms-header__summary").text()).toBe("1:1 · 1024×1024 · 4 steps");
  });

  it("does not open a routing menu with a single host", async () => {
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });

  it("toggles the routing menu open and closed from the chip", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: null } as never;
    addRemote();
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
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
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
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
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    await wrapper.get("[data-test='host-option-auto']").trigger("click");
    expect(update).toHaveBeenCalledWith({ generateTargetHost: null });
  });

  it("shows a stale persisted pick as Auto when the host is gone", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: "ghost-7680" } as never;
    addRemote();
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.get("[data-test='host-option-auto']").attributes("aria-checked")).toBe("true");
  });

  it("names the sticky pick on the chip", () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    addRemote();
    const wrapper = mount(CreateHeader, { props: { form: form() } });
    expect(wrapper.get("[data-test='host-chip']").text()).toContain("hal9000");
  });

  it("closes the menu on Escape", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: null } as never;
    addRemote();
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get("[data-test='host-chip']").trigger("click");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await flushPromises();
    expect(wrapper.find("[data-test='host-menu']").exists()).toBe(false);
  });
});
