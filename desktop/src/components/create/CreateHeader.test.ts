import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import CreateHeader from "./CreateHeader.vue";
import HostSelector from "../generate/HostSelector.vue";
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

describe("CreateHeader", () => {
  it("renders the live summary of shape, dimensions, and steps", () => {
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() } });
    expect(wrapper.get(".ms-header__summary").text()).toBe("1:1 · 1024×1024 · 4 steps");
  });

  it("does not open a routing popover with a single host", async () => {
    readyLocal();
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get(".ms-header__chip").trigger("click");
    expect(wrapper.findComponent(HostSelector).exists()).toBe(false);
  });

  it("opens the HostSelector popover when more than one host is connected", async () => {
    readyLocal();
    useAppPrefsStore().settings = { generateTargetHost: null } as never;
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    const wrapper = mount(CreateHeader, { props: { form: form() }, attachTo: document.body });
    await wrapper.get(".ms-header__chip").trigger("click");
    await flushPromises();
    expect(wrapper.findComponent(HostSelector).exists()).toBe(true);
  });
});
