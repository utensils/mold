import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import HostsSection from "./HostsSection.vue";
import { useConnectionStore } from "../../stores/connection";

// The slimmed section no longer does host CRUD; the store's refresh poll still
// hits /api/status, so keep it inert.
vi.mock("../../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/api/client")>()),
  apiJsonTo: vi.fn().mockResolvedValue({ queue_depth: 0, queue_capacity: 8, version: null }),
}));

vi.mock("./ConfigSettingRow.vue", () => ({ default: { template: "<div />" } }));

const stub = { template: "<div />" };
let router: Router;

async function mountSection() {
  setActivePinia(createPinia());
  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      { path: "/machines", component: stub },
      { path: "/settings", component: stub },
    ],
  });
  router.push("/settings");
  await router.isReady();
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  const wrapper = mount(HostsSection, { global: { plugins: [router] } });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  vi.clearAllMocks();
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

describe("HostsSection points to Machines", () => {
  it("links host management to the Machines workspace", async () => {
    const wrapper = await mountSection();
    const link = wrapper.get('[data-test="open-machines"]');
    expect(link.text()).toContain("Open Machines");
    await link.trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/machines");
  });

  it("no longer renders host CRUD (add / connected / remembered / discovery)", async () => {
    const wrapper = await mountSection();
    expect(wrapper.find('[data-test="add-host"]').exists()).toBe(false);
    expect(wrapper.find('[data-test="connected-host"]').exists()).toBe(false);
    expect(wrapper.find('[data-test="discovered-host"]').exists()).toBe(false);
    expect(wrapper.find('[data-test="saved-host-connect"]').exists()).toBe(false);
  });
});
