import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";

const testRemoteHost = vi.fn();
const discoverServers = vi.fn();
vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    testRemoteHost: (...a: unknown[]) => testRemoteHost(...a),
    secretSet: vi.fn().mockResolvedValue(undefined),
    secretGet: vi.fn().mockResolvedValue(null),
    appSettingsGet: vi
      .fn()
      .mockResolvedValue({ savedHosts: [], connectedHostIds: [], generateTargetHost: null }),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    discoverServers: (...a: unknown[]) => discoverServers(...(a as [])),
  },
}));
vi.mock("../../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/api/client")>()),
  apiJsonTo: vi.fn().mockResolvedValue({ queue_depth: 0, queue_capacity: 8, version: null }),
}));

import ConnectMachineModal from "./ConnectMachineModal.vue";
import { useConnectionStore } from "../../stores/connection";

const stub = { template: "<div />" };
let router: Router;

async function mountModal() {
  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      { path: "/machines/runpod", component: stub },
    ],
  });
  router.push("/");
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  const wrapper = mount(ConnectMachineModal, {
    props: { open: true },
    global: { plugins: [pinia, router] },
  });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  vi.clearAllMocks();
  discoverServers.mockResolvedValue([]);
  testRemoteHost.mockResolvedValue({
    ok: true,
    version: "1",
    error: null,
    instanceId: null,
    hostname: "hal9000",
  });
});

describe("ConnectMachineModal", () => {
  it("walks type → details → confirmation for a remote server", async () => {
    const wrapper = await mountModal();
    expect(wrapper.find("[data-test='connect-type-remote']").exists()).toBe(true);
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    const address = wrapper.get("[data-test='connect-address']");
    await address.setValue("hal9000");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    expect(testRemoteHost).toHaveBeenCalledWith("http://hal9000:7680", null);
    const confirm = wrapper.get("[data-test='connect-confirm']");
    expect(confirm.text()).toContain("hal9000");
    expect(confirm.text()).toContain("online and ready");
  });

  it("hands the RunPod type off to the provisioning view and closes", async () => {
    const wrapper = await mountModal();
    await wrapper.get("[data-test='connect-type-runpod']").trigger("click");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/machines/runpod");
    expect(wrapper.emitted("close")).toBeTruthy();
  });

  it("keeps the entered address and shows a blunt error when the connect fails", async () => {
    testRemoteHost.mockResolvedValue({
      ok: false,
      version: null,
      error: "Connection refused.",
      instanceId: null,
      hostname: null,
    });
    const wrapper = await mountModal();
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();
    const address = wrapper.get("[data-test='connect-address']");
    await address.setValue("hal9000");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='connect-error']").text()).toContain("Connection refused.");
    // Still on the details step with the value preserved.
    expect((wrapper.get("[data-test='connect-address']").element as HTMLInputElement).value).toBe(
      "hal9000",
    );
    expect(wrapper.find("[data-test='connect-confirm']").exists()).toBe(false);
  });

  it("offers the live discovered list for Local network and connects a pick", async () => {
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
    const wrapper = await mountModal();
    await wrapper.get("[data-test='connect-type-lan']").trigger("click");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    const row = wrapper.get("[data-test='connect-discovered']");
    expect(row.text()).toContain("studio-7680");
    await row.get("[data-test='connect-discovered-add']").trigger("click");
    await flushPromises();
    expect(testRemoteHost).toHaveBeenCalledWith("http://192.168.1.20:7680", null);
    expect(wrapper.find("[data-test='connect-confirm']").exists()).toBe(true);
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
    const wrapper = await mountModal();
    await wrapper.get("[data-test='connect-type-lan']").trigger("click");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    await wrapper.get("[data-test='connect-discovered-add']").trigger("click");
    await flushPromises();
    expect(wrapper.get("[data-test='connect-discovered-selected']").text()).toContain(
      "locked-7680",
    );
    expect(testRemoteHost).not.toHaveBeenCalled();

    await wrapper.get("[data-test='connect-key']").setValue("peer-secret");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();
    expect(testRemoteHost).toHaveBeenCalledWith("http://192.168.1.30:7680", "peer-secret");
  });

  it("clears a direct discovered-host prompt when navigating back to LAN discovery", async () => {
    const initialHost = {
      name: "locked-7680",
      url: "http://192.168.1.30:7680",
      host: "192.168.1.30",
      port: 7680,
      version: "1",
      authRequired: true,
      isThisMachine: false,
    };
    const wrapper = await mountModal();
    await wrapper.setProps({ open: false, initialHost });
    await wrapper.setProps({ open: true });
    await flushPromises();
    expect(wrapper.find("[data-test='connect-discovered-selected']").exists()).toBe(true);

    await wrapper.get("[data-test='connect-back']").trigger("click");
    await wrapper.get("[data-test='connect-type-lan']").trigger("click");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='connect-discovered-selected']").exists()).toBe(false);
  });
});
