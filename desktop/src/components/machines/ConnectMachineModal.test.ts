import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

const testRemoteHost = vi.fn();
const discoverServers = vi.fn();
const appSettingsSet = vi.fn().mockResolvedValue(undefined);
vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    testRemoteHost: (...a: unknown[]) => testRemoteHost(...a),
    secretSet: vi.fn().mockResolvedValue(undefined),
    secretGet: vi.fn().mockResolvedValue(null),
    appSettingsGet: vi
      .fn()
      .mockResolvedValue({ savedHosts: [], connectedHostIds: [], generateTargetHost: null }),
    appSettingsSet: (...a: unknown[]) => appSettingsSet(...a),
    discoverServers: (...a: unknown[]) => discoverServers(...(a as [])),
  },
}));
vi.mock("../../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../lib/api/client")>()),
  apiJsonTo: vi.fn().mockResolvedValue({ queue_depth: 0, queue_capacity: 8, version: null }),
}));

import ConnectMachineModal from "./ConnectMachineModal.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";

const LOCKED = {
  name: "locked-7680",
  url: "http://192.168.1.30:7680",
  host: "192.168.1.30",
  port: 7680,
  version: "1",
  authRequired: true,
  isThisMachine: false,
};

async function mountModal(props: Record<string, unknown> = {}) {
  const pinia = createPinia();
  setActivePinia(pinia);
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  const wrapper = mount(ConnectMachineModal, {
    props: { open: true, ...props },
    global: { plugins: [pinia] },
  });
  await flushPromises();
  return wrapper;
}

/** The same modal with the hosts store's `connect` stubbed. The spy is placed
 *  BEFORE the mount, on the very pinia the component is given, so what the
 *  dialog hands the store is read off one object and not two. */
async function mountWithConnectSpy() {
  const pinia = createPinia();
  setActivePinia(pinia);
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
  conn.status = "ready";
  const connect = vi
    .spyOn(useHostsStore(), "connect")
    .mockResolvedValue({ id: "connected-host" } as never);
  const wrapper = mount(ConnectMachineModal, {
    props: { open: true },
    global: { plugins: [pinia] },
  });
  await flushPromises();
  return { wrapper, connect };
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
  it("connects a typed address, then reports and closes", async () => {
    const wrapper = await mountModal();
    await wrapper.get("[data-test='connect-address']").setValue("hal9000");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    expect(testRemoteHost).toHaveBeenCalledWith("http://hal9000:7680", null);
    expect(wrapper.emitted("connected")).toHaveLength(1);
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("names a hand-typed machine at connect time", async () => {
    // A discovered machine arrives with a name; a typed address sent null and
    // got whatever label the store could derive from the URL, so the one
    // moment the user knows what to call the machine had no field for it.
    const { wrapper, connect } = await mountWithConnectSpy();
    await wrapper.get("[data-test='connect-address']").setValue("hal9000");
    await wrapper.get("[data-test='connect-name']").setValue("Render box");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    expect(connect).toHaveBeenCalledWith("hal9000", null, "Render box");
  });

  it("keeps a discovered machine's own name when the field is left empty", async () => {
    discoverServers.mockResolvedValue([{ ...LOCKED, name: "studio-7680", authRequired: false }]);
    const { wrapper, connect } = await mountWithConnectSpy();
    await wrapper.get("[data-test='connect-discovered']").trigger("click");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    expect(connect).toHaveBeenCalledWith("http://192.168.1.30:7680", null, "studio-7680");
  });

  it("probes a bare IP through the default protocol and port", async () => {
    const wrapper = await mountModal();
    await wrapper.get("[data-test='connect-address']").setValue("100.123.198.98");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    expect(testRemoteHost).toHaveBeenCalledWith("http://100.123.198.98:7680", null);
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
    await wrapper.get("[data-test='connect-address']").setValue("hal9000");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    expect(wrapper.get("[data-test='connect-error']").text()).toContain("Connection refused.");
    expect((wrapper.get("[data-test='connect-address']").element as HTMLInputElement).value).toBe(
      "hal9000",
    );
    expect(wrapper.emitted("close")).toBeUndefined();
  });

  it("lists the machines found on the network and connects the picked one", async () => {
    discoverServers.mockResolvedValue([{ ...LOCKED, name: "studio-7680", authRequired: false }]);
    const wrapper = await mountModal();

    const row = wrapper.get("[data-test='connect-discovered']");
    expect(row.text()).toContain("studio-7680");
    await row.trigger("click");
    expect(wrapper.get("[data-test='connect-discovered-selected']").text()).toContain("selected");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    expect(testRemoteHost).toHaveBeenCalledWith("http://192.168.1.30:7680", null);
    expect(wrapper.emitted("connected")).toHaveLength(1);
  });

  it("holds Connect until an authenticated pick has its key", async () => {
    discoverServers.mockResolvedValue([LOCKED]);
    const wrapper = await mountModal();
    await wrapper.get("[data-test='connect-discovered']").trigger("click");

    const connect = wrapper.get("[data-test='connect-continue']");
    expect(connect.attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("this machine asks for one");

    await wrapper.get("[data-test='connect-key']").setValue("peer-secret");
    await connect.trigger("click");
    await flushPromises();
    expect(testRemoteHost).toHaveBeenCalledWith("http://192.168.1.30:7680", "peer-secret");
  });

  it("opens straight on a machine that asked for a key, and lets a typed address replace a pick", async () => {
    const wrapper = await mountModal({ open: false });
    await wrapper.setProps({ open: true, initialHost: LOCKED });
    await flushPromises();
    expect(wrapper.get("[data-test='connect-discovered-selected']").text()).toContain(
      "locked-7680",
    );
    expect(discoverServers).not.toHaveBeenCalled();
    expect(wrapper.find("[data-test='connect-address']").exists()).toBe(false);

    // A pick from a scan can be swapped out by hand.
    discoverServers.mockResolvedValue([LOCKED]);
    await wrapper.setProps({ open: false, initialHost: null });
    await wrapper.setProps({ open: true });
    await flushPromises();
    await wrapper.get("[data-test='connect-discovered']").trigger("click");
    expect(wrapper.find("[data-test='connect-discovered-selected']").exists()).toBe(true);
    await wrapper.get("[data-test='connect-address']").setValue("hal9000");
    expect(wrapper.find("[data-test='connect-discovered-selected']").exists()).toBe(false);
  });

  it("makes the new machine the generation target when asked", async () => {
    const wrapper = await mountModal();
    await wrapper.get("[data-test='connect-address']").setValue("hal9000");
    await wrapper.get("[data-test='connect-make-target']").trigger("click");
    await wrapper.get("[data-test='connect-continue']").trigger("click");
    await flushPromises();

    expect(useAppPrefsStore().settings?.generateTargetHost).toBe("hal9000-7680");
    expect(appSettingsSet).toHaveBeenCalledWith(
      expect.objectContaining({ generateTargetHost: "hal9000-7680" }),
    );
  });

  it("sets `mold serve` in the mono face inside the dialog's sentence", async () => {
    const wrapper = await mountModal();
    const code = wrapper.get(".ms-modal__desc code");
    expect(code.text()).toBe("mold serve");
    expect(code.classes()).toContain("font-mono");
  });
});
