import { afterEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import NavRail from "./NavRail.vue";
import { ipc } from "../../lib/ipc";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useConnectionStore } from "../../stores/connection";
import { useContextMenuStore, type MenuItem } from "../../stores/contextMenu";
import { useHostsStore, type HostView } from "../../stores/hosts";

const stub = { template: "<div />" };

function makeRouter(): Router {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      "/generate",
      "/gallery",
      "/chains",
      "/models",
      "/history",
      "/jobs",
      "/runpod",
      "/settings",
      "/machines",
      "/machines/:id",
      "/hosts/:id",
    ].map((path) => ({
      path,
      component: stub,
    })),
  });
}

let router: Router;

async function mountAt(path: string) {
  router = makeRouter();
  router.push(path);
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  return mount(NavRail, {
    global: {
      plugins: [pinia, router],
      // StatusPopover opens its own telemetry streams on mount; the rail tests
      // don't exercise it, so stub it out to keep them off the network.
      stubs: { StatusPopover: stub },
    },
  });
}

describe("NavRail hosts section", () => {
  it("shows connected hosts with status and queue depth", async () => {
    const wrapper = await mountAt("/generate");
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
    conn.status = "ready";
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    hosts.telemetry["hal9000-7680"] = { queueDepth: 3, queueCapacity: 8, version: null };
    await flushPromises();
    const rows = wrapper.findAll("[data-test='host-row']");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.text()).toContain("This device");
    expect(rows[1]!.text()).toContain("hal9000");
    expect(rows[1]!.text()).toContain("3");
  });

  it("left-click on a host row navigates to its detail page", async () => {
    const wrapper = await mountAt("/generate");
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
    conn.status = "ready";
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    await flushPromises();
    const rows = wrapper.findAll("[data-test='host-row']");
    await rows[1]!.trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/machines/hal9000-7680");
    await rows[0]!.trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/machines/local");
  });

  it("shows an empty message when nothing is connected or detected", async () => {
    const wrapper = await mountAt("/generate");
    await flushPromises();
    expect(wrapper.get("[data-test='hosts-section']").text()).toContain("No hosts");
  });
});

describe("NavRail host context menu", () => {
  async function mountWithHosts() {
    const wrapper = await mountAt("/generate");
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
    conn.status = "ready";
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    await flushPromises();
    return { wrapper, hosts };
  }

  function menuLabels(): string[] {
    return useContextMenuStore()
      .entries.filter((e): e is MenuItem => !("separator" in e))
      .map((e) => e.label);
  }

  it("offers routing, web UI, copy URL, rename and disconnect for a ready extra", async () => {
    const { wrapper } = await mountWithHosts();
    const rows = wrapper.findAll("[data-test='host-row']");
    await rows[1]!.trigger("contextmenu");
    const labels = menuLabels();
    expect(labels).toContain("Set as generation target");
    expect(labels).toContain("View gallery");
    expect(labels).toContain("Open web UI");
    expect(labels).toContain("Copy URL");
    expect(labels).toContain("Rename…");
    expect(labels).toContain("Disconnect");
    expect(labels).toContain("Forget");
    expect(labels).not.toContain("Reconnect");
    // View gallery sits right under the routing entry.
    expect(labels.indexOf("View gallery")).toBe(labels.indexOf("Set as generation target") + 1);
  });

  it("View gallery deep-links to the host-filtered gallery", async () => {
    const { wrapper } = await mountWithHosts();
    const rows = wrapper.findAll("[data-test='host-row']");
    await rows[1]!.trigger("contextmenu");
    const entry = useContextMenuStore()
      .entries.filter((e): e is MenuItem => !("separator" in e))
      .find((e) => e.label === "View gallery");
    entry?.action?.();
    await flushPromises();
    const route = router.currentRoute.value;
    expect(route.path).toBe("/gallery");
    expect(route.query.host).toBe("hal9000-7680");
  });

  it("the local primary gets View gallery too, keyed by its host id", async () => {
    const { wrapper } = await mountWithHosts();
    const rows = wrapper.findAll("[data-test='host-row']");
    await rows[0]!.trigger("contextmenu");
    const entry = useContextMenuStore()
      .entries.filter((e): e is MenuItem => !("separator" in e))
      .find((e) => e.label === "View gallery");
    expect(entry).toBeDefined();
    entry?.action?.();
    await flushPromises();
    expect(router.currentRoute.value.query.host).toBe("local");
  });

  it("offers reconnect for an errored extra", async () => {
    const { wrapper, hosts } = await mountWithHosts();
    hosts.extras[0]!.status = "error";
    await flushPromises();
    const rows = wrapper.findAll("[data-test='host-row']");
    await rows[1]!.trigger("contextmenu");
    expect(menuLabels()).toContain("Reconnect");
  });

  it("offers settings (but never disconnect/forget) for the local primary", async () => {
    const { wrapper } = await mountWithHosts();
    const rows = wrapper.findAll("[data-test='host-row']");
    await rows[0]!.trigger("contextmenu");
    const labels = menuLabels();
    expect(labels).toContain("Manage in Settings");
    expect(labels).not.toContain("Disconnect");
    expect(labels).not.toContain("Forget");
    expect(labels).not.toContain("Rename…");
    expect(labels).not.toContain("Switch to built-in engine");
  });
});

describe("NavRail detected hosts", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("connectDetected finds a stored key under the saved instance-id twin's slug", async () => {
    // hal9000 was remembered (and keyed) by hostname; mDNS advertises the
    // same box by IP under a different slug but the same instance id. The
    // rail must find the stored key instead of bouncing to Settings.
    const base = await ipc.appSettingsGet();
    vi.spyOn(ipc, "discoverServers").mockResolvedValue([
      {
        name: "hal9000",
        url: "http://192.168.1.50:7680",
        host: "192.168.1.50",
        port: 7680,
        version: "0.18.0",
        authRequired: true,
        instanceId: "uuid-1",
        isThisMachine: false,
      },
    ]);
    vi.spyOn(ipc, "appSettingsGet").mockResolvedValue({
      ...base,
      savedHosts: [
        {
          id: "hal9000-7680",
          name: "hal9000",
          url: "http://hal9000:7680",
          lastUsedMs: 1,
          instanceId: "uuid-1",
        },
      ],
    });
    vi.spyOn(ipc, "secretGet").mockImplementation((name) =>
      Promise.resolve(name === "remote-api-key.hal9000-7680" ? "stored-key" : null),
    );
    const wrapper = await mountAt("/generate");
    await flushPromises();
    const hosts = useHostsStore();
    const connect = vi.spyOn(hosts, "connect").mockResolvedValue({
      id: "hal9000-7680",
      label: "hal9000",
      kind: "remote",
      baseUrl: "http://192.168.1.50:7680",
      apiKey: "stored-key",
      status: "ready",
      primary: false,
      queueDepth: null,
      queueCapacity: null,
      version: null,
      instanceId: "uuid-1",
    } satisfies HostView);
    await wrapper.get("[data-test='detected-host-connect']").trigger("click");
    await flushPromises();
    expect(connect).toHaveBeenCalledWith("http://192.168.1.50:7680", "stored-key", "hal9000");
    // Not bounced to Settings for a key the app already holds.
    expect(router.currentRoute.value.path).toBe("/generate");
  });

  it("right-click on a detected (disconnected) host opens a context menu", async () => {
    vi.spyOn(ipc, "discoverServers").mockResolvedValue([
      {
        name: "hal9000",
        url: "http://192.168.1.50:7680",
        host: "192.168.1.50",
        port: 7680,
        version: "0.18.0",
        authRequired: false,
        instanceId: "uuid-1",
        isThisMachine: false,
      },
    ]);
    const wrapper = await mountAt("/generate");
    await flushPromises();
    await wrapper.get("[data-test='detected-host-row']").trigger("contextmenu");
    const labels = useContextMenuStore()
      .entries.filter((e): e is MenuItem => !("separator" in e))
      .map((e) => e.label);
    expect(labels).toContain("Connect");
    expect(labels).toContain("Open web UI");
    expect(labels).toContain("Copy URL");
    // Not remembered — nothing to forget.
    expect(labels).not.toContain("Forget");
  });

  it("detected-host menu offers Forget when the box is remembered under any slug", async () => {
    const base = await ipc.appSettingsGet();
    vi.spyOn(ipc, "discoverServers").mockResolvedValue([
      {
        name: "hal9000",
        url: "http://192.168.1.50:7680",
        host: "192.168.1.50",
        port: 7680,
        version: "0.18.0",
        authRequired: false,
        instanceId: "uuid-1",
        isThisMachine: false,
      },
    ]);
    vi.spyOn(ipc, "appSettingsGet").mockResolvedValue({
      ...base,
      savedHosts: [
        {
          id: "hal9000-7680",
          name: "hal9000",
          url: "http://hal9000:7680",
          lastUsedMs: 1,
          instanceId: "uuid-1",
        },
      ],
    });
    const forget = vi.spyOn(ipc, "forgetRemoteHost").mockResolvedValue([]);
    const wrapper = await mountAt("/generate");
    await flushPromises();
    await wrapper.get("[data-test='detected-host-row']").trigger("contextmenu");
    const menu = useContextMenuStore();
    const forgetEntry = menu.entries.find(
      (e): e is MenuItem => !("separator" in e) && e.label === "Forget",
    );
    expect(forgetEntry).toBeDefined();
    await forgetEntry!.action?.();
    await flushPromises();
    // Forgets the remembered slug (hostname), not the advertised IP slug.
    expect(forget).toHaveBeenCalledWith("hal9000-7680");
  });
});

describe("NavRail a11y", () => {
  it("labels the primary navigation landmark", async () => {
    const wrapper = await mountAt("/generate");
    expect(wrapper.get("nav").attributes("aria-label")).toBe("Primary");
  });

  it("marks the active nav item with aria-current=page", async () => {
    const wrapper = await mountAt("/gallery");
    // Destinations render as @ui NavItem buttons now, not RouterLink anchors.
    const buttons = wrapper.findAll("button");
    const gallery = buttons.find((b) => b.text().includes("Gallery"));
    const generate = buttons.find((b) => b.text().includes("Generate"));
    expect(gallery?.attributes("aria-current")).toBe("page");
    expect(generate?.attributes("aria-current")).toBeUndefined();
  });

  it("keeps all eight destinations", async () => {
    const wrapper = await mountAt("/generate");
    for (const label of [
      "Generate",
      "Gallery",
      "Chains",
      "Catalog",
      "History",
      "Jobs",
      "RunPod",
      "Settings",
    ]) {
      expect(wrapper.text()).toContain(label);
    }
  });
});

describe("NavRail collapse", () => {
  function setCollapsed(collapsed: boolean) {
    const prefs = useAppPrefsStore();
    prefs.settings = { sidebarCollapsed: collapsed } as never;
  }

  it("expands to 210px with visible labels by default", async () => {
    const wrapper = await mountAt("/generate");
    expect(wrapper.get("nav").attributes("style")).toContain("210px");
    expect(wrapper.text()).toContain("Generate");
    // Gradient wordmark shows when expanded.
    expect(wrapper.find(".ms-wordmark").exists()).toBe(true);
  });

  it("collapses to a 62px icon rail with labels and wordmark hidden", async () => {
    const wrapper = await mountAt("/generate");
    setCollapsed(true);
    await flushPromises();
    expect(wrapper.get("nav").attributes("style")).toContain("62px");
    // Icon-only: nav labels and the wordmark are gone.
    expect(wrapper.text()).not.toContain("Generate");
    expect(wrapper.find(".ms-wordmark").exists()).toBe(false);
    // Destinations are still present as accessible icon buttons.
    const buttons = wrapper.findAll("button");
    expect(buttons.some((b) => b.attributes("aria-label") === "Generate")).toBe(true);
  });
});
