import { describe, expect, it } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import NavRail from "./NavRail.vue";
import { useConnectionStore } from "../../stores/connection";
import { useContextMenuStore, type MenuItem } from "../../stores/contextMenu";
import { useHostsStore } from "../../stores/hosts";

const stub = { template: "<div />" };

function makeRouter(): Router {
  return createRouter({
    history: createMemoryHistory(),
    routes: ["/generate", "/gallery", "/chains", "/models", "/history", "/runpod", "/settings"].map(
      (path) => ({
        path,
        component: stub,
      }),
    ),
  });
}

async function mountAt(path: string) {
  const router = makeRouter();
  router.push(path);
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  return mount(NavRail, {
    global: {
      plugins: [pinia, router],
      // DevelopCanvas paints to <canvas>, which happy-dom can't; stub it out
      // (it only renders inside job rows, of which there are none here anyway).
      stubs: { DevelopCanvas: stub },
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
    });
    hosts.telemetry["hal9000-7680"] = { queueDepth: 3, queueCapacity: 8, version: null };
    await flushPromises();
    const rows = wrapper.findAll("[data-test='host-row']");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.text()).toContain("This Mac");
    expect(rows[1]!.text()).toContain("hal9000");
    expect(rows[1]!.text()).toContain("3");
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
    expect(labels).toContain("Open web UI");
    expect(labels).toContain("Copy URL");
    expect(labels).toContain("Rename…");
    expect(labels).toContain("Disconnect");
    expect(labels).toContain("Forget");
    expect(labels).not.toContain("Reconnect");
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

  it("offers switch-to-built-in and rename for a remote primary", async () => {
    const wrapper = await mountAt("/generate");
    const conn = useConnectionStore();
    conn.info = { mode: "remote", baseUrl: "http://hal9000:7680", apiKey: null };
    conn.status = "ready";
    await flushPromises();
    const rows = wrapper.findAll("[data-test='host-row']");
    await rows[0]!.trigger("contextmenu");
    const labels = menuLabels();
    expect(labels).toContain("Switch to built-in engine");
    expect(labels).toContain("Rename…");
    expect(labels).toContain("Manage in Settings");
  });
});

describe("NavRail a11y", () => {
  it("labels the primary navigation landmark", async () => {
    const wrapper = await mountAt("/generate");
    expect(wrapper.get("nav").attributes("aria-label")).toBe("Primary");
  });

  it("marks the active route link with aria-current=page", async () => {
    const wrapper = await mountAt("/gallery");
    const links = wrapper.findAll("a");
    const gallery = links.find((a) => a.text().includes("Gallery"));
    const generate = links.find((a) => a.text().includes("Generate"));
    expect(gallery?.attributes("aria-current")).toBe("page");
    expect(generate?.attributes("aria-current")).toBeUndefined();
  });
});
