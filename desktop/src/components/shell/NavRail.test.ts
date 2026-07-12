import { describe, expect, it } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import NavRail from "./NavRail.vue";
import { useConnectionStore } from "../../stores/connection";
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
