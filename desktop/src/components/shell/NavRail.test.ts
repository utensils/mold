import { describe, expect, it } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import NavRail from "./NavRail.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useGalleryStore } from "../../stores/gallery";

const stub = { template: "<div />" };

function makeRouter(): Router {
  return createRouter({
    history: createMemoryHistory(),
    routes: ["/create", "/library", "/models", "/settings", "/machines", "/machines/:id"].map(
      (path) => ({
        path,
        component: stub,
      }),
    ),
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

describe("NavRail a11y", () => {
  it("labels the primary navigation landmark", async () => {
    const wrapper = await mountAt("/create");
    expect(wrapper.get("nav").attributes("aria-label")).toBe("Primary");
  });

  it("marks the active nav item with aria-current=page", async () => {
    const wrapper = await mountAt("/library");
    // Destinations render as @ui NavItem buttons now, not RouterLink anchors.
    const buttons = wrapper.findAll("button");
    const library = buttons.find((b) => b.text().includes("Library"));
    const create = buttons.find((b) => b.text().includes("Create"));
    expect(library?.attributes("aria-current")).toBe("page");
    expect(create?.attributes("aria-current")).toBeUndefined();
  });

  it("collapses to the five destinations plus Settings", async () => {
    const wrapper = await mountAt("/create");
    for (const label of ["Create", "Library", "Models", "Machines", "Settings"]) {
      expect(wrapper.text()).toContain(label);
    }
    // The folded destinations are gone from the rail (still deep-linkable).
    for (const label of ["Generate", "Gallery", "Chains", "Catalog", "History", "RunPod"]) {
      expect(wrapper.text()).not.toContain(label);
    }
  });
});

describe("NavRail collapse", () => {
  function setCollapsed(collapsed: boolean) {
    const prefs = useAppPrefsStore();
    prefs.settings = { sidebarCollapsed: collapsed } as never;
  }

  it("expands to 210px with visible labels by default", async () => {
    const wrapper = await mountAt("/create");
    expect(wrapper.get("nav").attributes("style")).toContain("210px");
    expect(wrapper.text()).toContain("Create");
    // Gradient wordmark shows when expanded.
    expect(wrapper.find(".ms-wordmark").exists()).toBe(true);
  });

  it("collapses to a 62px icon rail with labels and wordmark hidden", async () => {
    const wrapper = await mountAt("/create");
    setCollapsed(true);
    await flushPromises();
    expect(wrapper.get("nav").attributes("style")).toContain("62px");
    // Icon-only: nav labels and the wordmark are gone.
    expect(wrapper.text()).not.toContain("Create");
    expect(wrapper.find(".ms-wordmark").exists()).toBe(false);
    // Destinations are still present as accessible icon buttons.
    const buttons = wrapper.findAll("button");
    expect(buttons.some((b) => b.attributes("aria-label") === "Create")).toBe(true);
  });
});

describe("NavRail workspace badges (G11)", () => {
  it("shows a stop dot on Machines when a connected host is offline", async () => {
    const wrapper = await mountAt("/create");
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "error",
      error: "down",
      instanceId: null,
    });
    await flushPromises();
    expect(wrapper.find("[data-test='machines-error-dot']").exists()).toBe(true);
  });

  it("hides the Machines stop dot while every host is reachable", async () => {
    const wrapper = await mountAt("/create");
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
    await flushPromises();
    expect(wrapper.find("[data-test='machines-error-dot']").exists()).toBe(false);
  });

  it("badges Library with the count of prints developed since the last visit", async () => {
    const wrapper = await mountAt("/create");
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
    const gallery = useGalleryStore();
    const image = (filename: string, timestamp: number) =>
      ({ filename, timestamp, metadata: { prompt: "p" } }) as never;
    gallery.buckets.local = {
      items: [image("a.png", 1)],
      loading: false,
      error: null,
      loaded: true,
    };
    gallery.markLibrarySeen();
    gallery.buckets.local.items = [image("b.png", 3), image("a.png", 1)];
    await flushPromises();
    const badge = wrapper.find(".ms-nav__badge");
    expect(badge.exists()).toBe(true);
    expect(badge.text()).toBe("1");
  });
});
