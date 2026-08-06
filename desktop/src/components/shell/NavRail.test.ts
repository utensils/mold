import { describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import NavRail from "./NavRail.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useGalleryStore } from "../../stores/gallery";
import { useGenerationStore } from "../../stores/generation";
import { useChainJobsStore } from "../../stores/chainJobs";
import { useHostModelsStore } from "../../stores/hostModels";

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

describe("NavRail developing jobs", () => {
  it("uses the remaining rail height before scrolling queued jobs", async () => {
    const wrapper = await mountAt("/create");
    const generation = useGenerationStore();
    generation.jobs = [
      {
        clientId: 1,
        model: "flux-dev:q8",
        prompt: "queued print",
        status: "queued",
      } as never,
    ];
    await flushPromises();

    const region = wrapper.get("[data-test='developing-region']");
    const jobs = wrapper.get("[data-test='developing-jobs']");
    expect(wrapper.get("nav").classes()).toEqual(
      expect.arrayContaining(["min-h-0", "overflow-hidden"]),
    );
    expect(region.classes()).toEqual(expect.arrayContaining(["min-h-0", "flex-1"]));
    expect(jobs.classes()).toEqual(
      expect.arrayContaining(["min-h-0", "flex-1", "overflow-y-auto"]),
    );
    expect(jobs.classes()).not.toContain("max-h-44");
  });

  it("uses the available rail space for a longer finished-print history", async () => {
    const wrapper = await mountAt("/create");
    const generation = useGenerationStore();
    generation.jobs = Array.from({ length: 12 }, (_, index) => ({
      clientId: index + 1,
      model: "flux-dev:q8",
      prompt: `finished print ${index + 1}`,
      status: "complete",
    })) as never;
    await flushPromises();

    expect(wrapper.findAll("[data-test='developing-print']")).toHaveLength(12);
    expect(wrapper.get("[data-test='developing-jobs']").classes()).toContain("overflow-y-auto");
  });

  it("reacquires a compacted print when its history row is opened", async () => {
    const wrapper = await mountAt("/create");
    const generation = useGenerationStore();
    const refresh = vi.spyOn(generation, "refreshRemoteResultUrl").mockResolvedValue();
    generation.jobs = [
      {
        clientId: 13,
        model: "flux-dev:q8",
        prompt: "older compacted print",
        status: "complete",
        resultUrl: null,
        result: { filename: "older.png", image: "" },
        request: { prompt: "older compacted print", model: "flux-dev:q8" },
      } as never,
    ];
    await flushPromises();

    await wrapper.get("[data-test='developing-print']").trigger("click");
    expect(refresh).toHaveBeenCalledWith(13);
  });

  // G14 hole: the rail only ever read `generation.jobs`, so a running sequence
  // rendered on the canvas while the sidebar insisted "nothing developing".
  it("shows a running sequence with its clip counter", async () => {
    const wrapper = await mountAt("/create");
    const chains = useChainJobsStore();
    chains.byHost["hal9000-7680"] = {
      jobs: [
        {
          id: "job-1",
          state: "running",
          model: "ltx-2.3-22b-distilled:fp8",
          stage_count: 5,
          current_stage: 2,
          created_at_unix_ms: Date.now(),
          updated_at_unix_ms: Date.now(),
          error: null,
        },
      ],
      error: null,
    };
    await flushPromises();

    const rail = wrapper.get("[data-test='developing-jobs']");
    expect(rail.text()).toContain("clip 3/5");
    expect(rail.text()).toContain("developing");
    expect(wrapper.text()).not.toContain("nothing developing");
  });

  // Settled sequences have two homes already (the print in Library, the job in
  // History) — rebuilding the pile one route away is the thing we removed.
  it("keeps settled sequences out of the rail", async () => {
    const wrapper = await mountAt("/create");
    const chains = useChainJobsStore();
    chains.byHost.local = {
      jobs: [
        {
          id: "job-done",
          state: "completed",
          model: "ltx-video",
          stage_count: 3,
          current_stage: 2,
          created_at_unix_ms: Date.now(),
          updated_at_unix_ms: Date.now(),
          error: null,
        },
      ],
      error: null,
    };
    await flushPromises();

    expect(wrapper.find("[data-test='developing-jobs']").exists()).toBe(false);
    expect(wrapper.text()).toContain("nothing developing");
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

describe("NavRail model labels", () => {
  it("renders the human-readable catalog name for active jobs", async () => {
    const wrapper = await mountAt("/create");
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "plato-7680",
      label: "plato",
      url: "http://plato:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    useHostModelsStore().byHost["plato-7680"] = {
      entries: [
        {
          name: "cv:1759168",
          display_name: "Juggernaut XL - Ragnarok",
          downloaded: true,
          family: "sdxl",
        } as never,
      ],
      fetchedAt: Date.now(),
      error: null,
    };
    useGenerationStore().jobs.push({
      clientId: 1,
      model: "cv:1759168",
      prompt: "portrait",
      status: "queued",
      step: 0,
      total: 20,
      hostLabel: "plato",
      resultUrl: null,
      previewUrl: null,
      result: null,
    } as never);

    await flushPromises();
    expect(wrapper.text()).toContain("Juggernaut XL - Ragnarok · plato");
    expect(wrapper.text()).not.toContain("cv:1759168");
  });
});
