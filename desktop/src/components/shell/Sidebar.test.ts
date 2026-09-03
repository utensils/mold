import { describe, expect, it, vi } from "vitest";
import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import Sidebar from "./Sidebar.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useGalleryStore } from "../../stores/gallery";
import { useGenerationStore } from "../../stores/generation";
import { useChainJobsStore } from "../../stores/chainJobs";
import { useLiveActivityStore } from "../../stores/liveActivity";
import { useComposerStore } from "../../stores/composer";
import { isSeparator, useContextMenuStore } from "../../stores/contextMenu";
import { useJobsStore } from "../../stores/jobs";

const stub = { template: "<div />" };
const authedMediaStub = {
  props: ["path", "target", "cacheKey", "alt"],
  template:
    '<img data-test="rail-library-thumbnail" :data-path="path" :data-cache-key="cacheKey" :data-target-base-url="target?.baseUrl" :data-target-authenticated="String(Boolean(target?.apiKey))" :alt="alt" />',
};

function makeRouter(): Router {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      "/create",
      "/queue",
      "/library",
      "/models",
      "/settings",
      "/machines",
      "/machines/:id",
    ].map((path) => ({ path, component: stub })),
  });
}

let router: Router;

async function mountAt(path: string) {
  router = makeRouter();
  router.push(path);
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  return mount(Sidebar, {
    global: {
      plugins: [pinia, router],
      stubs: { AuthedMedia: authedMediaStub },
    },
  });
}

/** The destination button in the sidebar whose label starts with `label`. */
function navButton(wrapper: VueWrapper, label: string) {
  return wrapper.findAll("button").find((button) => button.text().startsWith(label));
}

describe("Sidebar a11y", () => {
  it("labels the primary navigation landmark", async () => {
    const wrapper = await mountAt("/create");
    expect(wrapper.get("nav").attributes("aria-label")).toBe("Primary");
  });

  it("marks the active nav item with aria-current=page", async () => {
    const wrapper = await mountAt("/library");
    // Destinations render as @ui NavItem buttons, not RouterLink anchors.
    expect(navButton(wrapper, "My images")?.attributes("aria-current")).toBe("page");
    expect(navButton(wrapper, "New image")?.attributes("aria-current")).toBeUndefined();
  });

  it("shows the MAKE and SETUP destinations plus Settings", async () => {
    const wrapper = await mountAt("/create");
    for (const label of ["New image", "Queue", "My images", "Styles", "Machines", "Settings"]) {
      expect(wrapper.text()).toContain(label);
    }
    // The old lexicon and the folded destinations are gone from the rail.
    for (const label of [
      "Create",
      "Library",
      "Models",
      "Generate",
      "Gallery",
      "Chains",
      "Catalog",
      "History",
      "RunPod",
    ]) {
      expect(wrapper.text()).not.toContain(label);
    }
  });
});

describe("Sidebar collapse", () => {
  function setCollapsed(collapsed: boolean) {
    const prefs = useAppPrefsStore();
    prefs.settings = { sidebarCollapsed: collapsed } as never;
  }

  it("expands to 270px with visible labels and the plain wordmark by default", async () => {
    const wrapper = await mountAt("/create");
    expect(wrapper.get("nav").attributes("style")).toContain("270px");
    expect(wrapper.text()).toContain("New image");
    // Wordmark: the mark plus plain mono "mold studio", no gradient type.
    expect(wrapper.find('img[alt="mold"]').exists()).toBe(true);
    expect(wrapper.text()).toContain("studio");
  });

  it("collapses to a 62px icon rail with labels, wordmark, and queue hidden", async () => {
    const wrapper = await mountAt("/create");
    setCollapsed(true);
    await flushPromises();
    expect(wrapper.get("nav").attributes("style")).toContain("62px");
    // Icon-only: nav labels, the wordmark, and the queue are gone.
    expect(wrapper.text()).not.toContain("New image");
    expect(wrapper.text()).not.toContain("studio");
    expect(wrapper.find("[data-test='queue-rail']").exists()).toBe(false);
    // Destinations are still present as accessible icon buttons.
    const buttons = wrapper.findAll("button");
    expect(buttons.some((b) => b.attributes("aria-label") === "New image")).toBe(true);
  });
});

describe("Sidebar queue", () => {
  function setLocalAuthority(baseUrl = "http://127.0.0.1:49152", instanceId = "local-instance") {
    const connection = useConnectionStore();
    connection.info = { mode: "local", baseUrl, apiKey: "secret" };
    connection.status = "ready";
    useHostsStore().telemetry.local = {
      queueDepth: 1,
      queueCapacity: 1,
      version: null,
      instanceId,
    };
  }

  it("keeps recovered and local work newest-first across phase changes", async () => {
    const wrapper = await mountAt("/create");
    useGenerationStore().jobs = [
      {
        clientId: 1,
        id: "older-local",
        model: "flux-dev:q8",
        prompt: "older queued print",
        status: "queued",
        submittedAtUnixMs: 1_000,
      } as never,
    ];
    useLiveActivityStore().hosts = {
      render: {
        hostId: "render",
        hostLabel: "Render box",
        target: { baseUrl: "http://render:7680", apiKey: null },
        routeUrl: "http://render:7680",
        instanceId: "render-instance",
        observedAtUnixMs: 3_000,
        stale: false,
        error: null,
        unavailableKinds: [],
        items: [
          {
            id: "newer-running",
            kind: "generation",
            phase: "running",
            model: "flux-schnell",
            created_at_unix_ms: 2_000,
            updated_at_unix_ms: 3_000,
            can_cancel: false,
          },
        ],
      },
    };
    await flushPromises();

    const text = wrapper.get("[data-test='queue-rail']").text();
    expect(text.indexOf("flux-schnell")).toBeLessThan(text.indexOf("older queued print"));
  });

  it("shows cancellation progress, then offers to remove the stopped row", async () => {
    const wrapper = await mountAt("/create");
    const generation = useGenerationStore();
    generation.jobs = [
      {
        clientId: 7,
        model: "ltx-2.5-22b-distilled:bf16-conv",
        prompt: "cancel me",
        status: "queued",
        cancelling: true,
      } as never,
    ];
    await flushPromises();

    const row = wrapper.get("[data-test='queue-row-print']");
    expect(row.text()).toContain("Stopping…");
    await row.trigger("contextmenu");
    expect(useContextMenuStore().entries[0]).toMatchObject({
      label: "Stop",
      disabled: true,
    });

    generation.jobs[0]!.status = "error";
    generation.jobs[0]!.error = "Cancelled";
    generation.jobs[0]!.cancelling = false;
    await flushPromises();
    expect(row.text()).toContain("Stopped");

    await row.trigger("contextmenu");
    const menu = useContextMenuStore();
    expect(menu.entries[0]).toMatchObject({ label: "Remove from queue" });
    menu.activate(menu.entries[0]!);
    await flushPromises();

    expect(generation.jobs).toEqual([]);
    expect(wrapper.find("[data-test='queue-row-print']").exists()).toBe(false);
  });

  it("offers the print row's own verbs: reuse the words, show it, clear finished", async () => {
    const wrapper = await mountAt("/create");
    useGenerationStore().jobs = [
      {
        clientId: 9,
        model: "flux-dev:q8",
        prompt: "a finished print",
        status: "complete",
        result: { filename: "done.png", image: "" },
      } as never,
    ];
    await flushPromises();

    await wrapper.get("[data-test='queue-row-print']").trigger("contextmenu");
    const labels = useContextMenuStore().entries.flatMap((entry) =>
      isSeparator(entry) ? [] : [entry.label],
    );
    expect(labels).toEqual([
      "Remove from queue",
      "Use these words",
      "Show in My images",
      "Clear finished",
    ]);
  });

  it("opens a row's actions from the ⋯ button, not only the right mouse button", async () => {
    const wrapper = await mountAt("/create");
    useGenerationStore().jobs = [
      {
        clientId: 10,
        model: "flux-dev:q8",
        prompt: "menu me",
        status: "queued",
      } as never,
    ];
    await flushPromises();

    await wrapper.get("[data-test='queue-row-menu']").trigger("click");
    const menu = useContextMenuStore();
    expect(menu.visible).toBe(true);
    expect(menu.entries[0]).toMatchObject({ label: "Stop", danger: true });
  });

  // A mesh print's saved file is binary glTF: the result-URL arm cannot draw
  // it, so the rendered poster the complete event carries is the rail's only
  // picture of it until the host has a gallery thumbnail to serve.
  it("draws a finished mesh print from its poster, never its glTF bytes", async () => {
    const wrapper = await mountAt("/create");
    useGenerationStore().jobs = [
      {
        clientId: 11,
        model: "hunyuan3d-mini-turbo:fp16",
        prompt: "an armchair",
        status: "complete",
        settledAtMs: Date.now(),
        resultUrl: "blob:mesh",
        result: {
          image: "R0xURg==",
          format: "glb",
          mesh_vertices: 24_576,
          mesh_faces: 49_152,
          mesh_poster: "UE9TVEVS",
        },
      } as never,
    ];
    await flushPromises();

    const row = wrapper.get("[data-test='queue-row-print']");
    const picture = row.get("img");
    expect(picture.attributes("src")).toBe("data:image/png;base64,UE9TVEVS");
    expect(row.find("img[src='blob:mesh']").exists()).toBe(false);
  });

  it("shows an unknown outcome in the muted ink, never as a failure", async () => {
    const wrapper = await mountAt("/create");
    useGenerationStore().jobs = [
      {
        clientId: 8,
        model: "flux-dev:q8",
        prompt: "replaced mid-print",
        status: "error",
        outcomeUnknown: true,
        stage: "Outcome unknown",
        error: "hal9000 was replaced by a new server instance.",
        settledAtMs: Date.now(),
      } as never,
    ];
    await flushPromises();

    const row = wrapper.get("[data-test='queue-row-print']");
    expect(row.text()).toContain("Outcome unknown");
    expect(row.text()).not.toContain("Failed");
    expect(row.find(".text-error").exists()).toBe(false);
    expect(row.find(".text-fg-dim").exists()).toBe(true);
  });

  it("stops another client's running job from its context menu", async () => {
    const wrapper = await mountAt("/create");
    setLocalAuthority();
    const liveActivity = useLiveActivityStore();
    vi.spyOn(liveActivity, "refresh").mockResolvedValue(undefined);
    const cancel = vi.spyOn(useJobsStore(), "cancelJob").mockResolvedValue(undefined);
    liveActivity.hosts = {
      local: {
        hostId: "local",
        hostLabel: "This Mac",
        target: { baseUrl: "http://127.0.0.1:49152", apiKey: "secret" },
        routeUrl: "http://127.0.0.1:49152",
        instanceId: "local-instance",
        observedAtUnixMs: 2,
        stale: false,
        error: null,
        unavailableKinds: [],
        items: [
          {
            id: "foreign-running",
            kind: "generation",
            phase: "running",
            model: "ltx-2.3-22b-distilled:fp8",
            created_at_unix_ms: 1,
            updated_at_unix_ms: 2,
            can_cancel: true,
          },
        ],
      },
    };
    await flushPromises();

    await wrapper.get("[data-test='queue-active']").trigger("contextmenu");
    const menu = useContextMenuStore();
    expect(menu.visible).toBe(true);
    expect(menu.entries).toMatchObject([{ label: "Stop", danger: true, disabled: false }]);

    menu.activate(menu.entries[0]!);
    await flushPromises();
    expect(cancel).toHaveBeenCalledWith("local", "foreign-running");
  });

  it("stops an auto-chain generation through its durable chain authority", async () => {
    const wrapper = await mountAt("/create");
    setLocalAuthority();
    const liveActivity = useLiveActivityStore();
    vi.spyOn(liveActivity, "refresh").mockResolvedValue(undefined);
    const queueCancel = vi.spyOn(useJobsStore(), "cancelJob").mockResolvedValue(undefined);
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 202 });
    vi.stubGlobal("fetch", fetchMock);
    liveActivity.hosts = {
      local: {
        hostId: "local",
        hostLabel: "This Mac",
        target: { baseUrl: "http://127.0.0.1:49152", apiKey: "secret" },
        routeUrl: "http://127.0.0.1:49152",
        instanceId: "local-instance",
        observedAtUnixMs: 2,
        stale: false,
        error: null,
        unavailableKinds: [],
        items: [
          {
            id: "foreign-auto-chain",
            kind: "generation",
            execution: "chain",
            phase: "running",
            model: "ltx-2.3-22b-distilled:fp8",
            created_at_unix_ms: 1,
            updated_at_unix_ms: 2,
            can_cancel: true,
          },
        ],
      },
    };
    await flushPromises();

    await wrapper.get("[data-test='queue-active']").trigger("contextmenu");
    const menu = useContextMenuStore();
    menu.activate(menu.entries[0]!);
    await flushPromises();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:49152/api/chain-jobs/foreign-auto-chain/cancel",
      expect.objectContaining({ method: "POST" }),
    );
    expect(queueCancel).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it("does not send duplicate stop requests while one is in flight", async () => {
    const wrapper = await mountAt("/create");
    setLocalAuthority();
    const liveActivity = useLiveActivityStore();
    vi.spyOn(liveActivity, "refresh").mockResolvedValue(undefined);
    liveActivity.hosts = {
      local: {
        hostId: "local",
        hostLabel: "This Mac",
        target: { baseUrl: "http://127.0.0.1:49152", apiKey: "secret" },
        routeUrl: "http://127.0.0.1:49152",
        instanceId: "local-instance",
        observedAtUnixMs: 2,
        stale: false,
        error: null,
        unavailableKinds: [],
        items: [
          {
            id: "foreign-running",
            kind: "generation",
            phase: "running",
            model: "flux-dev",
            created_at_unix_ms: 1,
            updated_at_unix_ms: 2,
            can_cancel: true,
          },
        ],
      },
    };
    let finishCancel!: () => void;
    const cancel = vi
      .spyOn(useJobsStore(), "cancelJob")
      .mockImplementation(() => new Promise<void>((resolve) => (finishCancel = resolve)));
    await flushPromises();

    const row = wrapper.get("[data-test='queue-active']");
    await row.trigger("contextmenu");
    const menu = useContextMenuStore();
    menu.activate(menu.entries[0]!);
    await row.trigger("contextmenu");
    expect(menu.entries).toMatchObject([{ label: "Stop", disabled: true }]);
    menu.activate(menu.entries[0]!);
    expect(cancel).toHaveBeenCalledTimes(1);

    finishCancel();
    await flushPromises();
  });

  it("refuses to stop if the host authority changed after the menu opened", async () => {
    const wrapper = await mountAt("/create");
    setLocalAuthority();
    const liveActivity = useLiveActivityStore();
    vi.spyOn(liveActivity, "refresh").mockResolvedValue(undefined);
    liveActivity.hosts = {
      local: {
        hostId: "local",
        hostLabel: "This Mac",
        target: { baseUrl: "http://127.0.0.1:49152", apiKey: "secret" },
        routeUrl: "http://127.0.0.1:49152",
        instanceId: "local-instance",
        observedAtUnixMs: 2,
        stale: false,
        error: null,
        unavailableKinds: [],
        items: [
          {
            id: "foreign-running",
            kind: "generation",
            phase: "running",
            model: "flux-dev",
            created_at_unix_ms: 1,
            updated_at_unix_ms: 2,
            can_cancel: true,
          },
        ],
      },
    };
    const cancel = vi.spyOn(useJobsStore(), "cancelJob").mockResolvedValue(undefined);
    await flushPromises();

    await wrapper.get("[data-test='queue-active']").trigger("contextmenu");
    useHostsStore().telemetry.local!.instanceId = "replacement-instance";
    const menu = useContextMenuStore();
    menu.activate(menu.entries[0]!);
    await flushPromises();

    expect(cancel).not.toHaveBeenCalled();
  });

  it("shows recovered work and reloads its submitted settings", async () => {
    const wrapper = await mountAt("/create");
    useHostsStore().extras.push({
      id: "render",
      label: "Render box",
      url: "http://render:7680",
      apiKey: "secret",
      status: "ready",
      error: null,
      instanceId: "render-instance",
    });
    useLiveActivityStore().hosts = {
      render: {
        hostId: "render",
        hostLabel: "Render box",
        target: { baseUrl: "http://render:7680", apiKey: null },
        routeUrl: "http://render:7680",
        instanceId: "render-instance",
        observedAtUnixMs: 2,
        stale: false,
        error: null,
        unavailableKinds: [],
        items: [
          {
            id: "foreign",
            kind: "generation",
            phase: "running",
            model: "flux-dev",
            created_at_unix_ms: 1,
            updated_at_unix_ms: 2,
            can_cancel: false,
          },
        ],
      },
    };
    const queueListing = {
      entries: [
        {
          id: "foreign",
          model: "flux-dev",
          state: "running",
          started_at_unix_ms: 1,
          position: 0,
          seed_pinned: true,
          metadata: {
            model: "flux-dev",
            prompt: "restore me",
            width: 1024,
            height: 1024,
            steps: 20,
            guidance: 3.5,
            seed: 42,
          },
        },
      ],
      plan: null,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        const body = String(url).includes("/api/queue") ? queueListing : { queue_capacity: 8 };
        return Promise.resolve(
          new Response(JSON.stringify(body), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
        );
      }),
    );
    await flushPromises();

    const active = wrapper.get("[data-test='queue-active']");
    expect(active.text()).toContain("flux-dev");
    await active.trigger("click");
    await flushPromises();

    expect(useComposerStore().prefill).toMatchObject({
      metadata: { prompt: "restore me", seed: 42 },
    });
    vi.unstubAllGlobals();
  });

  it("uses the remaining sidebar height before scrolling the queue", async () => {
    const wrapper = await mountAt("/create");
    useGenerationStore().jobs = [
      {
        clientId: 1,
        model: "flux-dev:q8",
        prompt: "queued print",
        status: "queued",
      } as never,
    ];
    await flushPromises();

    expect(wrapper.get("nav").classes()).toEqual(
      expect.arrayContaining(["min-h-0", "overflow-hidden"]),
    );
    expect(wrapper.get("[data-test='queue-rail']").classes()).toEqual(
      expect.arrayContaining(["min-h-0", "flex-1"]),
    );
    const rows = wrapper.get("[data-test='queue-rows']");
    expect(rows.classes()).toEqual(
      expect.arrayContaining(["min-h-0", "flex-1", "overflow-y-auto"]),
    );
    expect(rows.classes()).not.toContain("max-h-44");
  });

  it("uses the available sidebar space for a longer finished-print history", async () => {
    const wrapper = await mountAt("/create");
    useGenerationStore().jobs = Array.from({ length: 12 }, (_, index) => ({
      clientId: index + 1,
      model: "flux-dev:q8",
      prompt: `finished print ${index + 1}`,
      status: "complete",
    })) as never;
    await flushPromises();

    expect(wrapper.findAll("[data-test='queue-row-print']")).toHaveLength(12);
    expect(wrapper.get("[data-test='queue-rows']").classes()).toContain("overflow-y-auto");
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

    await wrapper.get("[data-test='queue-row-print']").trigger("click");
    expect(refresh).toHaveBeenCalledWith(13);
  });

  it("renews a finished print's result URL when it is re-selected, not only when it is missing", async () => {
    const wrapper = await mountAt("/create");
    const generation = useGenerationStore();
    const refresh = vi.spyOn(generation, "refreshRemoteResultUrl").mockResolvedValue();
    generation.jobs = [
      {
        clientId: 15,
        model: "hunyuan3d:fp16",
        prompt: "",
        status: "complete",
        // A media ticket minted an hour ago: present, but no longer valid.
        resultUrl: "http://hal9000:7680/api/gallery/image/mesh.glb?media_token=old&expires=1",
        resultUrlExpiresAt: 1_000,
        result: { filename: "mesh.glb", image: "" },
        request: { prompt: "", model: "hunyuan3d:fp16" },
      } as never,
    ];
    await flushPromises();

    await wrapper.get("[data-test='queue-row-print']").trigger("click");
    // The store decides whether the URL is still fresh; the rail always asks.
    expect(refresh).toHaveBeenCalledWith(15);
  });

  it("shows the authenticated My images thumbnail for a completed video", async () => {
    const wrapper = await mountAt("/create");
    const generation = useGenerationStore();
    vi.spyOn(generation, "targetForJob").mockReturnValue({
      baseUrl: "http://hal9000:7680",
      apiKey: "secret",
    });
    generation.jobs = [
      {
        clientId: 14,
        model: "ltx-2.3-22b-distilled:fp8",
        prompt: "a finished clip",
        status: "complete",
        hostId: "hal9000-7680",
        resultUrl: "https://example.test/result.mp4",
        previewUrl: null,
        result: { filename: "finished clip.mp4", video_frames: 97 },
      } as never,
    ];
    await flushPromises();

    const thumbnail = wrapper.get("[data-test='rail-library-thumbnail']");
    expect(thumbnail.attributes("data-path")).toBe("/api/gallery/thumbnail/finished%20clip.mp4");
    expect(thumbnail.attributes("data-cache-key")).toBe("hal9000-7680");
    expect(thumbnail.attributes("data-target-base-url")).toBe("http://hal9000:7680");
    expect(thumbnail.attributes("data-target-authenticated")).toBe("true");
    expect(thumbnail.attributes("alt")).toBe("a finished clip");
  });

  // G14 hole: the rail only ever read `generation.jobs`, so a running sequence
  // rendered on the canvas while the sidebar showed nothing.
  it("shows a running sequence with its scene counter", async () => {
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

    const rail = wrapper.get("[data-test='queue-rail']");
    expect(rail.text()).toContain("Making scene 3 of 5");
    expect(rail.text()).toContain("5-scene clip");
    expect(wrapper.get("[data-test='queue-count']").text()).toBe("1");
  });

  // Settled sequences have two homes already (the print in My images, the job
  // in History) — rebuilding the pile one route away is the thing we removed.
  it("keeps settled sequences out of the queue", async () => {
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

    expect(wrapper.find("[data-test='queue-active']").exists()).toBe(false);
    expect(wrapper.find("[data-test='queue-row-sequence']").exists()).toBe(false);
    expect(wrapper.find("[data-test='queue-count']").exists()).toBe(false);
  });
});

describe("Sidebar queue controls", () => {
  it("opens the full queue view", async () => {
    const wrapper = await mountAt("/create");
    await wrapper.get("[data-test='queue-open']").trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/queue");
  });

  it("disables Stop everything while nothing is in flight", async () => {
    const wrapper = await mountAt("/create");
    expect(wrapper.get("[data-test='queue-stop-all']").attributes("disabled")).toBeDefined();
  });

  it("offers pause only when the display host reports a pausable queue", async () => {
    const wrapper = await mountAt("/create");
    const connection = useConnectionStore();
    connection.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    connection.status = "ready";
    await flushPromises();
    expect(wrapper.find("[data-test='queue-pause']").exists()).toBe(false);

    const jobs = useJobsStore();
    jobs.queues.local = { entries: [], caps: { canPause: true }, paused: false } as never;
    const pause = vi.spyOn(jobs, "pause").mockResolvedValue(undefined as never);
    await flushPromises();

    await wrapper.get("[data-test='queue-pause']").trigger("click");
    await flushPromises();
    expect(pause).toHaveBeenCalledWith("local");
  });
});

describe("Sidebar destination badges (G11)", () => {
  function readyLocal() {
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
  }

  it("turns the Machines dot red when a connected host is offline", async () => {
    const wrapper = await mountAt("/create");
    readyLocal();
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "error",
      error: "down",
      instanceId: null,
    });
    await flushPromises();
    expect(wrapper.get("[data-test='machines-dot']").classes()).toContain("bg-error");
  });

  it("keeps the Machines dot green while every host is reachable", async () => {
    const wrapper = await mountAt("/create");
    readyLocal();
    await flushPromises();
    expect(wrapper.get("[data-test='machines-dot']").classes()).toContain("bg-success");
    expect(wrapper.find("[data-test='machines-error-dot']").exists()).toBe(false);
  });

  it("keeps the offline signal visible on the collapsed icon rail", async () => {
    const wrapper = await mountAt("/create");
    readyLocal();
    useAppPrefsStore().settings = { sidebarCollapsed: true } as never;
    useHostsStore().extras.push({
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

  it("badges My images with the count of prints made since the last visit", async () => {
    const wrapper = await mountAt("/create");
    readyLocal();
    const gallery = useGalleryStore();
    const image = (filename: string, timestamp: number) =>
      ({ filename, timestamp, metadata: { prompt: "p" } }) as never;
    gallery.buckets.local = {
      items: [image("a.png", 1)],
      loading: false,
      error: null,
      loaded: true,
    };
    // Before the first visit nothing is new: the trailing readout is the total.
    await flushPromises();
    expect(navButton(wrapper, "My images")?.find(".ms-nav__badge").exists()).toBe(false);

    gallery.markLibrarySeen();
    gallery.buckets.local.items = [image("b.png", 3), image("a.png", 1)];
    await flushPromises();
    const badge = navButton(wrapper, "My images")!.find(".ms-nav__badge");
    expect(badge.exists()).toBe(true);
    expect(badge.text()).toBe("1");
  });

  it("badges Queue with everything in flight", async () => {
    const wrapper = await mountAt("/create");
    useGenerationStore().jobs = [
      { clientId: 1, model: "flux-dev:q8", prompt: "one", status: "queued" } as never,
      { clientId: 2, model: "flux-dev:q8", prompt: "two", status: "denoising" } as never,
    ];
    await flushPromises();
    expect(navButton(wrapper, "Queue")!.find(".ms-nav__badge").text()).toBe("2");
  });
});

describe("Sidebar row titles", () => {
  it("titles a print by its words, never by a raw catalog model id", async () => {
    const wrapper = await mountAt("/create");
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

    const rail = wrapper.get("[data-test='queue-rail']");
    expect(rail.text()).toContain("portrait");
    expect(rail.text()).not.toContain("cv:1759168");
  });
});
