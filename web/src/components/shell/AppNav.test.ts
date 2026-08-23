import { mount } from "@vue/test-utils";
import { createPinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AppNav from "./AppNav.vue";
import MobileNavSheet from "./MobileNavSheet.vue";
import { takeGenerationHandoff } from "../../composables/useGenerationHandoff";

const routeState = vi.hoisted(() => ({
  name: "library" as string,
  query: {} as Record<string, unknown>,
}));
const pushMock = vi.hoisted(() => vi.fn());
const dlState = vi.hoisted(() => ({
  activeJobs: [] as unknown[],
  queued: [] as unknown[],
}));
const liveState = vi.hoisted(() => ({ rows: [] as Record<string, unknown>[] }));
const routingState = vi.hoisted(() => ({
  hosts: [
    {
      id: "render",
      label: "Render box",
      url: "http://render:7680",
      apiKey: "secret",
      status: "ready",
    },
  ],
}));
const listQueueMock = vi.hoisted(() => vi.fn());

vi.mock("vue-router", () => ({
  useRoute: () => routeState,
  useRouter: () => ({ push: pushMock }),
}));

vi.mock("../../composables/useDownloads", () => ({
  useDownloads: () => ({
    activeJobs: { value: dlState.activeJobs },
    queued: { value: dlState.queued },
    active: { value: null },
  }),
}));
vi.mock("../../composables/useHostRouting", () => ({
  useHostRouting: () => ({
    hosts: { value: routingState.hosts },
    start: vi.fn(),
    stop: vi.fn(),
  }),
}));
vi.mock("../../composables/useLiveActivity", () => ({
  useLiveActivity: () => ({
    rows: { value: liveState.rows },
    start: vi.fn(),
    stop: vi.fn(),
  }),
}));
vi.mock("@studio/api/queuePlan", () => ({ listQueue: listQueueMock }));

const notifState = vi.hoisted(() => ({ fresh: 0, offline: false }));
const statusState = vi.hoisted(() => ({
  error: null as string | null,
  stale: false,
}));
vi.mock("../../composables/useStatusPoll", () => ({
  useStatusPoll: () => ({
    error: { value: statusState.error },
    stale: { value: statusState.stale },
  }),
}));
const markGalleryVisitedMock = vi.hoisted(() => vi.fn());
vi.mock("../../lib/notifications", () => ({
  useNotificationSignals: () => ({
    freshPrintCount: { value: notifState.fresh },
    hasOfflineHost: { value: notifState.offline },
  }),
  markGalleryVisited: markGalleryVisitedMock,
}));

function mountNav() {
  return mount(AppNav, {
    global: {
      // The notifications bell reads its Pinia store.
      plugins: [createPinia()],
      stubs: {
        RouterLink: { template: "<a><slot /></a>" },
      },
    },
  });
}

describe("AppNav", () => {
  beforeEach(() => {
    routeState.name = "library";
    routeState.query = {};
    dlState.activeJobs = [];
    dlState.queued = [];
    notifState.fresh = 0;
    notifState.offline = false;
    statusState.error = null;
    statusState.stale = false;
    liveState.rows = [];
    takeGenerationHandoff();
    listQueueMock.mockReset();
    markGalleryVisitedMock.mockClear();
    pushMock.mockClear();
  });

  it("marks the pill matching the active route", () => {
    routeState.name = "create";
    const wrapper = mountNav();

    expect(
      wrapper.get('[data-test="nav-create"]').attributes("data-active"),
    ).toBe("true");
    expect(
      wrapper.get('[data-test="nav-library"]').attributes("data-active"),
    ).toBe("false");
  });

  it("keeps the Machines pill active on a host detail route", () => {
    routeState.name = "host-detail";
    const wrapper = mountNav();

    expect(
      wrapper.get('[data-test="nav-machines"]').attributes("data-active"),
    ).toBe("true");
  });

  it("navigates when a pill is clicked", async () => {
    const wrapper = mountNav();
    await wrapper.get('[data-test="nav-models"]').trigger("click");
    expect(pushMock).toHaveBeenCalledWith("/models");
  });

  it("shows the downloads badge with the active + queued count", () => {
    dlState.activeJobs = [{ id: "a" }, { id: "b" }];
    dlState.queued = [{ id: "c" }];
    const wrapper = mountNav();

    const badges = wrapper.findAll(".ms-badge");
    expect(badges.length).toBeGreaterThan(0);
    expect(badges[0]?.text()).toBe("3");
  });

  it("hides the downloads badge when nothing is downloading", () => {
    const wrapper = mountNav();
    expect(wrapper.find(".ms-badge").exists()).toBe(false);
  });

  it("routes search submissions to the Library with a q query", async () => {
    const wrapper = mountNav();
    const input = wrapper.get('input[type="search"]');
    await input.setValue("misty forest");
    await wrapper.get('form[role="search"]').trigger("submit");

    expect(pushMock).toHaveBeenCalledWith({
      name: "library",
      query: { q: "misty forest" },
    });
  });

  it("opens the downloads drawer via the shared window event", async () => {
    const wrapper = mountNav();
    const listener = vi.fn();
    window.addEventListener("mold:open-downloads", listener);
    await wrapper.get('[aria-label="Open downloads"]').trigger("click");
    window.removeEventListener("mold:open-downloads", listener);

    expect(listener).toHaveBeenCalled();
  });

  it("opens recovered work from Now developing and restores its settings", async () => {
    liveState.rows = [
      {
        key: "render:generation:foreign",
        id: "foreign",
        kind: "generation",
        phase: "running",
        model: "flux-dev",
        hostId: "render",
        hostLabel: "Render box",
        stale: false,
        hostError: null,
        created_at_unix_ms: 1,
        updated_at_unix_ms: 2,
        can_cancel: false,
      },
    ];
    listQueueMock.mockResolvedValue({
      entries: [
        {
          id: "foreign",
          seed_pinned: true,
          metadata: { model: "flux-dev", prompt: "restore me", seed: 42 },
        },
      ],
      plan: null,
    });
    const wrapper = mountNav();
    await wrapper
      .findAll("[data-test='now-developing-trigger']")[0]!
      .trigger("click");
    await wrapper
      .findAll("[data-test^='live-activity-select-']")[0]!
      .trigger("click");

    expect(listQueueMock).toHaveBeenCalledWith({
      baseUrl: "http://render:7680",
      apiKey: "secret",
    });
    expect(takeGenerationHandoff()).toMatchObject({
      metadata: { prompt: "restore me", seed: 42 },
      seedPinned: true,
    });
    expect(pushMock).toHaveBeenCalledWith("/create");
  });

  it("toggles the mobile nav sheet from the hamburger", async () => {
    const wrapper = mountNav();
    expect(wrapper.findComponent(MobileNavSheet).props("open")).toBe(false);

    await wrapper.get('[data-test="nav-hamburger"]').trigger("click");
    expect(wrapper.findComponent(MobileNavSheet).props("open")).toBe(true);
  });

  it("shows an accent dot on the Library pill for fresh prints", () => {
    notifState.fresh = 2;
    routeState.name = "create";
    const wrapper = mountNav();
    expect(wrapper.find('[data-test="nav-dot-library"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="nav-dot-machines"]').exists()).toBe(false);
  });

  it("shows a stop dot on the Machines pill while a host is offline", () => {
    notifState.offline = true;
    const wrapper = mountNav();
    const dot = wrapper.find('[data-test="nav-dot-machines"]');
    expect(dot.exists()).toBe(true);
    expect(dot.classes()).toContain("seg-dot--stop");
  });

  it("shows reconnecting instead of offline for a transient engine status failure", () => {
    routeState.name = "library";
    statusState.error = "connection refused";
    const wrapper = mountNav();
    expect(wrapper.get("[data-test='global-engine-status']").text()).toContain(
      "Engine reconnecting",
    );
    expect(wrapper.text()).not.toContain("Engine offline");
  });

  it("clears fresh prints when the library route is entered", () => {
    routeState.name = "library";
    mountNav();
    expect(markGalleryVisitedMock).toHaveBeenCalled();
  });
});

describe("mobile nav sheet anchoring", () => {
  it("mounts the sheet in a viewport-fixed host, not inside the 52px bar", async () => {
    // SheetPanel is `position: absolute; inset: 0`; inside the relative header
    // it resolved against the compact bar and rendered as a 52px sliver, which
    // made the whole phone navigation unusable.
    const w = mountNav();
    const host = w.get("[data-test='mobile-nav-host']");
    // Closed: no fixed overlay, so it can't swallow clicks on the page.
    expect(host.classes()).not.toContain("fixed");
    await w.get('[data-test="nav-hamburger"]').trigger("click");
    expect(host.classes()).toContain("fixed");
    expect(host.classes()).toContain("inset-0");
  });
});
