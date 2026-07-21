import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AppNav from "./AppNav.vue";
import MobileNavSheet from "./MobileNavSheet.vue";

const routeState = vi.hoisted(() => ({
  name: "gallery" as string,
  query: {} as Record<string, unknown>,
}));
const pushMock = vi.hoisted(() => vi.fn());
const dlState = vi.hoisted(() => ({
  activeJobs: [] as unknown[],
  queued: [] as unknown[],
}));

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

const notifState = vi.hoisted(() => ({ fresh: 0, offline: false }));
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
      stubs: {
        RouterLink: { template: "<a><slot /></a>" },
      },
    },
  });
}

describe("AppNav", () => {
  beforeEach(() => {
    routeState.name = "gallery";
    routeState.query = {};
    dlState.activeJobs = [];
    dlState.queued = [];
    notifState.fresh = 0;
    notifState.offline = false;
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
      wrapper.get('[data-test="nav-gallery"]').attributes("data-active"),
    ).toBe("false");
  });

  it("keeps the Machines pill active on a host detail route", () => {
    routeState.name = "machine-detail";
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

  it("routes search submissions to the gallery with a q query", async () => {
    const wrapper = mountNav();
    const input = wrapper.get('input[type="search"]');
    await input.setValue("misty forest");
    await wrapper.get('form[role="search"]').trigger("submit");

    expect(pushMock).toHaveBeenCalledWith({
      name: "gallery",
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

  it("toggles the mobile nav sheet from the hamburger", async () => {
    const wrapper = mountNav();
    expect(wrapper.findComponent(MobileNavSheet).props("open")).toBe(false);

    await wrapper.get('[data-test="nav-hamburger"]').trigger("click");
    expect(wrapper.findComponent(MobileNavSheet).props("open")).toBe(true);
  });

  it("shows an accent dot on the Gallery pill for fresh prints", () => {
    notifState.fresh = 2;
    routeState.name = "create";
    const wrapper = mountNav();
    expect(wrapper.find('[data-test="nav-dot-gallery"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="nav-dot-machines"]').exists()).toBe(false);
  });

  it("shows a stop dot on the Machines pill while a host is offline", () => {
    notifState.offline = true;
    const wrapper = mountNav();
    const dot = wrapper.find('[data-test="nav-dot-machines"]');
    expect(dot.exists()).toBe(true);
    expect(dot.classes()).toContain("seg-dot--stop");
  });

  it("clears fresh prints when the gallery route is entered", () => {
    routeState.name = "gallery";
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
