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
});
