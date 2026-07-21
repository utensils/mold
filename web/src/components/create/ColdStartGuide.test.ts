import { describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import ColdStartGuide from "./ColdStartGuide.vue";

const enqueue = vi.hoisted(() => vi.fn(async () => undefined));
const dlState = vi.hoisted(() => ({
  active: [] as unknown[],
  queued: [] as unknown[],
}));
vi.mock("../../composables/useDownloads", () => ({
  useDownloads: () => ({
    activeJobs: { value: dlState.active },
    queued: { value: dlState.queued },
    enqueue,
  }),
}));

function mountGuide() {
  return mount(ColdStartGuide);
}

describe("ColdStartGuide", () => {
  it("renders the guide with three starter models, recommended first", () => {
    dlState.active = [];
    dlState.queued = [];
    const wrapper = mountGuide();
    expect(wrapper.text()).toContain("pull a model to start");
    expect(wrapper.text()).toContain("this machine's GPU");
    expect(wrapper.find('[data-test="starter-flux2-klein:q4"]').exists()).toBe(
      true,
    );
    expect(wrapper.find('[data-test="starter-z-image:turbo"]').exists()).toBe(
      true,
    );
    expect(wrapper.find('[data-test="starter-sdxl:base"]').exists()).toBe(true);
    expect(wrapper.text()).toContain("recommended");
  });

  it("enqueues a download when a starter's Pull is clicked", async () => {
    dlState.active = [];
    dlState.queued = [];
    const wrapper = mountGuide();
    await wrapper
      .get('[data-test="starter-pull-flux2-klein:q4"]')
      .trigger("click");
    expect(enqueue).toHaveBeenCalledWith("flux2-klein:q4");
  });

  it("shows inline progress in place of Pull while a starter is downloading", () => {
    dlState.active = [
      {
        id: "d1",
        model: "flux2-klein:q4",
        status: "active",
        bytes_done: 3_000,
        bytes_total: 10_000,
      },
    ];
    dlState.queued = [];
    const wrapper = mountGuide();
    expect(
      wrapper.find('[data-test="starter-progress-flux2-klein:q4"]').exists(),
    ).toBe(true);
    expect(
      wrapper.find('[data-test="starter-pull-flux2-klein:q4"]').exists(),
    ).toBe(false);
    expect(wrapper.text()).toContain("pulling 30%");
  });

  it("marks a queued starter as queued", () => {
    dlState.active = [];
    dlState.queued = [
      { id: "q1", model: "sdxl:base", status: "queued", bytes_total: 0 },
    ];
    const wrapper = mountGuide();
    expect(wrapper.text()).toContain("queued");
  });
});
