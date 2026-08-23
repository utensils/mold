import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import ColdStartGuide from "./ColdStartGuide.vue";

const enqueue = vi.hoisted(() => vi.fn(async () => undefined));
const toastMock = vi.hoisted(() => vi.fn());
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
vi.mock("../../lib/toasts", () => ({ toast: toastMock }));

function mountGuide() {
  return mount(ColdStartGuide);
}

beforeEach(() => {
  vi.clearAllMocks();
  dlState.active = [];
  dlState.queued = [];
});

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
    expect(
      wrapper.find('[data-test="starter-z-image-turbo:q8"]').exists(),
    ).toBe(true);
    expect(wrapper.find('[data-test="starter-sdxl-base:fp16"]').exists()).toBe(
      true,
    );
    expect(wrapper.text()).toContain("recommended");
  });

  it("enqueues every starter with its canonical manifest id", async () => {
    dlState.active = [];
    dlState.queued = [];
    const wrapper = mountGuide();
    for (const model of [
      "flux2-klein:q4",
      "z-image-turbo:q8",
      "sdxl-base:fp16",
    ]) {
      await wrapper.get(`[data-test="starter-pull-${model}"]`).trigger("click");
    }
    expect(enqueue.mock.calls).toEqual([
      ["flux2-klein:q4"],
      ["z-image-turbo:q8"],
      ["sdxl-base:fp16"],
    ]);
  });

  it("reports a rejected starter pull instead of failing silently", async () => {
    enqueue.mockRejectedValueOnce(new Error("unknown model"));
    const wrapper = mountGuide();

    await wrapper
      .get('[data-test="starter-pull-z-image-turbo:q8"]')
      .trigger("click");

    expect(toastMock).toHaveBeenCalledWith(
      "error",
      "couldn't pull z-image-turbo:q8 — unknown model",
    );
  });

  it("shows inline progress in place of Pull while a starter is downloading", () => {
    dlState.active = [
      {
        id: "d1",
        model: "flux2-klein:q4",
        status: "active",
        bytes_done: 45_870_258_557,
        bytes_total: 100_000_000_000,
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
    expect(wrapper.text()).toContain("pulling 46%");
  });

  it("marks a queued starter as queued", () => {
    dlState.active = [];
    dlState.queued = [
      { id: "q1", model: "sdxl-base:fp16", status: "queued", bytes_total: 0 },
    ];
    const wrapper = mountGuide();
    expect(wrapper.text()).toContain("queued");
  });
});
