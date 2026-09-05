import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import StarterCards from "./StarterCards.vue";
import { useDownloadsStore } from "../../stores/downloads";
import { useToastStore } from "../../stores/toasts";
import type { DownloadJob } from "../../lib/api/types";

function job(overrides: Partial<DownloadJob> = {}): DownloadJob {
  return {
    id: "j1",
    model: "flux2-klein:q4",
    status: "active",
    files_done: 1,
    files_total: 4,
    bytes_done: 25,
    bytes_total: 100,
    ...overrides,
  };
}

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("StarterCards (cold start G10)", () => {
  it("renders the first-run guide with three starters, one recommended", () => {
    const wrapper = mount(StarterCards);
    expect(wrapper.text()).toContain("Make your first picture.");
    expect(wrapper.text()).toContain("mold makes pictures on this machine's own graphics card.");
    const cards = wrapper.findAll("[data-test='starter-card']");
    expect(cards).toHaveLength(3);
    expect(wrapper.text()).toContain("flux2-klein:q4");
    const recommended = wrapper.findAll("[data-test='starter-recommended']");
    expect(recommended).toHaveLength(1);
    // The recommended starter is first.
    expect(cards[0]!.text()).toContain("Recommended");
    expect(wrapper.text()).toContain("Browse more");
  });

  it("pulls every starter with its canonical manifest id", async () => {
    const store = useDownloadsStore();
    const createDownload = vi.spyOn(store, "createDownload").mockResolvedValue(undefined);
    const subscribe = vi.spyOn(store, "subscribe").mockResolvedValue(undefined);
    const wrapper = mount(StarterCards);

    for (const button of wrapper.findAll("[data-test='starter-pull']")) {
      await button.trigger("click");
      await flushPromises();
    }

    expect(createDownload.mock.calls).toEqual([
      ["flux2-klein:q4"],
      ["z-image-turbo:q8"],
      ["sdxl-base:fp16"],
    ]);
    expect(subscribe).toHaveBeenCalledTimes(3);
  });

  it("reports a rejected starter pull instead of failing silently", async () => {
    const store = useDownloadsStore();
    vi.spyOn(store, "createDownload").mockRejectedValue(new Error("unknown model"));
    const wrapper = mount(StarterCards);

    await wrapper.findAll("[data-test='starter-pull']")[1]!.trigger("click");
    await flushPromises();

    expect(useToastStore().items.at(-1)).toMatchObject({
      kind: "error",
      message: "Couldn't get z-image-turbo:q8 — unknown model",
    });
  });

  it("shows inline progress on the pulling card instead of a Pull button", () => {
    const store = useDownloadsStore();
    store.activeJobs = [
      job({
        model: "flux2-klein:q4",
        bytes_done: 45_870_258_557,
        bytes_total: 100_000_000_000,
      }),
    ];
    const wrapper = mount(StarterCards);

    const cards = wrapper.findAll("[data-test='starter-card']");
    // The first (matching) card is pulling; the others still offer Pull.
    expect(cards[0]!.find("[data-test='starter-pulling']").exists()).toBe(true);
    expect(cards[0]!.find("[data-test='starter-pull']").exists()).toBe(false);
    expect(cards[0]!.get("[data-test='starter-pulling']").text()).toBe("Getting it… 46%");
    expect(cards[1]!.find("[data-test='starter-pull']").exists()).toBe(true);
  });

  it("emits browse for the escape hatch", async () => {
    const wrapper = mount(StarterCards);
    await wrapper.get("button.text-sapphire").trigger("click");
    expect(wrapper.emitted("browse")).toHaveLength(1);
  });
});
