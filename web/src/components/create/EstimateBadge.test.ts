import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import EstimateBadge from "./EstimateBadge.vue";

const estimateMock = vi.hoisted(() => vi.fn());
vi.mock("../../api", () => ({ fetchGenerationEstimate: estimateMock }));

describe("EstimateBadge", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    estimateMock.mockReset();
    estimateMock.mockResolvedValue({
      model: "flux-dev:q4",
      peak_memory_bytes: 4_000_000_000,
      available_memory_bytes: 8_000_000_000,
      activation_memory_bytes: 1,
      load_strategy: "resident",
      fits_available_memory: true,
      capacity_peak_memory_bytes: 4_000_000_000,
      device_capacity_bytes: 8_000_000_000,
      fits_device_capacity: true,
    });
  });

  afterEach(() => vi.useRealTimers());

  it("estimates the pending request against its routed host", async () => {
    const target = { baseUrl: "http://studio:7680", apiKey: "secret" };
    const wrapper = mount(EstimateBadge, {
      props: {
        request: {
          prompt: "a cat",
          model: "flux-dev:q4",
          width: 1024,
          height: 1024,
          steps: 20,
          guidance: 3.5,
          output_format: "png",
        },
        target,
      },
    });
    await vi.advanceTimersByTimeAsync(600);
    expect(estimateMock).toHaveBeenCalledWith(
      expect.objectContaining({ model: "flux-dev:q4" }),
      target,
    );
    expect(wrapper.get("[data-test='vram-estimate']").text()).toContain(
      "VRAM · fits — est. 4.0 GB of 8.0 GB",
    );
  });

  it("surfaces an unavailable estimate instead of implying that it fits", async () => {
    estimateMock.mockRejectedValue(new Error("offline"));
    const wrapper = mount(EstimateBadge, {
      props: {
        request: {
          prompt: "a cat",
          model: "flux-dev:q4",
          width: 1024,
          height: 1024,
          steps: 20,
          guidance: 3.5,
          output_format: "png",
        },
      },
    });
    await vi.advanceTimersByTimeAsync(600);
    expect(wrapper.text()).toContain("VRAM · estimate unavailable");
  });

  it("ignores an in-flight estimate as soon as the request changes", async () => {
    let resolveFirst!: (value: unknown) => void;
    estimateMock
      .mockImplementationOnce(
        () => new Promise((resolve) => (resolveFirst = resolve)),
      )
      .mockResolvedValueOnce({
        model: "sdxl:fp16",
        peak_memory_bytes: 6_000_000_000,
        available_memory_bytes: 8_000_000_000,
        activation_memory_bytes: 1,
        load_strategy: "resident",
        fits_available_memory: true,
        capacity_peak_memory_bytes: 6_000_000_000,
        device_capacity_bytes: 8_000_000_000,
        fits_device_capacity: true,
      });
    const request = {
      prompt: "a cat",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      output_format: "png" as const,
    };
    const wrapper = mount(EstimateBadge, { props: { request } });
    await vi.advanceTimersByTimeAsync(600);
    await wrapper.setProps({ request: { ...request, model: "sdxl:fp16" } });
    resolveFirst({
      model: "flux-dev:q4",
      peak_memory_bytes: 9_000_000_000,
      available_memory_bytes: 8_000_000_000,
      activation_memory_bytes: 1,
      load_strategy: "resident",
      fits_available_memory: false,
      capacity_peak_memory_bytes: 4_000_000_000,
      device_capacity_bytes: 8_000_000_000,
      fits_device_capacity: true,
    });
    await flushPromises();
    expect(wrapper.text()).not.toContain("won't fit");
    await vi.advanceTimersByTimeAsync(600);
    expect(wrapper.text()).toContain("est. 6.0 GB of 8.0 GB");
  });

  it("keeps the row mounted across refreshes so the layout never shifts", async () => {
    const request = {
      prompt: "a cat",
      model: "flux-dev:q4",
      width: 1024,
      height: 1024,
      steps: 20,
      guidance: 3.5,
      output_format: "png" as const,
    };
    const wrapper = mount(EstimateBadge, { props: { request } });
    // The line reserves its height immediately, before the debounce fires.
    expect(wrapper.get("[data-test='vram-estimate']").text()).toContain(
      "estimating",
    );
    await vi.advanceTimersByTimeAsync(600);
    expect(wrapper.get("[data-test='vram-estimate']").text()).toContain(
      "est. 4.0 GB of 8.0 GB",
    );

    // A request change keeps the previous estimate visible (dimmed) instead
    // of unmounting the row and letting the page grow/shrink.
    await wrapper.setProps({ request: { ...request, steps: 24 } });
    const row = wrapper.get("[data-test='vram-estimate']");
    expect(row.text()).toContain("est. 4.0 GB of 8.0 GB");
    expect(row.attributes("data-refreshing")).toBe("true");
    await vi.advanceTimersByTimeAsync(600);
    expect(
      wrapper.get("[data-test='vram-estimate']").attributes("data-refreshing"),
    ).toBeUndefined();

    // Clearing the model is the one thing that hides the row.
    await wrapper.setProps({ request: { ...request, model: "" } });
    expect(wrapper.find("[data-test='vram-estimate']").exists()).toBe(false);
  });
});
