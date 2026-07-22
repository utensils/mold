import { mount } from "@vue/test-utils";
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
});
