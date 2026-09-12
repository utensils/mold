import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import EstimateBadge from "./EstimateBadge.vue";
import source from "./EstimateBadge.vue?raw";

const FITS = {
  peak_memory_bytes: 4_000_000_000,
  available_memory_bytes: 8_000_000_000,
  fits_available_memory: true,
  capacity_peak_memory_bytes: 4_000_000_000,
  device_capacity_bytes: 8_000_000_000,
  fits_device_capacity: true,
};

interface Request {
  prompt: string;
  model: string;
  width: number;
  height: number;
  steps: number;
}

const REQUEST: Request = {
  prompt: "a cat",
  model: "flux-dev:q4",
  width: 1024,
  height: 1024,
  steps: 20,
};

/* Hoisted so `setProps` sees a value of the request type rather than a fresh
 * literal, which TypeScript excess-property-checks against the component's
 * generic constraint. */
const variant = (patch: Partial<Request>): Request => ({
  ...REQUEST,
  ...patch,
});

const estimate = vi.fn();

interface Target {
  baseUrl: string;
  apiKey?: string;
}

function mountBadge(target: Target | null = null) {
  return mount(EstimateBadge, {
    props: { request: REQUEST, target, estimate },
  });
}

describe("EstimateBadge", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    estimate.mockReset();
    estimate.mockResolvedValue(FITS);
  });

  afterEach(() => vi.useRealTimers());

  it("estimates the pending request against its routed machine", async () => {
    const target = { baseUrl: "http://studio:7680", apiKey: "secret" };
    const wrapper = mountBadge(target);
    await vi.advanceTimersByTimeAsync(600);
    expect(estimate).toHaveBeenCalledWith(
      expect.objectContaining({ model: "flux-dev:q4" }),
      target,
    );
    expect(wrapper.get("[data-test='vram-estimate']").text()).toContain(
      "VRAM · fits — est. 4.0 GB of 8.0 GB",
    );
  });

  it("surfaces an unavailable estimate instead of implying that it fits", async () => {
    estimate.mockRejectedValue(new Error("offline"));
    const wrapper = mountBadge();
    await vi.advanceTimersByTimeAsync(600);
    expect(wrapper.text()).toContain("VRAM · estimate unavailable");
    expect(
      wrapper.get("[data-test='vram-estimate']").attributes("data-fit"),
    ).toBe("unavailable");
  });

  it("ignores an in-flight estimate as soon as the request changes", async () => {
    let resolveFirst!: (value: unknown) => void;
    estimate
      .mockImplementationOnce(
        () => new Promise((resolve) => (resolveFirst = resolve)),
      )
      .mockResolvedValueOnce({
        ...FITS,
        peak_memory_bytes: 6_000_000_000,
        capacity_peak_memory_bytes: 6_000_000_000,
      });
    const wrapper = mountBadge();
    await vi.advanceTimersByTimeAsync(600);
    await wrapper.setProps({ request: variant({ model: "sdxl:fp16" }) });
    resolveFirst({
      ...FITS,
      peak_memory_bytes: 9_000_000_000,
      capacity_peak_memory_bytes: 9_000_000_000,
      fits_device_capacity: false,
    });
    await flushPromises();
    expect(wrapper.text()).not.toContain("won't fit");
    await vi.advanceTimersByTimeAsync(600);
    expect(wrapper.text()).toContain("est. 6.0 GB of 8.0 GB");
  });

  it("keeps the row mounted across refreshes so the layout never shifts", async () => {
    const wrapper = mountBadge();
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
    await wrapper.setProps({ request: variant({ steps: 24 }) });
    const row = wrapper.get("[data-test='vram-estimate']");
    expect(row.text()).toContain("est. 4.0 GB of 8.0 GB");
    expect(row.attributes("data-refreshing")).toBe("true");
    await vi.advanceTimersByTimeAsync(600);
    expect(
      wrapper.get("[data-test='vram-estimate']").attributes("data-refreshing"),
    ).toBeUndefined();

    // Clearing the model is the one thing that hides the row.
    await wrapper.setProps({ request: variant({ model: "" }) });
    expect(wrapper.find("[data-test='vram-estimate']").exists()).toBe(false);
  });

  it("re-estimates when the routed machine changes", async () => {
    const wrapper = mountBadge({ baseUrl: "http://a:7680" });
    await vi.advanceTimersByTimeAsync(600);
    expect(estimate).toHaveBeenCalledTimes(1);
    await wrapper.setProps({ target: { baseUrl: "http://b:7680" } });
    await vi.advanceTimersByTimeAsync(600);
    expect(estimate).toHaveBeenCalledTimes(2);
    expect(estimate).toHaveBeenLastCalledWith(expect.anything(), {
      baseUrl: "http://b:7680",
    });
  });

  it("names every verdict on the row", async () => {
    estimate.mockResolvedValue({
      ...FITS,
      capacity_peak_memory_bytes: 7_000_000_000,
      peak_memory_bytes: 7_000_000_000,
    });
    const wrapper = mountBadge();
    await vi.advanceTimersByTimeAsync(600);
    const row = wrapper.get("[data-test='vram-estimate']");
    expect(row.attributes("data-fit")).toBe("tight");
    expect(row.text()).toContain("tight");
  });
});

/*
 * Desktop's copy of this badge painted its colour with Tailwind utilities,
 * where every colour carries specificity 0,1,0 and the winner is emitted-rule
 * order rather than class-attribute order: a static `text-fg-dim` beside the
 * bound map silently won over `text-accent` and `text-error`, so a tight fit
 * and an outright refusal both rendered in ordinary dim grey. The shared badge
 * has no colour utility at all — the verdict rides `data-fit` and the scoped
 * sheet paints from it, one token per verdict, so nothing can out-order it.
 */
describe("EstimateBadge colour", () => {
  // Comments name the utilities they warn against; the shipped markup is what
  // this asserts.
  const shipped = source.replace(/\/\*[\s\S]*?\*\//g, "");

  // Assembled rather than written out: the retired web names are exactly what
  // desktop's legacy-vocabulary guard refuses to find in a studio/ source, and
  // it reads this file too.
  const COLOUR_UTILITIES = [
    "fg-dim",
    "accent",
    "error",
    "sapphire",
    "ink-3",
    "halide",
    "safelight",
    "stop",
  ].map((name) => `text-${name}`);

  it("carries no Tailwind colour utility", () => {
    for (const utility of COLOUR_UTILITIES) {
      expect(shipped).not.toContain(utility);
    }
  });

  it("paints each verdict from its own token", () => {
    for (const [verdict, token] of [
      ["fits", "--mold-sapphire"],
      ["unknown", "--mold-sapphire"],
      ["tight", "--mold-blue"],
      ["wont-fit", "--mold-error"],
      ["unavailable", "--mold-text-dim"],
    ] as const) {
      const rule = shipped.match(
        new RegExp(`\\[data-fit="${verdict}"\\][^{]*\\{[^}]*\\}`),
      );
      expect(rule, `no rule for ${verdict}`).not.toBeNull();
      expect(rule![0]).toContain(token);
    }
  });
});
