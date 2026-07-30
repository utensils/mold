import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ResultCanvas from "./ResultCanvas.vue";
import DevelopCanvas from "@ui/components/DevelopCanvas.vue";
import ProgressRing from "@ui/components/ProgressRing.vue";
import ErrorNotice from "@ui/components/ErrorNotice.vue";

describe("ResultCanvas", () => {
  it("renders the brand empty state", () => {
    const wrapper = mount(ResultCanvas, { props: { mode: "empty" } });
    expect(wrapper.text()).toContain("Your print develops here");
    expect(wrapper.text()).toContain("runs on your own machine");
  });

  it("renders the generating bed with the stage line", () => {
    const wrapper = mount(ResultCanvas, {
      props: { mode: "generating", progress: 42, stage: "Developing 12 / 28" },
    });
    expect(wrapper.find("[data-test='canvas-generating']").exists()).toBe(true);
    expect(wrapper.get("[data-test='canvas-stage']").text()).toBe(
      "Developing 12 / 28",
    );
    expect(wrapper.find(".ms-shimmer").exists()).toBe(true);
  });

  it("keeps the ring and stage until the first latent preview arrives", () => {
    const wrapper = mount(ResultCanvas, {
      props: {
        mode: "generating",
        progress: 10,
        stage: "Loading model",
        developSeed: "42",
        developPhase: "latent",
        progressFraction: 0,
      },
    });
    expect(wrapper.findComponent(ProgressRing).exists()).toBe(true);
    expect(wrapper.find("[data-test='canvas-preview']").exists()).toBe(false);
    expect(wrapper.get("[data-test='canvas-stage']").text()).toBe(
      "Loading model",
    );
    // The develop grain is the signature — it renders from the first frame.
    expect(wrapper.findComponent(DevelopCanvas).exists()).toBe(true);
    expect(wrapper.findComponent(DevelopCanvas).attributes("style")).toContain(
      "opacity: 1",
    );
  });

  it("renders the latent preview under thinning grain once one arrives", () => {
    const wrapper = mount(ResultCanvas, {
      props: {
        mode: "generating",
        progress: 50,
        stage: "Developing 4 / 8",
        previewSrc: "data:image/png;base64,AAAA",
        developSeed: "42",
        developPhase: "developing",
        progressFraction: 0.5,
      },
    });
    const preview = wrapper.get("[data-test='canvas-preview']");
    expect(preview.attributes("src")).toBe("data:image/png;base64,AAAA");
    // blur(max(2, 14 − 12·p)) at p = 0.5 → 8px.
    expect(preview.attributes("style")).toContain("blur(8px)");
    // Grain thins with progress: max(0.18, 1 − 0.9·0.5) = 0.55.
    expect(wrapper.findComponent(DevelopCanvas).attributes("style")).toContain(
      "opacity: 0.55",
    );
    // The forming print replaces the ring + stage overlay.
    expect(wrapper.findComponent(ProgressRing).exists()).toBe(false);
    expect(wrapper.find("[data-test='canvas-stage']").exists()).toBe(false);
    expect(wrapper.find("[data-test='canvas-generating']").exists()).toBe(true);
  });

  it("starts the preview blur at 14px and floors it at 2px", () => {
    const start = mount(ResultCanvas, {
      props: {
        mode: "generating",
        previewSrc: "data:image/png;base64,AAAA",
        developSeed: "1",
        progressFraction: 0,
      },
    });
    expect(
      start.get("[data-test='canvas-preview']").attributes("style"),
    ).toContain("blur(14px)");

    const end = mount(ResultCanvas, {
      props: {
        mode: "generating",
        previewSrc: "data:image/png;base64,AAAA",
        developSeed: "1",
        progressFraction: 1,
      },
    });
    expect(
      end.get("[data-test='canvas-preview']").attributes("style"),
    ).toContain("blur(2px)");
  });

  it("adopts the print aspect ratio for the develop bed when dims are known", () => {
    const wrapper = mount(ResultCanvas, {
      props: {
        mode: "generating",
        developSeed: "1",
        printWidth: 1216,
        printHeight: 704,
      },
    });
    expect(wrapper.get(".canvas__bed").attributes("style")).toContain(
      "aspect-ratio: 1216 / 704",
    );
  });

  it("keeps the square bed when print dims are not provided", () => {
    const wrapper = mount(ResultCanvas, {
      props: { mode: "generating", developSeed: "1" },
    });
    expect(wrapper.get(".canvas__bed").attributes("style") ?? "").not.toContain(
      "aspect-ratio",
    );
  });

  it("renders a copyable error notice", () => {
    const wrapper = mount(ResultCanvas, {
      props: {
        mode: "error",
        error: "plato couldn’t find that model or job.",
        errorCopy:
          "plato couldn’t find that model or job.\n\nTechnical details: HTTP 404",
      },
    });
    expect(wrapper.findComponent(ErrorNotice).props()).toMatchObject({
      message: "plato couldn’t find that model or job.",
      copyMessage:
        "plato couldn’t find that model or job.\n\nTechnical details: HTTP 404",
    });
  });

  it("renders a result image and caption", () => {
    const wrapper = mount(ResultCanvas, {
      props: {
        mode: "result",
        resultSrc: "blob:x",
        resultCaption: "flux-dev:q4 · seed 184023 · 12s · this server",
      },
    });
    expect(
      wrapper.get("[data-test='canvas-result'] img").attributes("src"),
    ).toBe("blob:x");
    expect(wrapper.get("[data-test='canvas-caption']").text()).toBe(
      "flux-dev:q4 · seed 184023 · 12s · this server",
    );
  });

  it("renders a generation error instead of silently returning to empty", () => {
    const wrapper = mount(ResultCanvas, {
      props: { mode: "error", error: "CUDA out of memory" },
    });
    expect(wrapper.get("[data-test='canvas-error']").text()).toContain(
      "CUDA out of memory",
    );
  });

  it("renders editable variations and emits edits, use, discard and queue", async () => {
    const wrapper = mount(ResultCanvas, {
      props: { mode: "variations", variations: ["one", "two"] },
    });
    expect(wrapper.get("[data-test='canvas-variations']").text()).toContain(
      "2 variations ready",
    );
    expect(wrapper.get("[data-test='variations-queue']").text()).toContain(
      "Queue 2 prints",
    );

    const input = wrapper.get("[data-test='variation-input-0']")
      .element as HTMLTextAreaElement;
    input.value = "one edited";
    await wrapper.get("[data-test='variation-input-0']").trigger("input");
    expect(wrapper.emitted("update:variations")?.[0]).toEqual([
      ["one edited", "two"],
    ]);

    await wrapper.get("[data-test='variation-use-1']").trigger("click");
    expect(wrapper.emitted("use-variation")?.[0]).toEqual([1]);

    await wrapper.get("[data-test='variations-discard']").trigger("click");
    expect(wrapper.emitted("discard")).toHaveLength(1);

    await wrapper.get("[data-test='variations-queue']").trigger("click");
    expect(wrapper.emitted("queue")).toHaveLength(1);
  });

  it("keeps large prepared batches compact and can reveal every prompt", async () => {
    const variations = Array.from(
      { length: 120 },
      (_, index) => `variation ${index + 1}`,
    );
    const wrapper = mount(ResultCanvas, {
      props: { mode: "variations", variations },
    });

    expect(wrapper.findAll(".canvas__var-row")).toHaveLength(8);
    expect(
      wrapper.get("[data-test='variations-compact-review']").text(),
    ).toContain("112 more variations");
    expect(wrapper.get("[data-test='variations-queue']").text()).toContain(
      "Queue 120 prints",
    );

    await wrapper.get("[data-test='variations-review-all']").trigger("click");
    expect(wrapper.findAll(".canvas__var-row")).toHaveLength(50);
    expect(
      wrapper.get("[data-test='variations-review-range']").text(),
    ).toContain("Variations 1–50 of 120");
    await wrapper.get("[data-test='variations-review-next']").trigger("click");
    expect(wrapper.findAll(".canvas__var-row")).toHaveLength(50);
    expect(
      wrapper.get("[data-test='variations-review-range']").text(),
    ).toContain("Variations 51–100 of 120");
    expect(
      wrapper.get("[data-test='variation-input-50']").attributes("data-test"),
    ).toBe("variation-input-50");

    await wrapper.setProps({ variationBatchId: "replacement" });
    expect(wrapper.findAll(".canvas__var-row")).toHaveLength(8);
    expect(wrapper.find("[data-test='variations-review-range']").exists()).toBe(
      false,
    );
  });

  it("freezes the reviewed set while its route is being revalidated", () => {
    const wrapper = mount(ResultCanvas, {
      props: {
        mode: "variations",
        variations: ["one", "two"],
        queueingVariations: true,
      },
    });
    expect(
      wrapper.get("[data-test='variations-queue']").attributes("disabled"),
    ).toBeDefined();
    expect(
      wrapper.get("[data-test='variations-discard']").attributes("disabled"),
    ).toBeDefined();
    expect(
      wrapper.get("[data-test='variation-input-0']").attributes("disabled"),
    ).toBeDefined();
    expect(wrapper.get("[data-test='variations-queue']").text()).toContain(
      "Checking machine",
    );
  });
});
