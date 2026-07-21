import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ResultCanvas from "./ResultCanvas.vue";

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
});
