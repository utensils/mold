import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import StageCard from "./StageCard.vue";
import { newStage, type ChainStageForm } from "../../lib/chainForm";

const IMG = "aGVyby1zdGlsbA==";

function mountCard(stage: ChainStageForm) {
  return mount(StageCard, {
    props: {
      stage,
      index: 0,
      baseSeed: "seed",
      canMoveLeft: false,
      canMoveRight: false,
      canRemove: false,
    },
    global: {
      // DevelopCanvas paints on a real 2D context happy-dom doesn't provide.
      stubs: { DevelopCanvas: true },
    },
  });
}

describe("StageCard source image well", () => {
  it("shows an attach button when the stage has no image and emits pick-image", async () => {
    const wrapper = mountCard(newStage("dawn"));
    expect(wrapper.find("[data-test='stage-image-thumb']").exists()).toBe(false);
    const attach = wrapper.find("[data-test='stage-image-attach']");
    expect(attach.exists()).toBe(true);
    await attach.trigger("click");
    expect(wrapper.emitted("pick-image")).toHaveLength(1);
    wrapper.unmount();
  });

  it("renders the attached image as a data-URI thumbnail", () => {
    const wrapper = mountCard({ ...newStage("dawn"), sourceImage: IMG });
    const img = wrapper.find("[data-test='stage-image-thumb'] img");
    expect(img.exists()).toBe(true);
    expect(img.attributes("src")).toBe(`data:image/png;base64,${IMG}`);
    expect(wrapper.find("[data-test='stage-image-attach']").exists()).toBe(false);
    wrapper.unmount();
  });

  it("emits pick-image to replace and clear-image to remove", async () => {
    const wrapper = mountCard({ ...newStage("dawn"), sourceImage: IMG });
    await wrapper.find("[data-test='stage-image-thumb']").trigger("click");
    expect(wrapper.emitted("pick-image")).toHaveLength(1);
    await wrapper.find("[data-test='stage-image-clear']").trigger("click");
    expect(wrapper.emitted("clear-image")).toHaveLength(1);
    wrapper.unmount();
  });
});
