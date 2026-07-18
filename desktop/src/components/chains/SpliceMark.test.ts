import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SpliceMark from "./SpliceMark.vue";
import type { ChainStageForm } from "../../lib/chainForm";

function stage(transition: ChainStageForm["transition"]): ChainStageForm {
  return {
    prompt: "a cat",
    negativePrompt: "",
    frames: 97,
    transition,
    fadeFrames: 8,
    sourceImage: null,
  };
}

function mountMark(transition: ChainStageForm["transition"]) {
  return mount(SpliceMark, {
    props: { stage: stage(transition), motionTail: 12, fadeMax: 24 },
  });
}

describe("SpliceMark a11y", () => {
  it("gives the cycle button an accessible name reflecting the transition", () => {
    const wrapper = mountMark("smooth");
    const btn = wrapper.get("button");
    expect(btn.attributes("aria-label")).toContain("smooth");
  });

  it("labels the fade steppers when the transition is fade", () => {
    const wrapper = mountMark("fade");
    const labels = wrapper.findAll("button").map((b) => b.attributes("aria-label"));
    expect(labels).toContain("Fewer fade frames");
    expect(labels).toContain("More fade frames");
  });
});
