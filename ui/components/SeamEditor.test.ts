import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SeamEditor from "./SeamEditor.vue";

function make(props: Record<string, unknown> = {}) {
  return mount(SeamEditor, {
    props: {
      transition: "fade",
      fadeFrames: 8,
      motionTail: 17,
      fps: 24,
      ...props,
    },
  });
}

describe("SeamEditor", () => {
  it("teaches all three transitions with labels and descriptions", () => {
    const wrapper = make();
    const text = wrapper.text();
    expect(text).toContain("Smooth");
    expect(text).toContain("Motion carries through");
    expect(text).toContain("Cut");
    expect(text).toContain("Hard change of shot");
    expect(text).toContain("Fade");
    expect(text).toContain("Blend across the seam");
  });

  it("relabels smooth as Join at zero motion tail", () => {
    const wrapper = make({ motionTail: 0 });
    expect(wrapper.text()).toContain("Join");
    expect(wrapper.text()).not.toContain("Join clips");
    expect(wrapper.text()).toContain("Clips join end to end");
  });

  it("checks the selected row and emits an update on pick", async () => {
    const wrapper = make();
    const rows = wrapper.findAll("[role=radio]");
    expect(rows[2]?.attributes("aria-checked")).toBe("true");
    await rows[1]?.trigger("click");
    expect(wrapper.emitted("update:transition")?.[0]).toEqual(["cut"]);
  });

  it("contains the radios in a labelled radiogroup (ARIA pattern)", () => {
    const wrapper = make();
    const group = wrapper.find("[role=radiogroup]");
    expect(group.exists()).toBe(true);
    expect(group.attributes("aria-label")).toBe("Transition");
    expect(group.findAll("[role=radio]")).toHaveLength(3);
  });

  it("emits apply-all on an alt-modified pick", async () => {
    const wrapper = make();
    const rows = wrapper.findAll("[role=radio]");
    await rows[0]?.trigger("click", { altKey: true });
    expect(wrapper.emitted("apply-all")?.[0]).toEqual(["smooth"]);
    expect(wrapper.emitted("update:transition")).toBeUndefined();
  });

  it("shows the fade stepper with seconds math and the server cap", () => {
    const wrapper = make({ fadeFramesMax: 32 });
    expect(wrapper.text()).toContain("Fade length");
    expect(wrapper.text()).toContain("8f");
    expect(wrapper.text()).toContain("0.33s @ 24fps");
    expect(wrapper.text()).toContain("max 32f");
  });

  it("hides fade controls for non-fade transitions", () => {
    const wrapper = make({ transition: "smooth" });
    expect(wrapper.text()).not.toContain("Fade length");
  });

  it("clamps fade length to the server cap via the stepper", async () => {
    const wrapper = make({ fadeFrames: 32, fadeFramesMax: 32 });
    const increase = wrapper.find(
      '[aria-label="Increase Fade length in frames"]',
    );
    await increase.trigger("click");
    expect(wrapper.emitted("update:fadeFrames")).toBeUndefined();
  });

  it("teaches with the same circular glyph badges the seam pill shows", () => {
    const wrapper = make();
    const badges = wrapper.findAll(
      ".ms-seam-editor__diagram .ms-seam-editor__glyph",
    );
    expect(badges).toHaveLength(3);
    expect(badges.every((badge) => badge.find("rect").exists())).toBe(true);
    // The legacy rectangle swatches are gone.
    expect(wrapper.find(".ms-seam-editor__line-smooth").exists()).toBe(false);
    expect(wrapper.find(".ms-seam-editor__line-cut").exists()).toBe(false);
    expect(wrapper.find(".ms-seam-editor__line-fade").exists()).toBe(false);
  });

  it("shows the seam context and the apply-all hint when asked", () => {
    const wrapper = make({
      fromLabel: "opening",
      toLabel: "clip 2",
      showApplyAllHint: true,
    });
    expect(wrapper.text()).toContain("join · opening → clip 2");
    expect(wrapper.text()).toContain("⌥ applies to every seam");
  });
});
