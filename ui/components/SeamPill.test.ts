import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SeamPill from "./SeamPill.vue";

describe("SeamPill", () => {
  it("names the transition in words", () => {
    const wrapper = mount(SeamPill, {
      props: { transition: "smooth", motionTail: 17 },
    });
    expect(wrapper.text()).toContain("Smooth");
  });

  it("renders Join clips when the motion tail is zero (LTX-Video)", () => {
    // The SpliceMark regression: it defaulted motionTail and always said
    // "Continue motion". motionTail is a REQUIRED prop here, and zero must
    // relabel the seam honestly.
    const wrapper = mount(SeamPill, {
      props: { transition: "smooth", motionTail: 0 },
    });
    expect(wrapper.text()).toContain("Join clips");
    expect(wrapper.text()).not.toContain("Smooth");
  });

  it("shows the fade length beside the word", () => {
    const wrapper = mount(SeamPill, {
      props: { transition: "fade", motionTail: 17, fadeFrames: 8 },
    });
    expect(wrapper.text()).toContain("Fade");
    expect(wrapper.text()).toContain("8f");
    expect(wrapper.attributes("aria-label")).toBe("Transition: Fade 8f");
  });

  it("marks the open seam and disables cleanly", async () => {
    const wrapper = mount(SeamPill, {
      props: { transition: "cut", motionTail: 17, active: true },
    });
    expect(wrapper.attributes("data-on")).toBe("true");

    await wrapper.setProps({ disabled: true });
    expect(wrapper.attributes("disabled")).toBeDefined();
  });

  it("renders the circular glyph badge and no legacy chevron lozenge", () => {
    for (const transition of ["smooth", "cut", "fade"] as const) {
      const wrapper = mount(SeamPill, {
        props: { transition, motionTail: 17 },
      });
      expect(wrapper.find(".ms-seam__diagram .ms-seam__glyph rect").exists()).toBe(true);
      expect(wrapper.find(".ms-seam__chevron").exists()).toBe(false);
      expect(wrapper.attributes("data-transition")).toBe(transition);
    }
  });

  it("emits click", async () => {
    const wrapper = mount(SeamPill, {
      props: { transition: "cut", motionTail: 17 },
    });
    await wrapper.trigger("click");
    expect(wrapper.emitted("click")).toHaveLength(1);
  });

  it("opens the editor from a right-click too", async () => {
    // Right-clicking a transition is a natural "edit this" gesture; it must
    // reach the same seam editor as a left click instead of a browser menu.
    const wrapper = mount(SeamPill, {
      props: { transition: "fade", motionTail: 17, fadeFrames: 8 },
    });
    await wrapper.trigger("contextmenu");
    expect(wrapper.emitted("click")).toHaveLength(1);
  });
});
