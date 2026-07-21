import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import Keycap from "./Keycap.vue";

function make(extra: Record<string, unknown> = {}, content = "esc") {
  return mount(Keycap, { props: extra, slots: { default: content } });
}

describe("Keycap", () => {
  it("renders slot content inside a semantic kbd element", () => {
    const wrapper = make();
    expect(wrapper.element.tagName).toBe("KBD");
    expect(wrapper.classes()).toContain("ms-keycap");
    expect(wrapper.text()).toBe("esc");
  });

  it("defaults to the subtle hint variant", () => {
    const wrapper = make();
    expect(wrapper.attributes("data-on-accent")).toBeUndefined();
  });

  it("marks the on-accent variant via data-on-accent", () => {
    const wrapper = make({ onAccent: true }, "⌘↵");
    expect(wrapper.attributes("data-on-accent")).toBe("true");
    expect(wrapper.text()).toBe("⌘↵");
  });
});
