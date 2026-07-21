import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import CardSurface from "./CardSurface.vue";

function make(props: Record<string, unknown> = {}) {
  return mount(CardSurface, {
    props,
    slots: { default: "<p>Card content</p>" },
  });
}

describe("CardSurface", () => {
  it("renders slot content", () => {
    const wrapper = make();
    expect(wrapper.text()).toContain("Card content");
  });

  it("is padded by default", () => {
    const wrapper = make();
    expect(wrapper.classes()).toContain("ms-card--padded");
  });

  it("drops padding when padded is false", () => {
    const wrapper = make({ padded: false });
    expect(wrapper.classes()).not.toContain("ms-card--padded");
  });

  it("applies the large radius modifier", () => {
    expect(make().classes()).not.toContain("ms-card--large");
    expect(make({ large: true }).classes()).toContain("ms-card--large");
  });

  it("applies the dashed modifier for empty/offer cards", () => {
    expect(make().classes()).not.toContain("ms-card--dashed");
    expect(make({ dashed: true }).classes()).toContain("ms-card--dashed");
  });

  it("is a non-interactive div", () => {
    const wrapper = make();
    expect(wrapper.element.tagName).toBe("DIV");
    expect(wrapper.attributes("role")).toBeUndefined();
  });
});
