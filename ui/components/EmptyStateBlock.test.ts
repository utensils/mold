import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { h } from "vue";
import EmptyStateBlock from "./EmptyStateBlock.vue";

function make(extra: Record<string, unknown> = {}, slots = {}) {
  return mount(EmptyStateBlock, {
    props: {
      headline: "Your print develops here",
      guidance: "Describe an image below, pick a look, and press Generate.",
      ...extra,
    },
    slots,
  });
}

describe("EmptyStateBlock", () => {
  it("renders the headline as a heading and the guidance line", () => {
    const wrapper = make();
    expect(wrapper.find("h3").text()).toBe("Your print develops here");
    expect(wrapper.find(".ms-empty__guidance").text()).toContain(
      "press Generate",
    );
  });

  it("defaults to the image icon", () => {
    const svg = make().find(".ms-empty__frame svg");
    expect(svg.exists()).toBe(true);
    expect(svg.attributes("width")).toBe("40");
    // The image glyph's photo rect.
    expect(svg.html()).toContain('rx="2.5"');
  });

  it("renders a custom icon when the icon prop is set", () => {
    const svg = make({ icon: "video" }).find(".ms-empty__frame svg");
    // The video glyph's play-flag path.
    expect(svg.html()).toContain("M16 10.5l5-3v9l-5-3z");
  });

  it("keeps the icon decorative", () => {
    expect(make().find("svg").attributes("aria-hidden")).toBe("true");
  });

  it("applies the brand treatment only when brand is set", () => {
    expect(make().find("h3").classes()).not.toContain(
      "ms-empty__headline--brand",
    );
    expect(make({ brand: true }).find("h3").classes()).toContain(
      "ms-empty__headline--brand",
    );
  });

  it("renders the action slot content when provided", () => {
    const wrapper = make(
      {},
      { action: () => h("button", { type: "button" }, "Browse models") },
    );
    expect(wrapper.find(".ms-empty__action button").text()).toBe(
      "Browse models",
    );
  });

  it("omits the action container when the slot is empty", () => {
    expect(make().find(".ms-empty__action").exists()).toBe(false);
  });
});
