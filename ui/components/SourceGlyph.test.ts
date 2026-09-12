import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { defineComponent, h } from "vue";
import SourceGlyph from "./SourceGlyph.vue";

describe("SourceGlyph", () => {
  it("renders the Hugging Face mark", () => {
    const wrapper = mount(SourceGlyph, { props: { source: "hf" } });
    const svg = wrapper.get("svg");
    expect(svg.attributes("role")).toBe("img");
    expect(svg.attributes("data-source")).toBe("hf");
    expect(svg.attributes("aria-label")).toBe("Hugging Face");
  });

  it("renders the Civitai mark", () => {
    const wrapper = mount(SourceGlyph, { props: { source: "civitai" } });
    const svg = wrapper.get("svg");
    expect(svg.attributes("role")).toBe("img");
    expect(svg.attributes("data-source")).toBe("civitai");
    expect(svg.attributes("aria-label")).toBe("Civitai");
  });

  it("renders the local file mark", () => {
    const wrapper = mount(SourceGlyph, { props: { source: "local" } });
    const svg = wrapper.get("svg");
    expect(svg.attributes("role")).toBe("img");
    expect(svg.attributes("data-source")).toBe("local");
    expect(svg.attributes("aria-label")).toBe("Local file");
  });

  it("sizes width and height from the size prop, defaulting to 12", () => {
    const defaultSize = mount(SourceGlyph, { props: { source: "hf" } }).get(
      "svg",
    );
    expect(defaultSize.attributes("width")).toBe("12");
    expect(defaultSize.attributes("height")).toBe("12");

    const sized = mount(SourceGlyph, { props: { source: "hf", size: 16 } }).get(
      "svg",
    );
    expect(sized.attributes("width")).toBe("16");
    expect(sized.attributes("height")).toBe("16");
  });

  it("gives Civitai's gradients unique ids across two rows in one app", () => {
    // `useId()` counts within its owning app instance, so the two rows must
    // share one app — the way two rows in the same menu actually render —
    // for a collision to be observable at all.
    const Rows = defineComponent({
      render: () => [
        h(SourceGlyph, { source: "civitai" }),
        h(SourceGlyph, { source: "civitai" }),
      ],
    });
    const wrapper = mount(Rows);
    const ids = wrapper
      .findAll("linearGradient")
      .map((n) => n.attributes("id"));
    expect(ids).toHaveLength(4);
    expect(new Set(ids).size).toBe(4);
  });
});
