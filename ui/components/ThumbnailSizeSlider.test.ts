import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ThumbnailSizeSlider from "./ThumbnailSizeSlider.vue";

describe("ThumbnailSizeSlider", () => {
  it("is a bare track led by a 13px grid glyph, with no box of its own", () => {
    const wrapper = mount(ThumbnailSizeSlider, {
      props: { modelValue: 220, min: 120, max: 360, step: 10 },
    });

    // The 34px bordered box overflowed the 40px view toolbar; the control is
    // a glyph beside a hairline track now.
    const glyph = wrapper.get('[data-test="thumbnail-size-glyph"]');
    expect(glyph.attributes("width")).toBe("13");
    expect(glyph.attributes("height")).toBe("13");
    expect(wrapper.find(".ms-thumbnail-size__track").exists()).toBe(true);
  });

  it("fills the track in proportion to the size within its range", async () => {
    const wrapper = mount(ThumbnailSizeSlider, {
      props: { modelValue: 120, min: 120, max: 360, step: 10 },
    });
    const fill = () =>
      (wrapper.get('[data-test="thumbnail-size-fill"]').element as HTMLElement)
        .style.width;

    expect(fill()).toBe("0%");
    await wrapper.setProps({ modelValue: 240 });
    expect(fill()).toBe("50%");
    await wrapper.setProps({ modelValue: 360 });
    expect(fill()).toBe("100%");
  });

  it("exposes an accessible pixel-valued range and emits numeric input", async () => {
    const wrapper = mount(ThumbnailSizeSlider, {
      props: { modelValue: 220, min: 120, max: 360, step: 10 },
    });
    const input = wrapper.get('input[aria-label="Thumbnail size"]');

    expect(input.attributes()).toMatchObject({
      min: "120",
      max: "360",
      step: "10",
      "aria-valuetext": "220 px",
    });

    (input.element as HTMLInputElement).value = "280";
    await input.trigger("input");
    expect(wrapper.emitted("update:modelValue")).toEqual([[280]]);
  });
});
