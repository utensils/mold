import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ThumbnailSizeSlider from "./ThumbnailSizeSlider.vue";

describe("ThumbnailSizeSlider", () => {
  it("renders the Lightroom-style small and large thumbnail wedges", () => {
    const wrapper = mount(ThumbnailSizeSlider, {
      props: { modelValue: 220, min: 120, max: 360, step: 10 },
    });

    expect(wrapper.find('[data-test="thumbnail-size-small"]').exists()).toBe(
      true,
    );
    expect(wrapper.find('[data-test="thumbnail-size-large"]').exists()).toBe(
      true,
    );
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
