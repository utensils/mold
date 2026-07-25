import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import BadgePill from "@ui/components/BadgePill.vue";
import ModelMetadataBadges from "./ModelMetadataBadges.vue";

describe("ModelMetadataBadges", () => {
  it("renders readable modality and kind classifications", () => {
    const wrapper = mount(ModelMetadataBadges, {
      props: { modality: "image", kind: "text-encoder" },
    });

    expect(wrapper.get('[data-test="model-modality-badge"]').text()).toBe(
      "Image",
    );
    expect(wrapper.get('[data-test="model-kind-badge"]').text()).toBe(
      "Text encoder",
    );
  });

  it("renders NSFW as explicit danger text rather than color alone", () => {
    const wrapper = mount(ModelMetadataBadges, {
      props: { family: "flux2", nsfw: true, showModality: false },
    });

    const badges = wrapper.findAllComponents(BadgePill);
    expect(wrapper.get('[data-test="model-kind-badge"]').text()).toBe(
      "Checkpoint",
    );
    const nsfw = wrapper.get('[data-test="model-nsfw-badge"]');
    expect(nsfw.text()).toBe("18+ NSFW");
    expect(nsfw.attributes("aria-label")).toBe("Mature content (NSFW)");
    expect(badges.at(-1)?.props("tone")).toBe("danger");
  });

  it("omits safety tags for both known-safe and unknown content", () => {
    const safe = mount(ModelMetadataBadges, {
      props: { kind: "lora", nsfw: false, showModality: false },
    });
    const unknown = mount(ModelMetadataBadges, {
      props: { kind: "lora", nsfw: null, showModality: false },
    });

    for (const wrapper of [safe, unknown]) {
      expect(wrapper.find('[data-test="model-nsfw-badge"]').exists()).toBe(
        false,
      );
      expect(wrapper.text()).not.toContain("Safe");
      expect(wrapper.text()).not.toContain("NSFW");
    }
  });
});
