import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MobileStyleChips from "./MobileStyleChips.vue";
import { STYLE_PRESETS } from "../lib/stylePresets";

describe("MobileStyleChips", () => {
  it("collapses the preset row until the header is tapped", async () => {
    const wrapper = mount(MobileStyleChips, { props: { modelValue: "" } });

    expect(wrapper.find("[data-test='mobile-style-row']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-style-toggle']").attributes("aria-expanded")).toBe(
      "false",
    );

    await wrapper.get("[data-test='mobile-style-toggle']").trigger("click");

    expect(wrapper.find("[data-test='mobile-style-row']").exists()).toBe(true);
    expect(wrapper.findAll("[data-test='mobile-style-row'] button")).toHaveLength(
      STYLE_PRESETS.length,
    );
  });

  it("shows the active preset label in the header chip", () => {
    const none = mount(MobileStyleChips, { props: { modelValue: "" } });
    expect(none.get("[data-test='mobile-style-active']").text()).toBe("None");
    expect(none.get("[data-test='mobile-style-active']").attributes("aria-pressed")).toBe("false");

    const cinematic = mount(MobileStyleChips, { props: { modelValue: "cinematic" } });
    expect(cinematic.get("[data-test='mobile-style-active']").text()).toBe("Cinematic");
    expect(cinematic.get("[data-test='mobile-style-active']").attributes("aria-pressed")).toBe(
      "true",
    );
  });

  it("emits the picked preset id and toggles it off when re-tapped", async () => {
    const wrapper = mount(MobileStyleChips, { props: { modelValue: "" } });
    await wrapper.get("[data-test='mobile-style-toggle']").trigger("click");

    await wrapper.get("[data-test='mobile-style-cinematic']").trigger("click");
    expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual(["cinematic"]);

    await wrapper.setProps({ modelValue: "cinematic" });
    await wrapper.get("[data-test='mobile-style-cinematic']").trigger("click");
    expect(wrapper.emitted("update:modelValue")?.at(-1)).toEqual([""]);
  });
});
