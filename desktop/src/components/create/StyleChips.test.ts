import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import StyleChips from "./StyleChips.vue";

afterEach(() => (document.body.innerHTML = ""));

describe("StyleChips", () => {
  it("labels the active preset and toggles the row open", async () => {
    const wrapper = mount(StyleChips, { props: { modelValue: "cinematic" } });
    expect(wrapper.get("[data-test='style-active']").text()).toBe("Cinematic");
    expect(wrapper.find("[data-test='style-anime']").exists()).toBe(false);
    await wrapper.get("[data-test='style-toggle']").trigger("click");
    expect(wrapper.find("[data-test='style-anime']").exists()).toBe(true);
  });

  it("selects a preset and clears it when re-picked", async () => {
    const wrapper = mount(StyleChips, { props: { modelValue: "" } });
    await wrapper.get("[data-test='style-toggle']").trigger("click");
    await wrapper.get("[data-test='style-cinematic']").trigger("click");
    expect(wrapper.emitted("update:modelValue")?.[0]).toEqual(["cinematic"]);

    await wrapper.setProps({ modelValue: "cinematic" });
    await wrapper.get("[data-test='style-cinematic']").trigger("click");
    expect(wrapper.emitted("update:modelValue")?.[1]).toEqual([""]);
  });
});
