import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import CatalogLayoutToggle from "./CatalogLayoutToggle.vue";

describe("CatalogLayoutToggle", () => {
  it("renders an accessible List and Grid radio group", () => {
    const wrapper = mount(CatalogLayoutToggle, {
      props: { modelValue: "list" },
    });
    const radios = wrapper.findAll("[role=radio]");
    expect(wrapper.get("[role=radiogroup]").attributes("aria-label")).toBe(
      "Catalog layout",
    );
    expect(radios.map((radio) => radio.text())).toEqual(["List", "Grid"]);
    expect(radios[0]!.attributes("aria-checked")).toBe("true");
    expect(radios[1]!.attributes("aria-checked")).toBe("false");
  });

  it("emits the selected layout", async () => {
    const wrapper = mount(CatalogLayoutToggle, {
      props: { modelValue: "list" },
    });
    await wrapper.get("[data-test='layout-grid']").trigger("click");
    expect(wrapper.emitted("update:modelValue")).toEqual([["grid"]]);
  });
});
