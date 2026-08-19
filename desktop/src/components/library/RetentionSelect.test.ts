import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import RetentionSelect from "./RetentionSelect.vue";

describe("RetentionSelect", () => {
  it("offers the shared retention ladder with human labels and selects the current value", () => {
    const wrapper = mount(RetentionSelect, { props: { modelValue: 30 } });
    const options = wrapper.findAll("option");
    expect(options.map((o) => o.text())).toEqual([
      "1 day",
      "7 days",
      "30 days",
      "90 days",
      "1 year",
      "Forever",
    ]);
    expect(options.map((o) => o.attributes("value"))).toEqual(["1", "7", "30", "90", "365", "0"]);
    expect(wrapper.find<HTMLSelectElement>("select").element.value).toBe("30");
    expect(wrapper.find("select").attributes("aria-label")).toBe("Trash retention");
  });

  it("emits the chosen number (0 = forever)", async () => {
    const wrapper = mount(RetentionSelect, { props: { modelValue: 30 } });
    await wrapper.find("select").setValue("0");
    await wrapper.find("select").setValue("7");
    expect(wrapper.emitted("update:modelValue")).toEqual([[0], [7]]);
  });

  it("keeps an off-ladder server value visible instead of rewriting it", () => {
    const wrapper = mount(RetentionSelect, { props: { modelValue: 14 } });
    expect(wrapper.findAll("option").map((o) => o.text())).toEqual([
      "1 day",
      "7 days",
      "14 days",
      "30 days",
      "90 days",
      "1 year",
      "Forever",
    ]);
    expect(wrapper.find<HTMLSelectElement>("select").element.value).toBe("14");
  });

  it("renders the hint and honours disabled", () => {
    const wrapper = mount(RetentionSelect, {
      props: { modelValue: 0, disabled: true, hint: "Set by MOLD_GALLERY_TRASH_RETENTION_DAYS" },
    });
    expect(wrapper.find("[data-test='retention-hint']").text()).toBe(
      "Set by MOLD_GALLERY_TRASH_RETENTION_DAYS",
    );
    expect(wrapper.find("select").attributes("disabled")).toBeDefined();
    expect(wrapper.find<HTMLSelectElement>("select").element.value).toBe("0");
  });
});
