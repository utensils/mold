import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import HostFilterChips from "./HostFilterChips.vue";

const chips = [
  { key: "local", label: "This Mac", count: 2 },
  { key: "hal9000-7680", label: "hal9000", count: 3 },
];

function mountChips(modelValue = "all") {
  return mount(HostFilterChips, { props: { chips, modelValue } });
}

describe("HostFilterChips", () => {
  it("renders an All chip plus one per source, with counts", () => {
    const wrapper = mountChips();
    const tabs = wrapper.findAll("[role='tab']");
    expect(tabs).toHaveLength(3);
    expect(tabs[0]!.text()).toContain("All");
    expect(tabs[0]!.text()).toContain("5"); // total across sources
    expect(tabs[1]!.text()).toContain("This Mac");
    expect(tabs[1]!.text()).toContain("2");
    expect(tabs[2]!.text()).toContain("hal9000");
    expect(tabs[2]!.text()).toContain("3");
  });

  it("can show a deduplicated All count without changing per-source counts", () => {
    const wrapper = mount(HostFilterChips, {
      props: { chips, modelValue: "all", allCount: 4 },
    });
    const tabs = wrapper.findAll("[role='tab']");
    expect(tabs[0]!.text()).toContain("4");
    expect(tabs[1]!.text()).toContain("2");
    expect(tabs[2]!.text()).toContain("3");
  });

  it("marks the active chip selected", () => {
    const wrapper = mountChips("hal9000-7680");
    const tabs = wrapper.findAll("[role='tab']");
    expect(tabs[0]!.attributes("aria-selected")).toBe("false");
    expect(tabs[2]!.attributes("aria-selected")).toBe("true");
  });

  it("emits the picked source key", async () => {
    const wrapper = mountChips();
    await wrapper.findAll("[role='tab']")[2]!.trigger("click");
    expect(wrapper.emitted("update:modelValue")).toEqual([["hal9000-7680"]]);
  });

  it("emits 'all' from the All chip", async () => {
    const wrapper = mountChips("local");
    await wrapper.findAll("[role='tab']")[0]!.trigger("click");
    expect(wrapper.emitted("update:modelValue")).toEqual([["all"]]);
  });

  it("is keyboard operable — chips are real buttons", () => {
    const wrapper = mountChips();
    for (const tab of wrapper.findAll("[role='tab']")) {
      expect(tab.element.tagName).toBe("BUTTON");
    }
  });
});
