import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SegmentedControl from "./SegmentedControl.vue";

const OPTIONS = [
  { value: "a", label: "Alpha" },
  { value: "b", label: "Beta", sub: "Standard" },
  { value: "c", label: "Gamma" },
] as const;

function make(modelValue = "a", extra: Record<string, unknown> = {}) {
  return mount(SegmentedControl, {
    props: { modelValue, options: OPTIONS, label: "test group", ...extra },
  });
}

describe("SegmentedControl", () => {
  it("renders one radio per option with the active one checked", () => {
    const wrapper = make("b");
    const radios = wrapper.findAll("[role=radio]");
    expect(radios).toHaveLength(3);
    expect(radios[1]!.attributes("aria-checked")).toBe("true");
    expect(radios[0]!.attributes("aria-checked")).toBe("false");
    expect(radios[1]!.attributes("data-on")).toBe("true");
  });

  it("emits update:modelValue on click", async () => {
    const wrapper = make("a");
    await wrapper.findAll("button")[2]!.trigger("click");
    expect(wrapper.emitted("update:modelValue")).toEqual([["c"]]);
  });

  it("renders sub-labels when provided", () => {
    const wrapper = make();
    expect(wrapper.text()).toContain("Standard");
  });

  it("moves selection with arrow keys and wraps", async () => {
    const wrapper = make("c");
    await wrapper.find("[role=radiogroup]").trigger("keydown", {
      key: "ArrowRight",
    });
    expect(wrapper.emitted("update:modelValue")).toEqual([["a"]]);
  });

  it("only the active segment is tabbable (roving tabindex)", () => {
    const wrapper = make("b");
    const tabs = wrapper.findAll("button").map((b) => b.attributes("tabindex"));
    expect(tabs).toEqual(["-1", "0", "-1"]);
  });

  it("applies the compact modifier class only when requested", () => {
    expect(make().classes()).not.toContain("ms-seg--compact");
    expect(make("a", { compact: true }).classes()).toContain("ms-seg--compact");
  });

  it("ignores interaction when disabled", async () => {
    const wrapper = make("a", { disabled: true });
    await wrapper.find("[role=radiogroup]").trigger("keydown", {
      key: "ArrowRight",
    });
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
  });
});
