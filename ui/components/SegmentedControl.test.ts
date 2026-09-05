import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SegmentedControl from "./SegmentedControl.vue";
import SegmentedControlSource from "./SegmentedControl.vue?raw";

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

  /*
   * Two treatments, one component. The accent one says "this picks a mode"
   * (nav rows, Quality, the mesh detail ladder); the neutral one says "this
   * picks a setting" (the toolbar's Still | Clip | 3-D, Keep | Surprise me).
   * With only the accent one implemented the two controls on the New-image
   * view could no longer be told apart, and the accent stopped being one
   * thing.
   */
  describe("variant", () => {
    it("is accent by default and never carries the neutral class", () => {
      expect(make().classes()).not.toContain("ms-seg--neutral");
      expect(make("a", { variant: "accent" }).classes()).not.toContain(
        "ms-seg--neutral",
      );
    });

    it("marks the neutral treatment on the root, where the CSS reads it", () => {
      expect(make("a", { variant: "neutral" }).classes()).toContain(
        "ms-seg--neutral",
      );
    });

    it("changes nothing else about the control", () => {
      const wrapper = make("b", { variant: "neutral" });
      const radios = wrapper.findAll("[role=radio]");
      expect(radios[1]!.attributes("data-on")).toBe("true");
      expect(radios.map((r) => r.attributes("tabindex"))).toEqual([
        "-1",
        "0",
        "-1",
      ]);
    });

    it("fills the neutral active segment without the accent ring", () => {
      expect(SegmentedControlSource).toMatch(
        /\.ms-seg--neutral \.ms-seg__btn\[data-on="true"\] \{[^}]*background: var\(--mold-surface-2\)/s,
      );
      expect(SegmentedControlSource).toMatch(
        /\.ms-seg--neutral \.ms-seg__btn\[data-on="true"\] \{[^}]*box-shadow: none/s,
      );
    });
  });

  it("ignores interaction when disabled", async () => {
    const wrapper = make("a", { disabled: true });
    await wrapper.find("[role=radiogroup]").trigger("keydown", {
      key: "ArrowRight",
    });
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
  });
});
