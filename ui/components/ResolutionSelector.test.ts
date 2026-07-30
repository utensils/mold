import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ResolutionSelector from "./ResolutionSelector.vue";
import { RESOLUTIONS, dimsLabel } from "../lib/resolution";

function make(modelValue = 1, ratio = 1, extra: Record<string, unknown> = {}) {
  return mount(ResolutionSelector, {
    props: { modelValue, ratio, ...extra },
  });
}

describe("ResolutionSelector", () => {
  it("renders one segment per resolution step with labels and subs", () => {
    const wrapper = make();
    const radios = wrapper.findAll("[role=radio]");
    expect(radios).toHaveLength(RESOLUTIONS.length);
    for (const step of RESOLUTIONS) {
      expect(wrapper.text()).toContain(step.label);
      expect(wrapper.text()).toContain(step.sub);
    }
  });

  it("marks the active megapixel step checked", () => {
    const wrapper = make(1.8);
    const radios = wrapper.findAll("[role=radio]");
    const index = RESOLUTIONS.findIndex((r) => r.mp === 1.8);
    expect(radios[index]!.attributes("aria-checked")).toBe("true");
    expect(radios[0]!.attributes("aria-checked")).toBe("false");
  });

  it("emits update:modelValue with the megapixel number on click", async () => {
    const wrapper = make(1);
    const index = RESOLUTIONS.findIndex((r) => r.mp === 0.5);
    await wrapper.findAll("button")[index]!.trigger("click");
    expect(wrapper.emitted("update:modelValue")).toEqual([[0.5]]);
  });

  it("resolves 1 MP at ratio 1 to 1024×1024 px", () => {
    const wrapper = make(1, 1);
    expect(wrapper.find(".ms-res__dims").text()).toBe("1024×1024 px");
  });

  it("updates the resolved line when the ratio changes", async () => {
    const wrapper = make(1, 1);
    expect(wrapper.find(".ms-res__dims").text()).toBe("1024×1024 px");
    await wrapper.setProps({ ratio: 16 / 9 });
    const expected = `${dimsLabel(1, 16 / 9)} px`;
    expect(wrapper.find(".ms-res__dims").text()).toBe(expected);
    expect(wrapper.find(".ms-res__dims").text()).not.toBe("1024×1024 px");
  });

  it("updates the resolved line when the megapixel value changes", async () => {
    const wrapper = make(1, 1);
    await wrapper.setProps({ modelValue: 2 });
    expect(wrapper.find(".ms-res__dims").text()).toBe(`${dimsLabel(2, 1)} px`);
  });

  it("accepts custom options", () => {
    const options = [
      { mp: 4, label: "4 MP", sub: "Huge" },
      { mp: 8, label: "8 MP" },
    ] as const;
    const wrapper = make(4, 1, { options });
    expect(wrapper.findAll("[role=radio]")).toHaveLength(2);
    expect(wrapper.text()).toContain("Huge");
  });

  it("forwards disabled to the segmented control", async () => {
    const wrapper = make(1, 1, { disabled: true });
    await wrapper.find("[role=radiogroup]").trigger("keydown", {
      key: "ArrowRight",
    });
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
    expect(wrapper.find("[role=radiogroup]").attributes("aria-disabled")).toBe(
      "true",
    );
  });
});
