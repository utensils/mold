import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import Stepper from "./Stepper.vue";

function make(modelValue = 2, extra: Record<string, unknown> = {}) {
  return mount(Stepper, {
    props: { modelValue, min: 1, max: 4, label: "Batch", ...extra },
  });
}

function buttons(wrapper: ReturnType<typeof make>) {
  return {
    dec: wrapper.find('[aria-label="Decrease Batch"]'),
    inc: wrapper.find('[aria-label="Increase Batch"]'),
  };
}

describe("Stepper", () => {
  it("exposes a spinbutton with value, bounds and label", () => {
    const wrapper = make(2);
    const spin = wrapper.find("[role=spinbutton]");
    expect(spin.attributes("aria-valuenow")).toBe("2");
    expect(spin.attributes("aria-valuemin")).toBe("1");
    expect(spin.attributes("aria-valuemax")).toBe("4");
    expect(spin.attributes("aria-label")).toBe("Batch");
    expect(spin.attributes("aria-valuetext")).toBe("2");
    expect(spin.attributes("tabindex")).toBe("0");
  });

  it("paints its own box by default and drops it in the compact variant", () => {
    // The compact variant sits inside another bordered control (the
    // composer's Make chip), so it must not paint a second border on top.
    expect(make(2).classes()).not.toContain("ms-stepper--compact");
    expect(make(2, { compact: true }).classes()).toContain(
      "ms-stepper--compact",
    );
  });

  it("renders the current value", () => {
    expect(make(3).find(".ms-stepper__value").text()).toBe("3");
  });

  it("exposes a formatted value without splitting its unit from the number", () => {
    const wrapper = make(24, {
      min: 1,
      max: 60,
      label: "Frames per second",
      format: (value: number) => `${value} fps`,
    });
    expect(wrapper.find(".ms-stepper__value").text()).toBe("24 fps");
    expect(wrapper.find("[role=spinbutton]").attributes("aria-valuetext")).toBe(
      "24 fps",
    );
    expect(
      wrapper.find('[aria-label="Decrease Frames per second"]').exists(),
    ).toBe(true);
    expect(
      wrapper.find('[aria-label="Increase Frames per second"]').exists(),
    ).toBe(true);
  });

  it("increments and decrements on button clicks", async () => {
    const wrapper = make(2);
    await buttons(wrapper).inc.trigger("click");
    await buttons(wrapper).dec.trigger("click");
    expect(wrapper.emitted("update:modelValue")).toEqual([[3], [1]]);
  });

  it("soft-disables the increment button at max and ignores clicks", async () => {
    const wrapper = make(4);
    const { inc, dec } = buttons(wrapper);
    expect(inc.attributes("aria-disabled")).toBe("true");
    expect(dec.attributes("aria-disabled")).toBeUndefined();
    await inc.trigger("click");
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
  });

  it("soft-disables the decrement button at min and ignores clicks", async () => {
    const wrapper = make(1);
    const { dec, inc } = buttons(wrapper);
    expect(dec.attributes("aria-disabled")).toBe("true");
    expect(inc.attributes("aria-disabled")).toBeUndefined();
    await dec.trigger("click");
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
  });

  it("adjusts with ArrowUp and ArrowDown", async () => {
    const wrapper = make(2);
    const spin = wrapper.find("[role=spinbutton]");
    await spin.trigger("keydown", { key: "ArrowUp" });
    await spin.trigger("keydown", { key: "ArrowDown" });
    expect(wrapper.emitted("update:modelValue")).toEqual([[3], [1]]);
  });

  it("does not emit past the bounds via keyboard", async () => {
    const wrapper = make(4);
    await wrapper.find("[role=spinbutton]").trigger("keydown", {
      key: "ArrowUp",
    });
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
  });

  it("clamps an out-of-range value back into bounds on adjust", async () => {
    const wrapper = make(9);
    await wrapper.find("[role=spinbutton]").trigger("keydown", {
      key: "ArrowDown",
    });
    expect(wrapper.emitted("update:modelValue")).toEqual([[4]]);
  });
});
