import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SliderRow from "./SliderRow.vue";

function make(extra: Record<string, unknown> = {}) {
  return mount(SliderRow, {
    props: { modelValue: 28, min: 6, max: 50, label: "Detail", ...extra },
  });
}

describe("SliderRow", () => {
  it("renders the label and the raw value as the default readout", () => {
    const wrapper = make();
    expect(wrapper.find(".ms-slider__label").text()).toBe("Detail");
    expect(wrapper.find(".ms-slider__value").text()).toBe("28");
  });

  it("prefers valueLabel for the readout when provided", () => {
    const wrapper = make({ valueLabel: "28 steps" });
    expect(wrapper.find(".ms-slider__value").text()).toBe("28 steps");
  });

  it("forwards min, max and step to the range input", () => {
    const wrapper = make({ step: 0.5 });
    const input = wrapper.find("input[type=range]");
    expect(input.attributes("min")).toBe("6");
    expect(input.attributes("max")).toBe("50");
    expect(input.attributes("step")).toBe("0.5");
  });

  it("defaults step to 1", () => {
    const wrapper = make();
    expect(wrapper.find("input[type=range]").attributes("step")).toBe("1");
  });

  it("emits update:modelValue with a number on input", async () => {
    const wrapper = make();
    const input = wrapper.find("input[type=range]");
    (input.element as HTMLInputElement).value = "34";
    await input.trigger("input");
    expect(wrapper.emitted("update:modelValue")).toEqual([[34]]);
  });

  it("labels the input and mirrors the readout in aria-valuetext", () => {
    const wrapper = make({ valueLabel: "28 steps" });
    const input = wrapper.find("input[type=range]");
    expect(input.attributes("aria-label")).toBe("Detail");
    expect(input.attributes("aria-valuetext")).toBe("28 steps");
  });
});
