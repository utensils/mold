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

  it("renders every numbered mark in a dedicated row", () => {
    const wrapper = make({
      marks: [
        { value: 14, label: "1x" },
        { value: 28, label: "2x" },
        { value: 42, label: "3x" },
      ],
    });

    expect(wrapper.get(".ms-slider__track").classes()).toContain(
      "ms-slider__track--marked",
    );
    expect(
      wrapper.findAll(".ms-slider__mark b").map((mark) => mark.text()),
    ).toEqual(["1x", "2x", "3x"]);
    const marks = wrapper.get(".ms-slider__marks").element;
    const input = wrapper.get("input[type=range]").element;
    expect(
      marks.compareDocumentPosition(input) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("snaps a pointer drag when mobile commits change before pointerup", async () => {
    const wrapper = make({
      modelValue: 30,
      step: 2,
      marks: [
        { value: 20, label: "1x" },
        { value: 40, label: "2x" },
      ],
      snapThresholdRatio: 0.1,
    });
    const input = wrapper.get("input[type=range]");

    await input.trigger("pointerdown");
    await input.setValue("38");
    await input.trigger("change");
    await input.trigger("pointerup");

    expect(wrapper.emitted("update:modelValue")).toEqual([[38], [40]]);
  });

  it("does not magnetically snap keyboard changes", async () => {
    const wrapper = make({
      modelValue: 30,
      step: 2,
      marks: [
        { value: 20, label: "1x" },
        { value: 40, label: "2x" },
      ],
      snapThresholdRatio: 0.1,
    });
    const input = wrapper.get("input[type=range]");

    await input.setValue("38");
    await input.trigger("change");

    expect(wrapper.emitted("update:modelValue")).toEqual([[38]]);
  });
});
