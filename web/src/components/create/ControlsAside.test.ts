import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import ControlsAside from "./ControlsAside.vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import Stepper from "@ui/components/Stepper.vue";
import { dimsForMp } from "@ui/lib/resolution";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import type { GenerateFormState } from "../../types";

function baseForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  __testing__.resetForTest();
  const state = useGenerateForm().state.value;
  return { ...state, ...overrides };
}

function factory(overrides: Partial<GenerateFormState> = {}, family = "flux") {
  return mount(ControlsAside, {
    props: { modelValue: baseForm(overrides), family, advCount: 0 },
  });
}

describe("ControlsAside", () => {
  beforeEach(() => {
    localStorage.clear();
  });
  afterEach(() => __testing__.resetForTest());

  it("projects the current pixels onto Shape and Resolution", () => {
    const wrapper = factory({ width: 1024, height: 1024 });
    expect(wrapper.getComponent(ShapePicker).props("modelValue")).toBe(
      "square",
    );
    expect(wrapper.getComponent(ResolutionSelector).props("modelValue")).toBe(
      1,
    );
  });

  it("applies the projected dims when a shape is picked", async () => {
    const wrapper = factory({ width: 1024, height: 1024 });
    wrapper
      .getComponent(ShapePicker)
      .vm.$emit("update:modelValue", "landscape");
    await wrapper.vm.$nextTick();
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    const expected = dimsForMp(1, 4 / 3);
    expect(next.width).toBe(expected.width);
    expect(next.height).toBe(expected.height);
  });

  it("applies the projected dims when resolution changes", async () => {
    const wrapper = factory({ width: 1024, height: 1024 });
    wrapper.getComponent(ResolutionSelector).vm.$emit("update:modelValue", 2);
    await wrapper.vm.$nextTick();
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    const expected = dimsForMp(2, 1);
    expect(next.width).toBe(expected.width);
    expect(next.height).toBe(expected.height);
  });

  function seedButton(wrapper: ReturnType<typeof factory>, label: string) {
    return wrapper
      .findAll("[data-test='seed-seg'] button")
      .find((b) => b.text() === label)!;
  }

  it("maps the seed control to Random/Fixed and reveals the seed input when fixed", async () => {
    const wrapper = factory({ seedMode: "random", seed: null });
    expect(wrapper.find("[data-test='controls-seed']").exists()).toBe(false);

    await seedButton(wrapper, "Fixed").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.seedMode).toBe("static");
  });

  it("keeps an increment seed under the Fixed segment", () => {
    const wrapper = factory({ seedMode: "increment", seed: 100 });
    // Fixed covers both static and increment, so the seed input is shown.
    expect(wrapper.find("[data-test='controls-seed']").exists()).toBe(true);
  });

  it("steps the batch size", async () => {
    const wrapper = factory({ batchSize: 1 });
    wrapper.getComponent(Stepper).vm.$emit("update:modelValue", 3);
    await wrapper.vm.$nextTick();
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.batchSize).toBe(3);
  });

  it("shows the advanced badge and opens the drawer", async () => {
    const wrapper = mount(ControlsAside, {
      props: { modelValue: baseForm(), family: "flux", advCount: 3 },
    });
    expect(wrapper.get("[data-test='adv-badge']").text()).toContain("3");
    await wrapper.get("[data-test='open-advanced']").trigger("click");
    expect(wrapper.emitted("open-advanced")).toHaveLength(1);
  });
});
