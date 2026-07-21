import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import InspectorPanel from "./InspectorPanel.vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import Stepper from "@ui/components/Stepper.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import { aspectIdFor } from "../../lib/resolutions";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useModelStore } from "../../stores/models";
import type { ModelEntry } from "../../lib/api/types";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("../../lib/ipc", () => ({ ipc: {}, inTauri: () => false }));

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => (document.body.innerHTML = ""));

function formFor(family: string): GenerateForm {
  return reactive({ ...newGenerateForm(), family });
}

describe("InspectorPanel — layout", () => {
  it("renders every primary generation control", () => {
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    expect(wrapper.findComponent(ShapePicker).exists()).toBe(true);
    expect(wrapper.findComponent(ResolutionSelector).exists()).toBe(true);
    // Detail + Prompt strength sliders.
    expect(wrapper.findAllComponents(SliderRow)).toHaveLength(2);
    expect(wrapper.findComponent(Stepper).exists()).toBe(true);
    expect(wrapper.find('[data-test="seed-mode-random"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="open-advanced"]').exists()).toBe(true);
  });
});

describe("InspectorPanel — shape + resolution projection", () => {
  it("applies a picked shape to the form dimensions at the current budget", async () => {
    const form = formFor("flux");
    form.width = 1024;
    form.height = 1024;
    const wrapper = mount(InspectorPanel, { props: { form } });
    expect(wrapper.findComponent(ShapePicker).props("modelValue")).toBe("square");
    wrapper.findComponent(ShapePicker).vm.$emit("update:modelValue", "wide");
    await flushPromises();
    expect(aspectIdFor(form.width, form.height)).toBe("wide");
    expect(form.width % 16).toBe(0);
    expect(form.height % 16).toBe(0);
  });

  it("reprojects the form onto a new megapixel budget", async () => {
    const form = formFor("flux");
    form.width = 1024;
    form.height = 1024;
    const wrapper = mount(InspectorPanel, { props: { form } });
    wrapper.findComponent(ResolutionSelector).vm.$emit("update:modelValue", 0.5);
    await flushPromises();
    expect(form.width * form.height).toBeLessThan(1024 * 1024);
    expect(form.width).toBe(form.height); // square ratio preserved
  });
});

describe("InspectorPanel — batch", () => {
  it("steps the batch size through the Stepper", async () => {
    const form = formFor("flux");
    const wrapper = mount(InspectorPanel, { props: { form } });
    wrapper.findComponent(Stepper).vm.$emit("update:modelValue", 3);
    await flushPromises();
    expect(form.batchSize).toBe(3);
    expect(wrapper.findComponent(Stepper).props("max")).toBe(8);
  });

  it("locks the batch to one for edit models", () => {
    const wrapper = mount(InspectorPanel, { props: { form: formFor("qwen-image-edit") } });
    expect(wrapper.findComponent(Stepper).props("max")).toBe(1);
    expect(wrapper.text()).toContain("Locked to 1");
  });
});

describe("InspectorPanel — seed mode", () => {
  it("starts Random with an empty seed and hides the value field", () => {
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    expect(wrapper.get('[data-test="seed-mode-random"]').attributes("aria-pressed")).toBe("true");
    expect(wrapper.find('[data-test="seed-input"]').exists()).toBe(false);
    expect(wrapper.text()).toContain("New seed every print");
  });

  it("switching to Fixed fills the field (last seed preferred) and locks it", async () => {
    const form = formFor("flux");
    const wrapper = mount(InspectorPanel, { props: { form, lastSeed: 1234 } });
    await wrapper.get('[data-test="seed-mode-fixed"]').trigger("click");
    expect(form.seed).toBe("1234");
    expect(wrapper.find('[data-test="seed-input"]').exists()).toBe(true);
  });

  it("lock-last jumps straight from Random to that seed", async () => {
    const form = formFor("flux");
    const wrapper = mount(InspectorPanel, { props: { form, lastSeed: 77 } });
    await wrapper.get('[data-test="lock-last-seed"]').trigger("click");
    expect(form.seed).toBe("77");
    expect(wrapper.get('[data-test="seed-mode-fixed"]').attributes("aria-pressed")).toBe("true");
  });

  it("clearing the field in Fixed mode keeps the input mounted with a hint", async () => {
    const form = formFor("flux");
    form.seed = "42";
    const wrapper = mount(InspectorPanel, { props: { form } });
    await wrapper.get('[data-test="seed-input"]').setValue("");
    expect(wrapper.find('[data-test="seed-input"]').exists()).toBe(true);
    expect(wrapper.get('[data-test="seed-mode-fixed"]').attributes("aria-pressed")).toBe("true");
    expect(wrapper.get('[data-test="seed-hint"]').text()).toContain("random seed will be used");
  });

  it("non-numeric seed text warns instead of silently generating random", async () => {
    const form = formFor("flux");
    form.seed = "42";
    const wrapper = mount(InspectorPanel, { props: { form } });
    await wrapper.get('[data-test="seed-input"]').setValue("banana");
    expect(wrapper.get('[data-test="seed-hint"]').text()).toContain("Not a number");
    await wrapper.get('[data-test="seed-input"]').setValue("1234");
    expect(wrapper.find('[data-test="seed-hint"]').exists()).toBe(false);
  });
});

describe("InspectorPanel — advanced", () => {
  it("opens the drawer and badges the active advanced count", async () => {
    const form = formFor("sdxl");
    form.negativePrompt = "blurry";
    const wrapper = mount(InspectorPanel, { props: { form } });
    expect(wrapper.findComponent(BadgePill).text()).toContain("1 on");
    await wrapper.get('[data-test="open-advanced"]').trigger("click");
    expect(wrapper.emitted("open-advanced")).toHaveLength(1);
  });
});

describe("InspectorPanel — model picker", () => {
  const model: ModelEntry = {
    name: "flux-dev:q8",
    family: "flux",
    downloaded: true,
    default_width: 1024,
    default_height: 1024,
    default_steps: 20,
    default_guidance: 4.5,
  } as ModelEntry;

  it("opens the picker and applies a chosen model to the shared form", async () => {
    useModelStore().all = [model];
    const form = useGenerateFormStore().form;
    const wrapper = mount(InspectorPanel, { props: { form }, attachTo: document.body });
    await wrapper.get('[data-test="selected-model-name"]').trigger("click");
    expect(wrapper.find('[data-test="model-option-name"]').exists()).toBe(true);
    await wrapper.get('[data-test="model-option-name"]').trigger("click");
    expect(form.model).toBe("flux-dev:q8");
  });
});
