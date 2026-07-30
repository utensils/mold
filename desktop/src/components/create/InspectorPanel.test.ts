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
import PanelResizeHandle from "../shell/PanelResizeHandle.vue";
import { aspectIdFor } from "../../lib/resolutions";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useModelStore } from "../../stores/models";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
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

  it("defaults wide enough for one ratio row and persists left-edge resizing", async () => {
    const prefs = useAppPrefsStore();
    const update = vi.spyOn(prefs, "update").mockResolvedValue();
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    const inspector = wrapper.get('[data-test="inspector-panel"]');
    const handle = wrapper.getComponent(PanelResizeHandle);

    expect(inspector.attributes("style")).toContain("width: 340px");
    expect(handle.props("label")).toBe("Resize generation settings");

    handle.vm.$emit("resize", -40);
    await flushPromises();
    expect(inspector.attributes("style")).toContain("width: 380px");

    handle.vm.$emit("commit");
    await flushPromises();
    expect(update).toHaveBeenCalledWith({ generateParamsWidth: 380 });

    handle.vm.$emit("reset");
    await flushPromises();
    expect(update).toHaveBeenCalledWith({ generateParamsWidth: null });
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
    expect(wrapper.findComponent(Stepper).props("max")).toBe(10_000);
    expect(wrapper.findComponent(Stepper).props("editable")).toBe(true);
  });

  it("accepts a directly entered large positive batch", async () => {
    const form = formFor("flux");
    const wrapper = mount(InspectorPanel, { props: { form } });
    const input = wrapper.get('input[aria-label="Batch size"]');
    await input.setValue("1000");
    await input.trigger("change");
    await flushPromises();
    expect(form.batchSize).toBe(1000);
  });

  it("does not overwrite an uncommitted direct entry on an arrow key", async () => {
    const form = formFor("flux");
    const wrapper = mount(InspectorPanel, { props: { form } });
    const input = wrapper.get('input[aria-label="Batch size"]');
    (input.element as HTMLInputElement).value = "120";
    await input.trigger("keydown", { key: "ArrowUp" });
    expect(form.batchSize).toBe(1);
    expect((input.element as HTMLInputElement).value).toBe("120");
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
  it("keeps the simplified inspector by default and expands Advanced inline", async () => {
    const form = formFor("sdxl");
    form.negativePrompt = "blurry";
    const wrapper = mount(InspectorPanel, { props: { form } });
    expect(wrapper.findComponent(BadgePill).text()).toContain("1 on");
    expect(wrapper.get('[data-test="open-advanced"]').attributes("aria-expanded")).toBe("false");
    expect(wrapper.find('[data-test="inline-advanced"]').exists()).toBe(false);

    await wrapper.get('[data-test="open-advanced"]').trigger("click");

    expect(wrapper.get('[data-test="open-advanced"]').attributes("aria-expanded")).toBe("true");
    expect(wrapper.find('[data-test="inline-advanced"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="seed-mode-random"]').exists()).toBe(true);

    await wrapper.get('[data-test="open-advanced"]').trigger("click");
    expect(wrapper.get('[data-test="open-advanced"]').attributes("aria-expanded")).toBe("false");
    expect(wrapper.find('[data-test="inline-advanced"]').exists()).toBe(false);
  });
});

describe("InspectorPanel — reset to model defaults", () => {
  const model: ModelEntry = {
    name: "sdxl:base",
    family: "sdxl",
    downloaded: true,
    default_width: 1024,
    default_height: 768,
    default_steps: 30,
    default_guidance: 7,
  } as ModelEntry;

  it("offers the reset without opening Advanced", () => {
    const wrapper = mount(InspectorPanel, { props: { form: formFor("sdxl") } });
    const reset = wrapper.get('[data-test="settings-reset"]');
    expect(wrapper.find('[data-test="inline-advanced"]').exists()).toBe(false);
    expect(reset.attributes("aria-label")).toBe("Reset settings to model defaults");
  });

  it("restores the model's defaults while preserving prompt, model, and batch", async () => {
    useModelStore().all = [model];
    const form = useGenerateFormStore().form;
    form.model = model.name;
    form.family = model.family;
    form.prompt = "a lighthouse at dusk";
    form.batchSize = 4;
    form.negativePrompt = "blurry";
    form.steps = 12;
    form.seed = "1234";
    const wrapper = mount(InspectorPanel, { props: { form }, attachTo: document.body });

    await wrapper.get('[data-test="settings-reset"]').trigger("click");

    expect(form.prompt).toBe("a lighthouse at dusk");
    expect(form.model).toBe("sdxl:base");
    expect(form.batchSize).toBe(4);
    expect(form.negativePrompt).toBe("");
    expect(form.seed).toBe("");
    expect(form.steps).toBe(30);
    expect(form.width).toBe(1024);
    expect(form.height).toBe(768);
  });
});

describe("InspectorPanel — output", () => {
  const videoModel: ModelEntry = {
    name: "ltx-video",
    family: "ltx-video",
    downloaded: true,
    default_width: 1024,
    default_height: 576,
    default_steps: 25,
    default_guidance: 3,
  } as ModelEntry;
  const stillModel: ModelEntry = {
    name: "flux-dev:q8",
    family: "flux",
    downloaded: true,
    default_width: 1024,
    default_height: 1024,
    default_steps: 20,
    default_guidance: 4.5,
  } as ModelEntry;

  function outputSegments(wrapper: ReturnType<typeof mount>) {
    return wrapper.get("[data-test='output-mode']").findAll("button[role='radio']");
  }

  it("renders the Output card between Model and Shape with One shot active", () => {
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    const card = wrapper.get("[data-test='output-card']");
    expect(card.text()).toContain("Output");
    const segments = outputSegments(wrapper);
    expect(segments.map((b) => b.text())).toEqual(["One shot", "Sequence"]);
    expect(segments[0]!.attributes("aria-checked")).toBe("true");
    // Card order: Model precedes the Output card, which precedes Shape.
    const html = wrapper.html();
    expect(html.indexOf("output-card")).toBeGreaterThan(html.indexOf("selected-model-name"));
    expect(html.indexOf("output-card")).toBeLessThan(html.indexOf(">Shape<"));
  });

  it("switching to Sequence keeps prompts separate, remembers + swaps the model, and locks batch", async () => {
    useModelStore().all = [stillModel, videoModel];
    const form = useGenerateFormStore().form;
    form.model = stillModel.name;
    form.family = stillModel.family;
    form.prompt = "a cat at dusk";
    const wrapper = mount(InspectorPanel, { props: { form }, attachTo: document.body });

    await outputSegments(wrapper)[1]!.trigger("click");
    await flushPromises();

    const draft = useSequenceDraftStore();
    expect(draft.output).toBe("sequence");
    expect(draft.clips).toHaveLength(2);
    expect(draft.clips[0]!.prompt).toBe("");
    // A non-capable model is remembered and swapped for the first capable one.
    expect(draft.lastSingleModel).toBe("flux-dev:q8");
    expect(form.model).toBe("ltx-video");
    // Batch locks to one timeline; the switch-back caption appears.
    expect(wrapper.text()).toContain("a sequence renders one timeline");
    expect(wrapper.text()).toContain("one-shot and sequence prompts stay separate");
  });

  it("switching back restores the remembered single model without leaking clip 1's prompt", async () => {
    useModelStore().all = [stillModel, videoModel];
    const form = useGenerateFormStore().form;
    form.model = videoModel.name;
    form.family = videoModel.family;
    form.prompt = "the one-shot prompt";
    const draft = useSequenceDraftStore();
    draft.output = "sequence";
    draft.ensureClips(25);
    draft.clips[0]!.prompt = "the opening clip";
    draft.lastSingleModel = "flux-dev:q8";
    const wrapper = mount(InspectorPanel, { props: { form }, attachTo: document.body });

    await outputSegments(wrapper)[0]!.trigger("click");
    await flushPromises();

    expect(draft.output).toBe("single");
    expect(form.model).toBe("flux-dev:q8");
    expect(form.prompt).toBe("the one-shot prompt");
    expect(draft.lastSingleModel).toBeNull();
    // Clips are parked, never erased.
    expect(draft.clips).toHaveLength(2);
  });

  it("filters the picker to sequence-capable models while in sequence mode", async () => {
    useModelStore().all = [stillModel, videoModel];
    const form = useGenerateFormStore().form;
    form.model = videoModel.name;
    useSequenceDraftStore().output = "sequence";
    const wrapper = mount(InspectorPanel, { props: { form }, attachTo: document.body });
    await wrapper.get('[data-test="selected-model-name"]').trigger("click");
    const options = wrapper.findAll('[data-test="model-option-name"]');
    expect(options.map((o) => o.text())).toEqual(["ltx-video"]);
  });

  it("surfaces a frame-rate stepper and hides lock-last-seed in sequence mode", async () => {
    useSequenceDraftStore().output = "sequence";
    const form = formFor("ltx-video");
    const wrapper = mount(InspectorPanel, { props: { form, lastSeed: 77 } });
    expect(wrapper.find('[data-test="sequence-fps"]').exists()).toBe(true);
    expect(wrapper.text()).toContain("Frame rate");
    expect(wrapper.find('[data-test="lock-last-seed"]').exists()).toBe(false);
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

  it("shows a human-readable catalog name while preserving the runnable id", async () => {
    const catalogModel = {
      ...model,
      name: "cv:23423432",
      family: "sdxl",
      description: "RealVisXL V5.0 by SG161222",
    };
    useModelStore().all = [catalogModel];
    const form = useGenerateFormStore().form;
    form.model = catalogModel.name;
    const wrapper = mount(InspectorPanel, { props: { form }, attachTo: document.body });

    expect(wrapper.get('[data-test="selected-model-name"]').text()).toBe(
      "RealVisXL V5.0 by SG161222",
    );
    await wrapper.get('[data-test="selected-model-name"]').trigger("click");
    const option = wrapper.get('[data-test="model-option-name"]');
    expect(option.text()).toBe("RealVisXL V5.0 by SG161222");
    await option.trigger("click");
    expect(form.model).toBe("cv:23423432");
  });
});
