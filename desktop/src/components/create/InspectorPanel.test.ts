import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import InspectorPanel from "./InspectorPanel.vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
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
  it("disables ineffective distilled guidance and re-enables a guided recipe", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-distilled:fp8";
    form.guidance = 6;
    const wrapper = mount(InspectorPanel, { props: { form } });
    const guidance = () =>
      wrapper.findAllComponents(SliderRow).find((row) => row.props("label") === "Prompt strength")!;
    expect(guidance().props("disabled")).toBe(true);
    expect(guidance().props("modelValue")).toBe(1);
    expect(wrapper.get("[data-test='fixed-guidance-hint']").text()).toContain("fixes CFG at 1.0");
    form.pipeline = "two-stage";
    await flushPromises();
    expect(guidance().props("disabled")).toBe(false);
  });

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

  it("shows duration in seconds for the selected one-shot video model", async () => {
    const model = {
      name: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      default_frames: 97,
      default_fps: 24,
      max_runtime_seconds: 20,
      max_frames_absolute: 604,
      frame_step: 8,
    } as ModelEntry;
    useModelStore().all = [model];
    const form = formFor("ltx2");
    form.model = model.name;
    form.frames = 97;
    form.fps = 24;
    const wrapper = mount(InspectorPanel, { props: { form } });
    const duration = wrapper.getComponent(VideoDurationSlider);
    expect(duration.text()).toContain("4.0s");
    duration.vm.$emit("update:frames", 241);
    await flushPromises();
    expect(form.frames).toBe(241);
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
  it("hides aspect ratios the selected wan checkpoint does not support", () => {
    const model = {
      name: "wan22-i2v-a14b:q5",
      family: "wan",
      downloaded: true,
      recommended_dimensions: [
        { width: 832, height: 480 },
        { width: 480, height: 832 },
      ],
      dimension_alignment: 16,
      max_pixels: 1280 * 720,
    } as ModelEntry;
    useModelStore().all = [model];
    const form = formFor("wan");
    form.model = model.name;
    form.width = 480;
    form.height = 832;

    const wrapper = mount(InspectorPanel, { props: { form } });

    expect(wrapper.getComponent(ShapePicker).props("options")).toEqual([
      expect.objectContaining({ id: "26:15", label: "26:15" }),
      expect.objectContaining({ id: "15:26", label: "15:26" }),
    ]);
  });

  it("exposes and applies Z-Image's exact 16:9 and 9:16 buckets", async () => {
    const model = {
      name: "z-image-turbo:q4",
      family: "z-image",
      downloaded: true,
      recommended_dimensions: [
        { width: 1024, height: 1024 },
        { width: 1280, height: 720 },
        { width: 720, height: 1280 },
      ],
      dimension_alignment: 16,
      max_pixels: 1_800_000,
    } as ModelEntry;
    useModelStore().all = [model];
    const form = formFor("z-image");
    form.model = model.name;
    form.width = 1024;
    form.height = 1024;

    const wrapper = mount(InspectorPanel, { props: { form } });
    expect(wrapper.getComponent(ShapePicker).props("options")).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "16:9", label: "16:9" }),
        expect.objectContaining({ id: "9:16", label: "9:16" }),
      ]),
    );
    wrapper.getComponent(ShapePicker).vm.$emit("update:modelValue", "16:9");
    await flushPromises();
    expect([form.width, form.height]).toEqual([1280, 720]);
    wrapper.getComponent(ShapePicker).vm.$emit("update:modelValue", "9:16");
    await flushPromises();
    expect([form.width, form.height]).toEqual([720, 1280]);
  });

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
    wrapper
      .findComponent(ResolutionSelector)
      .vm.$emit("update:modelValue", (768 * 768) / 1_000_000);
    await flushPromises();
    expect(form.width * form.height).toBeLessThan(1024 * 1024);
    expect(form.width).toBe(form.height); // square ratio preserved
  });

  it("labels a model's only runnable bucket as Native", () => {
    const wrapper = mount(InspectorPanel, {
      props: { form: formFor("wuerstchen") },
    });
    expect(wrapper.findComponent(ResolutionSelector).props("options")).toEqual([
      expect.objectContaining({ sub: "Native" }),
    ]);
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

  it("shows only sequence-specific Advanced controls in Sequence output", async () => {
    const draft = useSequenceDraftStore();
    draft.hydrate();
    draft.output = "sequence";
    draft.ensureClips(97);
    const form = formFor("ltx2");
    form.model = "ltx-2-19b-distilled:fp8";
    const wrapper = mount(InspectorPanel, {
      props: {
        form,
        chainLimits: {
          model: form.model,
          supports_sequence: true,
          supports_audio: true,
          max_stages: 16,
          max_total_frames: 1552,
          frames_per_clip_cap: 97,
          frames_per_clip_recommended: 97,
          fade_frames_max: 24,
          transition_modes: ["smooth", "cut", "fade"],
          quantization_family: "fp8",
        },
      },
    });

    await wrapper.get('[data-test="open-advanced"]').trigger("click");

    expect(wrapper.find('[data-test="sequence-section-opening-image"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="sequence-section-negative"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="sequence-section-audio"]').exists()).toBe(true);
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
