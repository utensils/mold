import { mount, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SeamPill from "@ui/components/SeamPill.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";
import type { ModelEntry } from "../lib/api/types";
import MobileSeamSheet from "./MobileSeamSheet.vue";
import MobileAdvancedSheet from "./MobileAdvancedSheet.vue";
import MobileSequenceComposer from "./MobileSequenceComposer.vue";
import { validateChain } from "@studio/api/chains";
import { newGenerateForm } from "../lib/generateForm";

vi.mock("@studio/api/chains", () => ({
  validateChain: vi.fn(),
}));

const validateChainMock = vi.mocked(validateChain);

installMemoryLocalStorage();

const ltx2: ModelEntry = {
  name: "ltx2-distilled:q8",
  family: "ltx2",
  size_gb: 20,
  is_loaded: false,
  hf_repo: "example/ltx2",
  default_steps: 8,
  default_guidance: 3,
  default_width: 768,
  default_height: 512,
  description: "Video model",
  downloaded: true,
  default_frames: 97,
};

const ltxVideo: ModelEntry = { ...ltx2, name: "ltx-video:bf16", family: "ltx-video" };
const cameraControls = [
  {
    id: "dolly-in",
    label: "Dolly in",
    size_bytes: 327_309_208,
    installed: false,
    download_model: "ltx2-camera-control-dolly-in-19b",
    download_repo: "Lightricks/camera",
    download_filename: "dolly-in.safetensors",
    download_sha256: "a".repeat(64),
  },
];

const limits: ChainLimits = {
  model: ltx2.name,
  frames_per_clip_cap: 97,
  frames_per_clip_recommended: 97,
  max_stages: 4,
  max_total_frames: 777,
  fade_frames_max: 24,
  transition_modes: ["smooth", "cut", "fade"],
  quantization_family: "q8",
  supports_audio: false,
  supports_sequence: true,
};

const shared = {
  model: ltx2.name,
  family: ltx2.family,
  width: 768,
  height: 512,
  fps: 24,
  steps: 8,
  guidance: 3,
  strength: 1,
  seed: "",
};

let wrapper: VueWrapper | null = null;

function mountComposer(props: Record<string, unknown> = {}): VueWrapper {
  wrapper = mount(MobileSequenceComposer, {
    props: {
      selectedModel: ltx2,
      form: newGenerateForm(),
      chainLimits: limits,
      target: { baseUrl: "http://studio:7680", apiKey: "secret" },
      shared,
      fps: 24,
      submitting: false,
      error: "",
      cameraControls,
      cameraControlsLoaded: true,
      ...props,
    },
  });
  return wrapper;
}

function clipCards() {
  return wrapper!.findAll("[data-test='mobile-sequence-clip']");
}

beforeEach(() => {
  validateChainMock.mockReset();
  localStorage.clear();
  setActivePinia(createPinia());
  const draft = useSequenceDraftStore();
  draft.hydrate();
  draft.ensureClips(97);
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

describe("MobileSequenceComposer clips", () => {
  it("preserves but disables clip negatives for distilled LTX", async () => {
    const form = newGenerateForm();
    form.family = "ltx2";
    form.model = "ltx-2.3-22b-distilled:fp8";
    useSequenceDraftStore().clips[0]!.negativePrompt = "flicker";
    mountComposer({ form });
    await wrapper!.get("[data-test='mobile-sequence-open-advanced']").trigger("click");
    const input = wrapper!.get("[data-test='mobile-sequence-advanced-negative'] input");
    expect(input.attributes("disabled")).toBeDefined();
    expect(useSequenceDraftStore().clips[0]!.negativePrompt).toBe("flicker");
    expect(
      wrapper!.get("[data-test='mobile-sequence-negative-unavailable-hint']").text(),
    ).toContain("does not use negative-prompt guidance");
  });

  it("stores and resets camera motion on the active clip", async () => {
    const draft = useSequenceDraftStore();
    mountComposer();
    await wrapper!.get("[data-test='mobile-sequence-open-advanced']").trigger("click");

    const select = wrapper!.get("[data-test='mobile-sequence-camera-motion']");
    expect(select.text()).toContain("downloads on first use");
    await select.setValue("dolly-in");
    expect(draft.clips[0]?.cameraControl).toBe("dolly-in");
    wrapper!.getComponent(MobileAdvancedSheet).vm.$emit("reset");
    expect(draft.clips[0]?.cameraControl).toBeNull();
  });

  it("locks source controls while a sequence is submitting", async () => {
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    mountComposer({ submitting: true });
    await wrapper!.get("[data-test='mobile-sequence-open-advanced']").trigger("click");
    expect(
      (wrapper!.get("[data-test='mobile-sequence-source-strength']").element as HTMLInputElement)
        .disabled,
    ).toBe(true);
    expect(
      (wrapper!.get("[data-test='mobile-sequence-source-fit']").element as HTMLSelectElement)
        .disabled,
    ).toBe(true);
  });

  it("renders the shared draft's clips instead of private component state", async () => {
    const draft = useSequenceDraftStore();
    draft.clips[0]!.prompt = "a paper boat";
    mountComposer();

    const prompts = wrapper!.findAll("[data-test='mobile-sequence-clip'] textarea");
    expect(prompts).toHaveLength(2);
    expect((prompts[0]!.element as HTMLTextAreaElement).value).toBe("a paper boat");

    await prompts[1]!.setValue("fireflies gather");
    // The prompt survives a remount because the draft owns it, which is the
    // whole point: the old composer's reactive form died on every tab switch.
    expect(draft.clips[1]?.prompt).toBe("fireflies gather");
    wrapper!.unmount();
    mountComposer();
    expect(
      (
        wrapper!.findAll("[data-test='mobile-sequence-clip'] textarea")[1]!
          .element as HTMLTextAreaElement
      ).value,
    ).toBe("fireflies gather");
  });

  it("names the opening clip and keeps Remove above the two-clip floor", async () => {
    const draft = useSequenceDraftStore();
    mountComposer();
    expect(clipCards()[0]!.text()).toContain("Opening clip");
    expect(clipCards()[1]!.text()).toContain("Clip 2");
    expect(wrapper!.findAll("[data-test='mobile-sequence-remove']")).toHaveLength(0);

    await wrapper!.get("[data-test='mobile-sequence-add']").trigger("click");
    expect(draft.clips).toHaveLength(3);
    const removes = wrapper!.findAll("[data-test='mobile-sequence-remove']");
    expect(removes).toHaveLength(3);

    await removes[2]!.trigger("click");
    expect(draft.clips).toHaveLength(2);
    expect(wrapper!.findAll("[data-test='mobile-sequence-remove']")).toHaveLength(0);
  });

  it("sizes a new clip from the model's own default rather than a hardcoded 25", async () => {
    const draft = useSequenceDraftStore();
    mountComposer();
    await wrapper!.get("[data-test='mobile-sequence-add']").trigger("click");
    expect(draft.clips[2]?.frames).toBe(97);

    // A model whose server default is smaller must win over the cap.
    wrapper!.unmount();
    mountComposer({
      selectedModel: { ...ltxVideo, default_frames: 25 },
      chainLimits: { ...limits, frames_per_clip_cap: 97, frames_per_clip_recommended: 97 },
    });
    await wrapper!.get("[data-test='mobile-sequence-add']").trigger("click");
    expect(draft.clips[3]?.frames).toBe(25);
  });

  it("stops adding clips at the server's max_stages", async () => {
    mountComposer({ chainLimits: { ...limits, max_stages: 3 } });
    await wrapper!.get("[data-test='mobile-sequence-add']").trigger("click");
    expect(useSequenceDraftStore().clips).toHaveLength(3);
    expect(
      (wrapper!.get("[data-test='mobile-sequence-add']").element as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("offers only clip durations strictly longer than the motion tail", () => {
    const draft = useSequenceDraftStore();
    for (const clip of draft.clips) clip.frames = 25;
    mountComposer({ chainLimits: { ...limits, frames_per_clip_cap: 41 } });
    const options = clipCards()[0]!
      .findAll("[data-test='mobile-sequence-frames'] option")
      .map((option) => Number((option.element as HTMLOptionElement).value));
    // LTX-2's 17-frame tail rules out 9 and 17.
    expect(options).toEqual([25, 33, 41]);

    // LTX-Video carries no tail, so the whole 8n+1 grid is available.
    wrapper!.unmount();
    mountComposer({
      selectedModel: ltxVideo,
      chainLimits: { ...limits, frames_per_clip_cap: 41 },
    });
    expect(
      clipCards()[0]!
        .findAll("[data-test='mobile-sequence-frames'] option")
        .map((option) => Number((option.element as HTMLOptionElement).value)),
    ).toEqual([9, 17, 25, 33, 41]);
  });

  it("keeps an off-grid loaded duration visible rather than re-snapping it", () => {
    const draft = useSequenceDraftStore();
    draft.clips[0]!.frames = 60;
    mountComposer({ chainLimits: { ...limits, frames_per_clip_cap: 41 } });
    expect(
      clipCards()[0]!
        .findAll("[data-test='mobile-sequence-frames'] option")
        .map((option) => Number((option.element as HTMLOptionElement).value)),
    ).toEqual([25, 33, 41, 60]);
  });

  it("drops the in-card transition radiogroup for a seam between cards", () => {
    mountComposer();
    expect(wrapper!.find(".mobile-sequence-transitions").exists()).toBe(false);
    // One seam for two clips — seams sit BETWEEN cards, never inside one.
    const seams = wrapper!.findAllComponents(SeamPill);
    expect(seams).toHaveLength(1);
    expect(seams[0]!.props("large")).toBe(true);
    expect(seams[0]!.props("motionTail")).toBe(17);
  });
});

describe("MobileSequenceComposer seam sheet", () => {
  it("opens the seam sheet on the pill and writes the transition to the draft", async () => {
    const draft = useSequenceDraftStore();
    mountComposer();
    expect(wrapper!.getComponent(MobileSeamSheet).props("open")).toBe(false);

    await wrapper!.getComponent(SeamPill).trigger("click");
    const sheet = wrapper!.getComponent(MobileSeamSheet);
    expect(sheet.props("open")).toBe(true);
    expect(sheet.props("fromLabel")).toBe("opening");
    expect(sheet.props("toLabel")).toBe("clip 2");
    expect(wrapper!.getComponent(SeamPill).props("active")).toBe(true);

    sheet.vm.$emit("update:transition", "fade");
    await wrapper!.vm.$nextTick();
    expect(draft.clips[1]?.transition).toBe("fade");

    sheet.vm.$emit("update:fadeFrames", 12);
    await wrapper!.vm.$nextTick();
    expect(draft.clips[1]?.fadeFrames).toBe(12);
    expect(wrapper!.getComponent(SeamPill).props("fadeFrames")).toBe(12);

    sheet.vm.$emit("close");
    await wrapper!.vm.$nextTick();
    expect(wrapper!.getComponent(MobileSeamSheet).props("open")).toBe(false);
    expect(wrapper!.getComponent(SeamPill).props("active")).toBe(false);
  });

  it("hands the sheet the server's fade cap, falling back to 32", async () => {
    mountComposer();
    await wrapper!.getComponent(SeamPill).trigger("click");
    expect(wrapper!.getComponent(MobileSeamSheet).props("fadeFramesMax")).toBe(24);

    wrapper!.unmount();
    mountComposer({ chainLimits: null });
    await wrapper!.getComponent(SeamPill).trigger("click");
    expect(wrapper!.getComponent(MobileSeamSheet).props("fadeFramesMax")).toBe(32);
  });
});

describe("MobileSequenceComposer guardrails", () => {
  it("discards an in-flight plan when same-named opening-image bytes change", async () => {
    let resolveValidation!: (value: Awaited<ReturnType<typeof validateChain>>) => void;
    validateChainMock.mockReturnValue(
      new Promise((resolve) => {
        resolveValidation = resolve;
      }),
    );
    const draft = useSequenceDraftStore();
    draft.clips[0]!.prompt = "a paper boat";
    draft.clips[1]!.prompt = "fireflies gather";
    draft.openingImage = { filename: "opening.png", base64: "AAAA" };
    mountComposer();

    await wrapper!.get("[data-test='mobile-sequence-validate']").trigger("click");
    draft.openingImage = { filename: "opening.png", base64: "BBBB" };
    await wrapper!.vm.$nextTick();
    resolveValidation({
      model: ltx2.name,
      width: 768,
      height: 512,
      fps: 24,
      motion_tail_frames: 17,
      stage_count: 2,
      estimated_total_frames: 177,
      estimated_duration_ms: 7_375,
      stages: [],
      warnings: [],
      vram_estimate: null,
    });
    await wrapper!.vm.$nextTick();
    expect(wrapper!.find("[data-test='mobile-sequence-validation-plan']").exists()).toBe(false);
  });

  it("validates on the selected host and clears the rendered plan after edits", async () => {
    validateChainMock.mockResolvedValue({
      model: ltx2.name,
      width: 768,
      height: 512,
      fps: 24,
      motion_tail_frames: 17,
      stage_count: 2,
      estimated_total_frames: 177,
      estimated_duration_ms: 7_375,
      stages: [
        {
          prompt: "a paper boat",
          frames: 97,
          output_frames: 97,
          transition: "smooth",
          fade_frames: null,
          has_source_image: true,
          has_negative_prompt: false,
        },
        {
          prompt: "fireflies gather",
          frames: 97,
          output_frames: 80,
          transition: "smooth",
          fade_frames: null,
          has_source_image: true,
          has_negative_prompt: false,
        },
      ],
      warnings: [],
      vram_estimate: null,
    });
    const draft = useSequenceDraftStore();
    draft.clips[0]!.prompt = "a paper boat";
    draft.clips[1]!.prompt = "fireflies gather";
    const target = { baseUrl: "http://studio:7680", apiKey: "secret" };
    mountComposer({ target });

    await wrapper!.get("[data-test='mobile-sequence-validate']").trigger("click");
    await wrapper!.vm.$nextTick();

    expect(validateChainMock).toHaveBeenCalledWith(
      expect.objectContaining({
        model: ltx2.name,
        width: 768,
        height: 512,
        stages: [
          expect.objectContaining({ prompt: "a paper boat" }),
          expect.objectContaining({ prompt: "fireflies gather" }),
        ],
      }),
      target,
    );
    expect(wrapper!.get("[data-test='mobile-sequence-validation-plan']").text()).toContain(
      "Validated · 2 clips · 177f · 7.4s",
    );
    const planText = wrapper!
      .get("[data-test='mobile-sequence-validation-plan']")
      .text()
      .replace(/\s+/g, " ");
    expect(planText).toContain("Clip 1 · 97f in / 97f out · Smooth · Opening image");
    expect(planText).toContain("Clip 2 · 97f in / 80f out · Smooth · Source image");
    expect(wrapper!.emitted("submit")).toBeUndefined();

    await wrapper!
      .findAll("[data-test='mobile-sequence-clip'] textarea")[0]!
      .setValue("edited opening");
    expect(wrapper!.find("[data-test='mobile-sequence-validation-plan']").exists()).toBe(false);
  });

  it("blocks Generate until every clip is described", async () => {
    const draft = useSequenceDraftStore();
    mountComposer();
    const generate = () => wrapper!.get("[data-test='mobile-generate-sequence']");
    expect((generate().element as HTMLButtonElement).disabled).toBe(true);
    expect(wrapper!.get("[data-test='mobile-sequence-error']").text()).toContain("clip 1");

    draft.clips[0]!.prompt = "a paper boat";
    draft.clips[1]!.prompt = "fireflies gather";
    await wrapper!.vm.$nextTick();
    expect((generate().element as HTMLButtonElement).disabled).toBe(false);

    await generate().trigger("click");
    expect(wrapper!.emitted("submit")).toHaveLength(1);
  });

  it("shows the server's reason when the model cannot render a sequence", () => {
    const draft = useSequenceDraftStore();
    draft.clips[0]!.prompt = "a";
    draft.clips[1]!.prompt = "b";
    mountComposer({
      chainLimits: {
        ...limits,
        supports_sequence: false,
        sequence_unsupported_reason: "Two-stage LTX-2 dev checkpoints can't chain.",
      },
    });

    expect(wrapper!.get("[data-test='mobile-sequence-error']").text()).toBe(
      "Two-stage LTX-2 dev checkpoints can't chain.",
    );
    expect(
      (wrapper!.get("[data-test='mobile-generate-sequence']").element as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(wrapper!.emitted("submit")).toBeUndefined();
  });

  it("renders the host's failure through the friendly sequence wording", () => {
    const draft = useSequenceDraftStore();
    draft.clips[0]!.prompt = "a";
    draft.clips[1]!.prompt = "b";
    mountComposer({ error: "CUDA_ERROR_OUT_OF_MEMORY while allocating latents" });
    expect(wrapper!.get("[data-test='mobile-sequence-submit-error']").text()).toContain(
      "needs more GPU memory",
    );
  });

  it("surfaces audio only when the routed checkpoint bundles it", async () => {
    mountComposer();
    expect(wrapper!.find("[data-test='mobile-sequence-audio']").exists()).toBe(false);

    wrapper!.unmount();
    mountComposer({ chainLimits: { ...limits, supports_audio: true } });
    const audio = wrapper!.get("[data-test='mobile-sequence-audio'] input");
    await audio.setValue(true);
    expect(useSequenceDraftStore().enableAudio).toBe(true);
  });

  it("lends the host form's shared params a disclosure between the clips and Generate", () => {
    wrapper = mount(MobileSequenceComposer, {
      props: {
        selectedModel: ltx2,
        form: newGenerateForm(),
        chainLimits: limits,
        target: { baseUrl: "http://studio:7680", apiKey: "secret" },
        shared,
        fps: 24,
        submitting: false,
        error: "",
        settingsSummary: "768×512 · 24fps",
      },
      slots: { settings: '<div data-test="host-params">shared</div>' },
    });
    const settings = wrapper.get("[data-test='mobile-sequence-settings']");
    expect(settings.text()).toContain("768×512 · 24fps");
    expect(settings.find("[data-test='host-params']").exists()).toBe(true);
  });

  it("omits the settings disclosure when the host supplies no params", () => {
    mountComposer();
    expect(wrapper!.find("[data-test='mobile-sequence-settings']").exists()).toBe(false);
  });

  it("attaches a local-or-gallery image only to the opening clip", async () => {
    const draft = useSequenceDraftStore();
    mountComposer();
    expect(wrapper!.findAll("[data-test='mobile-sequence-source-pick']")).toHaveLength(1);

    await wrapper!.get("[data-test='mobile-sequence-source-pick']").trigger("click");
    const picker = wrapper!.getComponent({ name: "MobileImagePickerSheet" });
    expect(picker.props("open")).toBe(true);
    picker.vm.$emit("pick", { filename: "opening.jpg", base64: "wire-bytes" });
    await wrapper!.vm.$nextTick();

    expect(draft.openingImage).toEqual({
      filename: "opening.jpg",
      base64: "wire-bytes",
    });
    expect(wrapper!.get("[data-test='mobile-sequence-source-preview']").attributes("src")).toBe(
      "data:image/jpeg;base64,wire-bytes",
    );
  });

  it("exposes strength and fit controls after an opening image is attached", async () => {
    const form = newGenerateForm();
    const draft = useSequenceDraftStore();
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    mountComposer({
      form,
      upscalers: [{ ...ltx2, name: "real-esrgan-x4plus:fp16", family: "upscaler" }],
    });
    await wrapper!.get("[data-test='mobile-sequence-open-advanced']").trigger("click");

    expect(
      (wrapper!.get("[data-test='mobile-sequence-source-fit']").element as HTMLSelectElement).value,
    ).toBe("crop-fill");
    await wrapper!.get("[data-test='mobile-sequence-source-strength']").setValue("0.6");
    expect(form.strength).toBe(0.6);
    await wrapper!.get("[data-test='mobile-sequence-source-fit']").setValue("upscale-then-fit");
    expect(form.sourceFit).toEqual({
      mode: "upscale-then-fit",
      upscalerModel: "real-esrgan-x4plus:fp16",
      fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
    });
  });

  it("summarizes the timeline at the form's own frame rate", () => {
    mountComposer({ fps: 12 });
    // Two 97-frame clips with LTX-2's 17-frame smooth tail: 97 + 80 = 177.
    const summary = wrapper!.get("[data-test='mobile-sequence-duration']").text();
    expect(summary).toContain("177f · 14.8s");
    expect(summary).toContain("14.8s");
  });
});

/**
 * Wan's clip grid is its own: the family's VAE compresses time by 4, so its
 * clips are `4k+1` and the LTX `8k+1` ladder hid every value wan actually
 * routes to — including its 53-frame auto-chaining default (#783).
 */
describe("MobileSequenceComposer — wan clip grid", () => {
  const wan: ModelEntry = {
    ...ltx2,
    name: "wan22-i2v-a14b:q5",
    family: "wan",
    source_image: "required",
    default_frames: 53,
  };

  it("offers wan durations on the 4k+1 grid", () => {
    const form = newGenerateForm();
    form.family = "wan";
    form.model = wan.name;
    mountComposer({
      selectedModel: wan,
      form,
      chainLimits: { ...limits, model: wan.name, frames_per_clip_cap: 121 },
      shared: { ...shared, model: wan.name, family: "wan", fps: 16 },
      fps: 16,
    });

    const select = wrapper!.get<HTMLSelectElement>("[data-test='mobile-sequence-frames']");
    const values = Array.from(select.element.options).map((option) => Number(option.value));
    expect(values).toContain(53);
    for (const value of values) expect((value - 1) % 4).toBe(0);
  });
});
