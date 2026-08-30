import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ControlsAside from "./ControlsAside.vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import Stepper from "@ui/components/Stepper.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import { createPinia, setActivePinia } from "pinia";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import type { ModelInfoExtended } from "../../types";
import {
  addHost,
  getGenerateTargetId,
  setGenerateTargetId,
} from "../../lib/hostRegistry";
import { __testing__ as routingTesting } from "../../composables/useHostRouting";
import { CAPABLE_TARGET_ID } from "../../lib/hostRouting";
import type { GenerateFormState } from "../../types";
import type { ChainLimits } from "../../api";

const pushMock = vi.hoisted(() => vi.fn());
vi.mock("vue-router", () => ({
  useRouter: () => ({ push: pushMock }),
}));

// The rail's host picker polls every registered machine. Keep the unit under
// test off the network: no host answers, so rows render from the registry with
// their pre-poll status.
vi.mock("../machines/hostClient", () => ({
  hostStatus: () => Promise.reject(new Error("offline in tests")),
  hostModels: () => Promise.reject(new Error("offline in tests")),
  hostCapabilities: () => Promise.reject(new Error("offline in tests")),
  hostQueue: () => Promise.resolve({ entries: [], plan: null }),
  hostDevices: () => Promise.reject(new Error("offline in tests")),
}));

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

function chainLimits(supportsAudio: boolean): ChainLimits {
  return {
    model: "ltx-2-19b-distilled:fp8",
    frames_per_clip_cap: 121,
    frames_per_clip_recommended: 97,
    max_stages: 16,
    max_total_frames: 1936,
    fade_frames_max: 24,
    transition_modes: ["smooth", "cut", "fade"],
    quantization_family: "fp8",
    supports_audio: supportsAudio,
    supports_sequence: true,
  };
}

describe("ControlsAside", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    localStorage.clear();
    pushMock.mockClear();
    routingTesting.reset();
  });
  afterEach(() => __testing__.resetForTest());

  it("fixes distilled LTX guidance while guided pipelines remain editable", async () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({
          model: "ltx-2.3-22b-distilled:fp8",
          modelFamily: "ltx2",
          guidance: 7,
        }),
        family: "ltx2",
        advCount: 0,
        model: noteModel(
          "ltx-2.3-22b-distilled:fp8",
          "ltx2",
          { default: 20, min: 1, max: 100, step: 1, mode: "adjustable" },
          {
            default: 1,
            min: 1,
            max: 1,
            step: 0.1,
            mode: "fixed",
            note: DISTILLED_NOTE,
          },
          [
            {
              id: "two-stage",
              label: "Two stage",
              request_selector: { pipeline: "two-stage" },
              defaults: { width: 1024, height: 576, steps: 20, guidance: 3 },
              resolution: {
                domain: "dynamic",
                alignment: 32,
                min_width: 64,
                min_height: 64,
                max_pixels: 1_032_192,
                aspect_groups: [],
              },
              steps: {
                default: 20,
                min: 1,
                max: 100,
                step: 1,
                mode: "adjustable",
              },
              guidance: {
                default: 3,
                min: 0,
                max: 100,
                step: 0.1,
                mode: "adjustable",
              },
              capabilities: {
                ...noteCapabilities,
                guidance: { adjustable: true, supports_negative_prompt: true },
              },
              provenance: [],
            },
          ],
        ),
      },
    });
    const guidance = wrapper
      .findAllComponents(SliderRow)
      .find((row) => row.props("label") === "Prompt strength")!;
    expect(guidance.props("disabled")).toBe(true);
    expect(guidance.props("modelValue")).toBe(1);
    // The sentence is the profile's own note, never rail copy.
    expect(wrapper.get("[data-test='fixed-guidance-hint']").text()).toBe(
      DISTILLED_NOTE,
    );

    await wrapper.setProps({
      modelValue: {
        ...(wrapper.props("modelValue") as GenerateFormState),
        pipeline: "two-stage",
      },
    });
    expect(
      wrapper
        .findAllComponents(SliderRow)
        .find((row) => row.props("label") === "Prompt strength")!
        .props("disabled"),
    ).toBe(false);
    expect(wrapper.find("[data-test='fixed-guidance-hint']").exists()).toBe(
      false,
    );
  });

  it("renders the host's own note for a fixed H3 Turbo step count and guidance", () => {
    const name = "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step";
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({
          model: name,
          modelFamily: "minimax-h3",
          steps: 9,
          guidance: 0,
        }),
        family: "minimax-h3",
        advCount: 0,
        model: noteModel(
          name,
          "minimax-h3",
          {
            default: 9,
            min: 9,
            max: 9,
            step: 1,
            mode: "fixed",
            note: H3_TURBO_STEPS_NOTE,
          },
          {
            default: 0,
            min: 0,
            max: 0,
            step: 0.1,
            mode: "fixed",
            note: H3_GUIDANCE_NOTE,
          },
        ),
      },
    });

    expect(wrapper.get("[data-test='fixed-steps-hint']").text()).toBe(
      H3_TURBO_STEPS_NOTE,
    );
    expect(wrapper.get("[data-test='fixed-guidance-hint']").text()).toBe(
      H3_GUIDANCE_NOTE,
    );
    // The old hard-coded sentence was false here: H3 pins guidance at 0 and
    // offers no Dev checkpoint to switch to.
    expect(wrapper.text()).not.toContain("Distilled recipe fixes CFG");
  });

  it("renders no note for an adjustable control, and none for a silent fixed one", () => {
    const adjustable = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ model: "flux-dev:q8", modelFamily: "flux" }),
        family: "flux",
        advCount: 0,
        model: noteModel(
          "flux-dev:q8",
          "flux",
          { default: 20, min: 1, max: 100, step: 1, mode: "adjustable" },
          { default: 3.5, min: 0, max: 100, step: 0.1, mode: "adjustable" },
        ),
      },
    });
    expect(adjustable.find("[data-test='fixed-steps-hint']").exists()).toBe(
      false,
    );
    expect(adjustable.find("[data-test='fixed-guidance-hint']").exists()).toBe(
      false,
    );

    // An older host fixes the control and says nothing; invent no copy.
    const silent = mount(ControlsAside, {
      props: {
        modelValue: baseForm({
          model: "silent:fixed",
          modelFamily: "minimax-h3",
          steps: 9,
          guidance: 0,
        }),
        family: "minimax-h3",
        advCount: 0,
        model: noteModel(
          "silent:fixed",
          "minimax-h3",
          { default: 9, min: 9, max: 9, step: 1, mode: "fixed" },
          { default: 0, min: 0, max: 0, step: 0.1, mode: "fixed" },
        ),
      },
    });
    expect(silent.find("[data-test='fixed-steps-hint']").exists()).toBe(false);
    expect(silent.find("[data-test='fixed-guidance-hint']").exists()).toBe(
      false,
    );
  });

  it("keeps one-shot generated audio in the primary settings", async () => {
    const model = {
      name: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      supports_audio: true,
    } as ModelInfoExtended;
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({
          model: model.name,
          modelFamily: "ltx2",
          enableAudio: null,
        }),
        family: "ltx2",
        model,
      },
    });
    expect(wrapper.find("[data-test='generate-audio-control']").exists()).toBe(
      true,
    );
    wrapper.getComponent(SwitchToggle).vm.$emit("update:modelValue", false);
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.enableAudio).toBe(false);
  });

  it("shows why generated audio is unavailable for a video-only LTX checkpoint", () => {
    const model = noteModel(
      "ltx-2.5-22b-distilled:q4",
      "ltx2",
      { default: 8, min: 8, max: 8, step: 1, mode: "fixed" },
      { default: 1, min: 1, max: 1, step: 0.1, mode: "fixed" },
    );
    model.supports_audio = false;
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ model: model.name, modelFamily: "ltx2" }),
        family: "ltx2",
        model,
      },
    });

    expect(wrapper.find("[data-test='generate-audio-control']").exists()).toBe(
      true,
    );
    expect(wrapper.getComponent(SwitchToggle).props("disabled")).toBe(true);
    expect(wrapper.text()).toContain(
      "Audio assets are not included with this checkpoint",
    );
  });

  it("keeps LTX audio visible when the host recipe cannot deliver it", () => {
    const model = noteModel(
      "ltx-2.5-22b-distilled:q4",
      "ltx2",
      { default: 8, min: 8, max: 8, step: 1, mode: "fixed" },
      { default: 1, min: 1, max: 1, step: 0.1, mode: "fixed" },
    );
    model.supports_audio = true;
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ model: model.name, modelFamily: "ltx2" }),
        family: "ltx2",
        model,
      },
    });

    expect(wrapper.find("[data-test='generate-audio-control']").exists()).toBe(
      true,
    );
    expect(wrapper.getComponent(SwitchToggle).props("disabled")).toBe(true);
    expect(wrapper.text()).toContain(
      "Generated audio is unavailable for this recipe",
    );
  });

  it("keeps sequence generated audio in the primary settings", async () => {
    const draft = useSequenceDraftStore();
    const model = {
      name: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      supports_audio: true,
    } as ModelInfoExtended;
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ model: model.name, modelFamily: "ltx2" }),
        family: "ltx2",
        model,
        output: "sequence",
        chainLimits: chainLimits(true),
      },
    });
    wrapper.getComponent(SwitchToggle).vm.$emit("update:modelValue", true);
    expect(draft.enableAudio).toBe(true);
  });

  it("keeps sequence audio disabled when the routed host rejects it", () => {
    const model = {
      name: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      supports_audio: true,
    } as ModelInfoExtended;
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ model: model.name, modelFamily: "ltx2" }),
        family: "ltx2",
        model,
        output: "sequence",
        chainLimits: chainLimits(false),
      },
    });

    expect(wrapper.getComponent(SwitchToggle).props("disabled")).toBe(true);
    expect(wrapper.text()).toContain(
      "Generated audio is unavailable for this sequence on the selected host",
    );
  });

  it("does not expose the audio toggle for an H3 model restored without a family", () => {
    const model = {
      name: "minimax-h3-fl2va:official-bf16",
      family: "minimax-h3",
      supports_audio: true,
    } as ModelInfoExtended;
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ model: model.name, modelFamily: "" }),
        family: "",
        model,
      },
    });
    expect(wrapper.find("[data-test='generate-audio-control']").exists()).toBe(
      false,
    );
  });

  it("projects the current pixels onto Shape and Resolution", () => {
    const wrapper = factory({ width: 1024, height: 1024 });
    expect(wrapper.getComponent(ShapePicker).props("modelValue")).toBe("1:1");
    expect(wrapper.getComponent(ResolutionSelector).props("modelValue")).toBe(
      "1024x1024",
    );
    expect(wrapper.getComponent(ResolutionSelector).props("options")).toEqual([
      expect.objectContaining({ label: "768×768", sub: "0.6 MP" }),
      expect.objectContaining({ label: "1024×1024", sub: "1 MP" }),
    ]);
  });

  it("exposes and applies Qwen Image aspect ratios on web", async () => {
    const wrapper = factory(
      {
        model: "qwen-image:q4",
        modelFamily: "qwen-image",
        width: 1328,
        height: 1328,
      },
      "qwen-image",
    );
    const shape = wrapper.getComponent(ShapePicker);
    expect(shape.props("options")).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "1:1", label: "1:1" }),
        expect.objectContaining({ id: "4:3", label: "4:3" }),
        expect.objectContaining({ id: "3:4", label: "3:4" }),
        expect.objectContaining({ id: "16:9", label: "16:9" }),
        expect.objectContaining({ id: "9:16", label: "9:16" }),
      ]),
    );

    shape.vm.$emit("update:modelValue", "16:9");
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      width: 1664,
      height: 928,
    });
  });

  it("renders a custom source canvas and restores it after a manual override", async () => {
    const sourceDimensions = { width: 896, height: 1152 };
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm(sourceDimensions),
        family: "qwen-image-edit",
        advCount: 0,
        sourceDimensions,
      },
    });

    expect(wrapper.getComponent(ShapePicker).props("modelValue")).toBe("3:4");
    expect(wrapper.getComponent(ResolutionSelector).props()).toMatchObject({
      resolvedWidth: 896,
      resolvedHeight: 1152,
      customLabel: "Source",
      status: "896×1152 · Matches source",
    });
    expect(wrapper.find("[data-test='match-source-resolution']").exists()).toBe(
      false,
    );

    await wrapper.setProps({
      modelValue: baseForm({ width: 1024, height: 1024 }),
      canvasIntent: "manual",
    });
    expect(wrapper.getComponent(ResolutionSelector).props("status")).toContain(
      "Manual",
    );
    expect(wrapper.getComponent(ResolutionSelector).props("customLabel")).toBe(
      "Manual",
    );
    await wrapper.get("[data-test='match-source-resolution']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next).toMatchObject(sourceDimensions);
  });

  it("renders the profile-compatible Detail admission range", () => {
    const wrapper = factory({}, "sdxl");
    const detail = wrapper
      .findAllComponents(SliderRow)
      .find((row) => row.props("label") === "Detail");
    expect(detail?.props("min")).toBe(1);
    expect(detail?.props("max")).toBe(100);
  });

  it("renders the profile-compatible guidance admission range", () => {
    const wrapper = factory({}, "sdxl");
    const strength = wrapper
      .findAllComponents(SliderRow)
      .find((row) => row.props("label") === "Prompt strength");
    expect(strength?.props("min")).toBe(0);
    expect(strength?.props("max")).toBe(100);
    expect(strength?.props("step")).toBe(0.1);
  });

  it("shows a per-model duration slider for one-shot video", async () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ frames: 97, fps: 24 }),
        family: "ltx2",
        model: {
          name: "ltx-2-19b-distilled:fp8",
          family: "ltx2",
          default_frames: 97,
          default_fps: 24,
          max_runtime_seconds: 20,
          max_frames_absolute: 604,
          frame_step: 8,
        } as never,
      },
    });
    const duration = wrapper.getComponent(VideoDurationSlider);
    expect(duration.text()).toContain("4.0s");
    expect(
      duration.findAll(".ms-slider__mark b").map((mark) => mark.text()),
    ).toEqual(["1×", "2×", "3×", "4×", "5×", "6×"]);
    duration.vm.$emit("update:frames", 241);
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      frames: 241,
    });
  });

  it("applies the projected dims when a shape is picked", async () => {
    const wrapper = factory({ width: 1024, height: 1024 });
    wrapper.getComponent(ShapePicker).vm.$emit("update:modelValue", "4:3");
    await wrapper.vm.$nextTick();
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.width).toBe(1024);
    expect(next.height).toBe(768);
  });

  it("hides aspect ratios the selected wan checkpoint does not support", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({
          model: "wan22-i2v-a14b:q5",
          modelFamily: "wan",
          width: 480,
          height: 832,
        }),
        family: "wan",
        advCount: 0,
        model: {
          name: "wan22-i2v-a14b:q5",
          family: "wan",
          recommended_dimensions: [
            { width: 832, height: 480 },
            { width: 480, height: 832 },
          ],
          dimension_alignment: 16,
          max_pixels: 1280 * 720,
        } as never,
      },
    });

    expect(wrapper.getComponent(ShapePicker).props("options")).toEqual([
      expect.objectContaining({ id: "16:9", label: "16:9" }),
      expect.objectContaining({ id: "9:16", label: "9:16" }),
    ]);
  });

  it("exposes and applies Z-Image's exact 16:9 and 9:16 buckets", async () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({
          model: "z-image-turbo:q4",
          modelFamily: "z-image",
          width: 1024,
          height: 1024,
        }),
        family: "z-image",
        model: {
          name: "z-image-turbo:q4",
          family: "z-image",
          recommended_dimensions: [
            { width: 1024, height: 1024 },
            { width: 1280, height: 720 },
            { width: 720, height: 1280 },
          ],
          dimension_alignment: 16,
          max_pixels: 1_800_000,
        } as never,
      },
    });
    const picker = wrapper.getComponent(ShapePicker);
    expect(picker.props("options")).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "16:9", label: "16:9" }),
        expect.objectContaining({ id: "9:16", label: "9:16" }),
      ]),
    );
    picker.vm.$emit("update:modelValue", "16:9");
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      width: 1280,
      height: 720,
    });
    await wrapper.setProps({
      modelValue: baseForm({
        model: "z-image-turbo:q4",
        modelFamily: "z-image",
        width: 1280,
        height: 720,
      }),
    });
    picker.vm.$emit("update:modelValue", "9:16");
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("update:modelValue")?.at(-1)?.[0]).toMatchObject({
      width: 720,
      height: 1280,
    });
  });

  it("applies the exact pixels of the picked size", async () => {
    const wrapper = factory({ width: 1024, height: 1024 }, "flux");
    wrapper
      .getComponent(ResolutionSelector)
      .vm.$emit("update:modelValue", "768x768");
    await wrapper.vm.$nextTick();
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.width).toBe(768);
    expect(next.height).toBe(768);
  });

  it("labels sizes by pixels and megapixels, never by list position", () => {
    const wrapper = factory({ width: 1024, height: 1024 }, "wuerstchen");
    expect(wrapper.getComponent(ResolutionSelector).props("options")).toEqual([
      expect.objectContaining({ label: "1024×1024", sub: "1 MP" }),
    ]);
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
    const stepper = wrapper
      .findAllComponents(Stepper)
      .find((candidate) => candidate.props("label") === "Batch size")!;
    expect(stepper.props("editable")).toBe(true);
    expect(stepper.props("max")).toBe(10_000);
    stepper.vm.$emit("update:modelValue", 300);
    await wrapper.vm.$nextTick();
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.batchSize).toBe(300);
  });

  it("shows the advanced badge and opens the sheet on phones", async () => {
    // On tablet+ the Advanced sections render inline, so the sheet button is
    // phone-only (mobile: true).
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm(),
        family: "flux",
        advCount: 3,
        mobile: true,
      },
    });
    expect(wrapper.get("[data-test='adv-badge']").text()).toContain("3");
    await wrapper.get("[data-test='open-advanced']").trigger("click");
    expect(wrapper.emitted("open-advanced")).toHaveLength(1);
  });

  it("hides the Advanced sheet button on tablet+ (inline advanced instead)", () => {
    const wrapper = mount(ControlsAside, {
      props: { modelValue: baseForm(), family: "flux", advCount: 3 },
    });
    expect(wrapper.find("[data-test='open-advanced']").exists()).toBe(false);
  });

  it("offers a settings reset in the rail header", async () => {
    const wrapper = factory();
    const reset = wrapper.get("[data-test='settings-reset']");
    expect(reset.attributes("aria-label")).toBe(
      "Reset settings to model defaults",
    );
    await reset.trigger("click");
    expect(wrapper.emitted("reset-settings")).toHaveLength(1);
  });

  it("offers the same settings reset on phones", async () => {
    const wrapper = mount(ControlsAside, {
      props: { modelValue: baseForm(), family: "flux", mobile: true },
    });
    await wrapper.get("[data-test='settings-reset']").trigger("click");
    expect(wrapper.emitted("reset-settings")).toHaveLength(1);
  });

  it("locks batch to 1 for edit families", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm(),
        family: "qwen-image-edit",
        advCount: 0,
      },
    });
    expect(wrapper.find("[data-test='batch-locked']").exists()).toBe(true);
    expect(wrapper.getComponent(Stepper).props("max")).toBe(1);
  });

  it("offers lock-last-seed while random once a run has completed", async () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ seedMode: "random", seed: null }),
        family: "flux",
        advCount: 0,
        lastSeed: 184023,
      },
    });
    const lock = wrapper.get("[data-test='lock-last-seed']");
    expect(lock.text()).toContain("lock last (184023)");

    await lock.trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.seedMode).toBe("static");
    expect(next.seed).toBe(184023);
  });

  it("hides the lock control before any run completes", () => {
    const wrapper = factory({ seedMode: "random", seed: null });
    expect(wrapper.find("[data-test='lock-last-seed']").exists()).toBe(false);
  });

  it("hides the lock control while the seed is already fixed", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ seedMode: "static", seed: 7 }),
        family: "flux",
        advCount: 0,
        lastSeed: 184023,
      },
    });
    expect(wrapper.find("[data-test='lock-last-seed']").exists()).toBe(false);
  });

  it("reroll switches seed back to random", async () => {
    const wrapper = factory({ seedMode: "static", seed: 42 });
    await wrapper.get("[data-test='seed-reroll']").trigger("click");
    const events = wrapper.emitted("update:modelValue") ?? [];
    const last = events.at(-1)?.[0] as GenerateFormState;
    expect(last.seedMode).toBe("random");
    expect(last.seed).toBeNull();
  });

  it("collapses the run-on row to this server with no remote machines", () => {
    const wrapper = factory();
    expect(wrapper.get("[data-test='controls-host']").text()).toContain(
      "Run on this server",
    );
    expect(wrapper.find("[data-test='host-chip']").exists()).toBe(false);
  });

  it("opens the machines workspace when the collapsed host row is clicked", async () => {
    const wrapper = factory();
    await wrapper.get("[data-test='controls-host']").trigger("click");
    expect(pushMock).toHaveBeenCalledWith("/machines");
  });

  it("offers the routing menu once a remote machine is registered", async () => {
    const host = addHost({ url: "http://studio:7680", name: "Studio" });
    const wrapper = factory();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    expect(wrapper.find("[data-test='host-option-auto']").exists()).toBe(true);
    expect(wrapper.find(`[data-test='host-option-${host.id}']`).exists()).toBe(
      true,
    );
  });

  it("persists a routing pick made from the rail", async () => {
    addHost({ url: "http://studio:7680", name: "Studio" });
    const wrapper = factory();
    await wrapper.get("[data-test='host-chip']").trigger("click");
    await wrapper.get("[data-test='host-option-capable']").trigger("click");
    expect(getGenerateTargetId()).toBe(CAPABLE_TARGET_ID);
  });

  it("names an already-persisted sticky pick on the chip", async () => {
    const host = addHost({ url: "http://studio:7680", name: "Studio" });
    setGenerateTargetId(host.id);
    const wrapper = factory();
    await wrapper.vm.$nextTick();
    expect(wrapper.get("[data-test='host-chip']").text()).toContain(
      "Run on Studio",
    );
  });

  // ── Output card (mockup 1c/3a: "mode is a setting, not a place") ─────
  it("renders the Output card ahead of Shape and emits the mode change", async () => {
    const wrapper = factory();
    const card = wrapper.get("[data-test='output-card']");
    // The Output card is the first group so it reads as part of the model
    // decision, above Shape.
    const firstGroup = wrapper.find(".controls__group");
    expect(firstGroup.element).toBe(card.element);
    const sequenceButton = card
      .findAll("button")
      .find((b) => b.text() === "Sequence")!;
    await sequenceButton.trigger("click");
    expect(wrapper.emitted("update:output")?.[0]).toEqual(["sequence"]);
  });

  it("captions Sequence mode with the parked-clips explanation", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm(),
        family: "ltx2",
        advCount: 0,
        output: "sequence",
        clipCount: 3,
      },
    });
    expect(wrapper.get("[data-test='output-caption']").text()).toBe(
      "3 clips on the composer rail · one-shot and sequence prompts stay separate.",
    );
  });

  it("locks Batch to 1 in sequence mode with the one-timeline caption", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ batchSize: 4 }),
        family: "ltx2",
        advCount: 0,
        output: "sequence",
        clipCount: 2,
      },
    });
    const stepper = wrapper
      .findAllComponents(Stepper)
      .find((s) => s.props("label") === "Batch size")!;
    expect(stepper.props("modelValue")).toBe(1);
    expect(stepper.props("max")).toBe(1);
    expect(wrapper.get("[data-test='batch-locked']").text()).toContain(
      "a sequence renders one timeline",
    );
  });

  it("keeps Shape, Resolution, Detail, Prompt strength, and Seed live in sequence mode", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({ width: 1024, height: 1024 }),
        family: "ltx2",
        advCount: 0,
        output: "sequence",
        clipCount: 2,
      },
    });
    expect(wrapper.findComponent(ShapePicker).exists()).toBe(true);
    expect(wrapper.findComponent(ResolutionSelector).exists()).toBe(true);
    const labels = wrapper
      .findAllComponents(SliderRow)
      .map((row) => row.props("label"));
    expect(labels).toContain("Detail");
    expect(labels).toContain("Prompt strength");
    expect(wrapper.find("[data-test='seed-seg']").exists()).toBe(true);
  });
});

describe("ControlsAside — model aspect vs source tie", () => {
  it("defaults to the canonical aspect and preserves an explicit Source pick", async () => {
    const sourceDimensions = { width: 1024, height: 1024 };
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm(sourceDimensions),
        family: "flux",
        advCount: 0,
        sourceDimensions,
      },
    });
    const shape = wrapper.getComponent(ShapePicker);
    const canonical = (
      shape.props("options") as ReadonlyArray<{ id: string }>
    ).find((option) => option.id !== "source")!.id;
    expect(shape.props("modelValue")).toBe(canonical);

    shape.vm.$emit("update:modelValue", "source");
    await wrapper.vm.$nextTick();
    const emitted = wrapper.emitted("update:modelValue");
    if (emitted) {
      await wrapper.setProps({
        modelValue: emitted.at(-1)![0] as GenerateFormState,
      });
    }
    expect(wrapper.emitted("canvas-intent")?.at(-1)).toEqual(["source"]);
    await wrapper.setProps({ canvasIntent: "source" });
    expect(shape.props("modelValue")).toBe("source");

    const remounted = mount(ControlsAside, {
      props: {
        modelValue: emitted?.at(-1)?.[0] as GenerateFormState,
        family: "flux",
        advCount: 0,
        sourceDimensions,
        canvasIntent: "source",
      },
    });
    expect(remounted.getComponent(ShapePicker).props("modelValue")).toBe(
      "source",
    );
  });

  it("advises on an off-profile custom size instead of blocking", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({
          model: "minimax-h3-fl2va:official-bf16",
          modelFamily: "minimax-h3",
          width: 1024,
          height: 576,
        }),
        family: "minimax-h3",
        advCount: 0,
        model: rejectBucketModel(),
      },
    });
    const warning = wrapper.get("[data-test='resolution-warning']");
    expect(warning.text()).toContain("1344 × 768");
    expect(warning.text()).toContain("server may reject");
  });

  it("marks the nearest aspect chip approximate for a custom size", () => {
    const wrapper = mount(ControlsAside, {
      props: {
        modelValue: baseForm({
          model: "minimax-h3-fl2va:official-bf16",
          modelFamily: "minimax-h3",
          width: 1000,
          height: 600,
        }),
        family: "minimax-h3",
        advCount: 0,
        model: rejectBucketModel(),
      },
    });
    const shape = wrapper.getComponent(ShapePicker);
    expect(shape.props("modelValue")).toBe("16:9");
    expect(shape.props("approximate")).toBe(true);
  });
});

/** A reject-policy single-bucket profile (H3's shape): min 1344×768, one 7:4
 * preset. Exercises both the advisory downgrade and the approximate chip. */
function rejectBucketModel(): ModelInfoExtended {
  return {
    name: "minimax-h3-fl2va:official-bf16",
    family: "minimax-h3",
    downloaded: true,
    default_width: 1344,
    default_height: 768,
    generation_profile: {
      schema_version: 1,
      profile_id: "h3.v1",
      profile_hash: "hash",
      default_recipe_id: "default",
      recipes: [
        {
          id: "default",
          label: "Default",
          request_selector: {},
          defaults: { width: 1344, height: 768, steps: 21, guidance: 1 },
          resolution: {
            domain: "buckets",
            alignment: 32,
            min_width: 1344,
            min_height: 768,
            max_pixels: 1_032_192,
            off_bucket: "reject",
            aspect_groups: [
              {
                id: "7:4",
                label: "7:4",
                presets: [
                  {
                    id: "1344x768",
                    width: 1344,
                    height: 768,
                    tier: "recommended",
                  },
                ],
              },
            ],
          },
          steps: { default: 21, min: 1, max: 100, step: 1, mode: "adjustable" },
          guidance: { default: 1, min: 0, max: 20, step: 0.1, mode: "fixed" },
          capabilities: {
            guidance: {
              adjustable: false,
              supports_negative_prompt: false,
              fixed_scale: 1,
            },
            negative_prompt: { mode: "hidden", required: false },
            supports_lora: false,
            supports_controlnet: false,
            supports_identity: false,
            supports_sequence: false,
            supports_extend: false,
            supports_audio: false,
            source_video: { mode: "hidden", required: false },
            mask: { mode: "hidden", required: false },
            keyframes: { mode: "hidden", required: false },
            audio: { mode: "hidden", required: false },
            lora: { mode: "hidden", max_count: 0 },
            controlnet: { mode: "hidden", max_count: 0 },
            output: {
              default_format: "mp4",
              formats: ["mp4"],
              audio_requires_mp4: false,
            },
            wan_recipe: {
              mode: "hidden",
              supports_distill_strength: false,
              supports_first_last_frame: false,
            },
            schedulers: [],
          },
          provenance: [],
        },
      ],
    },
  } as unknown as ModelInfoExtended;
}

const noteCapabilities = {
  guidance: {
    adjustable: false,
    supports_negative_prompt: false,
    fixed_scale: 1,
  },
  negative_prompt: { mode: "hidden", required: false },
  supports_lora: false,
  supports_controlnet: false,
  supports_identity: false,
  supports_sequence: false,
  supports_extend: false,
  supports_audio: false,
  source_video: { mode: "hidden", required: false },
  mask: { mode: "hidden", required: false },
  keyframes: { mode: "hidden", required: false },
  audio: { mode: "hidden", required: false },
  lora: { mode: "hidden", max_count: 0 },
  controlnet: { mode: "hidden", max_count: 0 },
  output: {
    default_format: "mp4",
    formats: ["mp4"],
    audio_requires_mp4: false,
  },
  wan_recipe: {
    mode: "hidden",
    supports_distill_strength: false,
    supports_first_last_frame: false,
  },
  schedulers: [],
};

const DISTILLED_NOTE =
  "Distilled recipe fixes CFG at 1.0. Choose a Dev checkpoint with Auto or a guided pipeline to adjust it.";
const H3_GUIDANCE_NOTE =
  "MiniMax H3 does not use classifier-free guidance; guidance is fixed at 0.";
const H3_TURBO_STEPS_NOTE =
  "Fixed by the 8-step Turbo tier: 9 terminal-inclusive sampler grid points (8 denoise intervals).";

/** A minimal advertised v1 profile whose two numeric controls carry exactly
 * the mode and note under test. Recipe defaults mirror the controls because
 * the client validator cross-checks them. */
function noteModel(
  name: string,
  family: string,
  steps: { default: number; [key: string]: unknown },
  guidance: { default: number; [key: string]: unknown },
  extraRecipes: Record<string, unknown>[] = [],
): ModelInfoExtended {
  return {
    name,
    family,
    downloaded: true,
    generation_profile: {
      schema_version: 1,
      profile_id: `${family}.${name}`,
      profile_hash: "hash",
      default_recipe_id: "default",
      recipes: [
        {
          id: "default",
          label: "Default",
          request_selector: {},
          defaults: {
            width: 1024,
            height: 576,
            steps: steps.default,
            guidance: guidance.default,
          },
          resolution: {
            domain: "dynamic",
            alignment: 32,
            min_width: 64,
            min_height: 64,
            max_pixels: 1_032_192,
            aspect_groups: [],
          },
          steps,
          guidance,
          capabilities: noteCapabilities,
          provenance: [],
        },
        ...extraRecipes,
      ],
    },
  } as unknown as ModelInfoExtended;
}
