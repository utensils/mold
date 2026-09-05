import { mount, type DOMWrapper, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import { beforeEach, describe, expect, it } from "vitest";
import { hunyuan3dRecipe } from "@studio/lib/generationProfile.testFixtures";
import { newGenerateForm, type GenerateForm } from "../lib/generateForm";
import type { ModelEntry } from "../lib/api/types";
import MobileSharedParams from "./MobileSharedParams.vue";

describe("MobileSharedParams video duration", () => {
  beforeEach(() => setActivePinia(createPinia()));

  it("renders distilled guidance fixed but preserves the stored value", async () => {
    const form = reactive({
      ...newGenerateForm(),
      model: "ltx-2.3-22b-distilled:fp8",
      family: "ltx2",
      guidance: 7,
    });
    const model = profileModel(
      form.model,
      form.family,
      { default: 20, min: 1, max: 100, step: 1, mode: "adjustable" },
      { default: 1, min: 1, max: 1, step: 0.1, mode: "fixed", note: DISTILLED_NOTE },
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
          steps: { default: 20, min: 1, max: 100, step: 1, mode: "adjustable" },
          guidance: { default: 3, min: 0, max: 100, step: 0.1, mode: "adjustable" },
          capabilities: {
            ...baseCapabilities,
            guidance: { adjustable: true, supports_negative_prompt: true },
          },
          provenance: [],
        },
      ],
    );
    const wrapper = mount(MobileSharedParams, {
      props: { form, model, lastSeed: null },
      global: { stubs: { MobileResolutionPicker: true, MobileSeedPicker: true } },
    });
    const guidance = wrapper.get("input[step='0.1']");
    expect(guidance.attributes("disabled")).toBeDefined();
    expect((guidance.element as HTMLInputElement).value).toBe("1");
    expect(form.guidance).toBe(7);
    expect(wrapper.get("[data-test='mobile-fixed-guidance-hint']").text()).toContain(
      "fixes CFG at 1.0",
    );
    expect(wrapper.get("[data-test='mobile-fixed-guidance-hint']").classes()).toContain(
      "mobile-generate-hint",
    );
    expect(wrapper.get("[data-test='mobile-fixed-guidance-hint']").classes()).not.toContain(
      "mobile-generate-validation",
    );
    form.pipeline = "two-stage";
    await wrapper.vm.$nextTick();
    expect(wrapper.get("input[step='0.1']").attributes("disabled")).toBeUndefined();
  });

  it("shows the model-aware seconds slider for one-shot video", async () => {
    const form = reactive({
      ...newGenerateForm(),
      model: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      frames: 97,
      fps: 24,
    });
    const model = {
      name: form.model,
      family: form.family,
      max_runtime_seconds: 20,
      max_frames_absolute: 604,
      frame_step: 8,
    } as ModelEntry;
    const wrapper = mount(MobileSharedParams, {
      props: { form, durationModel: model, lastSeed: null },
      global: {
        stubs: { MobileResolutionPicker: true, MobileSeedPicker: true },
      },
    });

    const slider = wrapper.get("[data-test='mobile-duration'] input[type='range']");
    expect(wrapper.get("[data-test='mobile-duration']").text()).toContain("4.0s");
    expect(
      wrapper
        .findAll("[data-test='mobile-duration'] .ms-slider__mark b")
        .map((mark) => mark.text()),
    ).toEqual(["1×", "2×", "3×", "4×", "5×", "6×"]);
    await slider.setValue("241");
    expect(form.frames).toBe(241);
  });

  it("shows an image-to-video tier default at one generation", () => {
    const form = reactive({
      ...newGenerateForm(),
      model: "wan22-i2v-a14b:q5",
      family: "wan",
      frames: 81,
      fps: 16,
    });
    const model = {
      name: form.model,
      family: form.family,
      source_image: "required",
      default_frames: 81,
      default_fps: 16,
      max_frames: 257,
      frame_step: 4,
    } as ModelEntry;
    const wrapper = mount(MobileSharedParams, {
      props: { form, durationModel: model, lastSeed: null },
      global: {
        stubs: { MobileResolutionPicker: true, MobileSeedPicker: true },
      },
    });

    expect(wrapper.get("[data-test='mobile-duration']").text()).toContain(
      "81 frames · 16 fps · 5.1s · 1 generation",
    );
    expect(
      wrapper
        .findAll("[data-test='mobile-duration'] .ms-slider__mark b")
        .map((mark) => mark.text())[0],
    ).toBe("1×");
  });

  it("keeps sequence FPS visible without duplicating the duration control", () => {
    const form = reactive({
      ...newGenerateForm(),
      model: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
    });
    const wrapper = mount(MobileSharedParams, {
      props: { form, lastSeed: null, showFps: true },
      global: {
        stubs: { MobileResolutionPicker: true, MobileSeedPicker: true },
      },
    });

    expect(wrapper.find("[data-test='mobile-duration']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-sequence-fps']").exists()).toBe(true);
  });
});

const baseCapabilities = {
  guidance: { adjustable: false, supports_negative_prompt: false, fixed_scale: 1 },
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
  output: { default_format: "mp4", formats: ["mp4"], audio_requires_mp4: false },
  wan_recipe: {
    mode: "hidden",
    supports_distill_strength: false,
    supports_first_last_frame: false,
  },
  schedulers: [],
};

/** A minimal advertised v1 profile: only the two controls under test carry
 * interesting values, and their note is whatever the host authored. */
function profileModel(
  name: string,
  family: string,
  steps: { default: number; [key: string]: unknown },
  guidance: { default: number; [key: string]: unknown },
  extraRecipes: Record<string, unknown>[] = [],
): ModelEntry {
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
          // The validator cross-checks defaults against the controls, so the
          // recipe default is the control default by construction.
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
          capabilities: baseCapabilities,
          provenance: [],
        },
        ...extraRecipes,
      ],
    },
  } as unknown as ModelEntry;
}

const DISTILLED_NOTE =
  "Distilled recipe fixes CFG at 1.0. Choose a Dev checkpoint with Auto or a guided pipeline to adjust it.";
const H3_GUIDANCE_NOTE =
  "MiniMax H3 does not use classifier-free guidance; guidance is fixed at 0.";
const H3_TURBO_STEPS_NOTE =
  "Fixed by the 8-step Turbo tier: 9 terminal-inclusive sampler grid points (8 denoise intervals).";

describe("MobileSharedParams fixed-control notes", () => {
  beforeEach(() => setActivePinia(createPinia()));

  it("renders the host's own note for a fixed H3 Turbo step count and guidance", () => {
    const form = reactive({
      ...newGenerateForm(),
      model: "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step",
      family: "minimax-h3",
      steps: 9,
      guidance: 0,
    });
    const model = profileModel(
      form.model,
      form.family,
      { default: 9, min: 9, max: 9, step: 1, mode: "fixed", note: H3_TURBO_STEPS_NOTE },
      { default: 0, min: 0, max: 0, step: 0.1, mode: "fixed", note: H3_GUIDANCE_NOTE },
    );
    const wrapper = mount(MobileSharedParams, {
      props: { form, model, lastSeed: null },
      global: { stubs: { MobileResolutionPicker: true, MobileSeedPicker: true } },
    });

    expect(wrapper.get("[data-test='mobile-fixed-steps-hint']").text()).toBe(H3_TURBO_STEPS_NOTE);
    expect(wrapper.get("[data-test='mobile-fixed-guidance-hint']").text()).toBe(H3_GUIDANCE_NOTE);

    // Each note lives INSIDE its own field, directly under that field's input,
    // so it reads as an explanation of that control rather than as trailing
    // prose after the whole two-column grid.
    const stepsField = fieldFor(wrapper, "Steps");
    const guidanceField = fieldFor(wrapper, "Guidance");
    expect(stepsField.find("[data-test='mobile-fixed-steps-hint']").exists()).toBe(true);
    expect(stepsField.find("[data-test='mobile-fixed-guidance-hint']").exists()).toBe(false);
    expect(guidanceField.find("[data-test='mobile-fixed-guidance-hint']").exists()).toBe(true);
    expect(guidanceField.find("[data-test='mobile-fixed-steps-hint']").exists()).toBe(false);
    // The note follows the input it explains.
    const stepsInput = stepsField.get("input").element;
    const stepsHint = stepsField.get("[data-test='mobile-fixed-steps-hint']").element;
    expect(
      stepsInput.compareDocumentPosition(stepsHint) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    // A noted field takes the whole row so the sentence stays readable at
    // phone width; the pair keeps its two-column layout when neither is noted.
    expect(stepsField.classes()).toContain("field--with-note");
    expect(guidanceField.classes()).toContain("field--with-note");
    // The old hard-coded sentence is false here: H3 pins guidance at 0 and
    // offers no Dev checkpoint to switch to.
    expect(wrapper.text()).not.toContain("Distilled recipe fixes CFG");
  });

  it("renders no note for adjustable controls", () => {
    const form = reactive({ ...newGenerateForm(), model: "flux-dev:q8", family: "flux" });
    const model = profileModel(
      form.model,
      form.family,
      { default: 20, min: 1, max: 100, step: 1, mode: "adjustable" },
      { default: 3.5, min: 0, max: 100, step: 0.1, mode: "adjustable" },
    );
    const wrapper = mount(MobileSharedParams, {
      props: { form, model, lastSeed: null },
      global: { stubs: { MobileResolutionPicker: true, MobileSeedPicker: true } },
    });

    expect(wrapper.find("[data-test='mobile-fixed-steps-hint']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-fixed-guidance-hint']").exists()).toBe(false);
    expect(fieldFor(wrapper, "Steps").classes()).not.toContain("field--with-note");
    expect(fieldFor(wrapper, "Guidance").classes()).not.toContain("field--with-note");
  });

  it("invents no copy when a fixed control carries no note (older host)", () => {
    const form = reactive({
      ...newGenerateForm(),
      model: "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step",
      family: "minimax-h3",
    });
    const model = profileModel(
      form.model,
      form.family,
      { default: 9, min: 9, max: 9, step: 1, mode: "fixed" },
      { default: 0, min: 0, max: 0, step: 0.1, mode: "fixed" },
    );
    const wrapper = mount(MobileSharedParams, {
      props: { form, model, lastSeed: null },
      global: { stubs: { MobileResolutionPicker: true, MobileSeedPicker: true } },
    });

    expect(wrapper.find("[data-test='mobile-fixed-steps-hint']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-fixed-guidance-hint']").exists()).toBe(false);
  });
});

/**
 * The 3-D group. Every control here is driven by the recipe's advertised
 * `capabilities.mesh` block — the phone never carries an octree ladder, a
 * threshold range or a face budget of its own — and an untouched control
 * stays `null` so the request omits it and the engine's own default renders.
 */
describe("MobileSharedParams mesh controls", () => {
  beforeEach(() => setActivePinia(createPinia()));

  function meshModel(recipe = hunyuan3dRecipe()): ModelEntry {
    return {
      name: "hunyuan3d-mini-turbo:fp16",
      family: "hunyuan3d",
      downloaded: true,
      source_image: "required",
      generation_profile: {
        schema_version: 1,
        profile_id: "hunyuan3d.mini",
        profile_hash: "hunyuan3d-mini-hash",
        default_recipe_id: "default",
        recipes: [recipe],
      },
    } as ModelEntry;
  }

  function meshForm(): GenerateForm {
    return reactive({
      ...newGenerateForm(),
      model: "hunyuan3d-mini-turbo:fp16",
      family: "hunyuan3d",
      width: 0,
      height: 0,
    });
  }

  function mountMesh(form: GenerateForm, model = meshModel()) {
    return mount(MobileSharedParams, {
      props: { form, model, lastSeed: null },
      global: { stubs: { MobileResolutionPicker: true, MobileSeedPicker: true } },
    });
  }

  it("renders no mesh group for a recipe that advertises none", () => {
    const form = reactive({ ...newGenerateForm(), model: "sdxl:test", family: "sdxl" });
    const wrapper = mount(MobileSharedParams, {
      props: { form, lastSeed: null },
      global: { stubs: { MobileResolutionPicker: true, MobileSeedPicker: true } },
    });

    expect(wrapper.find("[data-test='mobile-mesh-controls']").exists()).toBe(false);
  });

  it("lights the advertised octree default while the control is untouched", async () => {
    const form = meshForm();
    const wrapper = mountMesh(form);

    const segments = wrapper.get("[data-test='mobile-mesh-octree']").findAll("button");
    expect(segments.map((segment) => segment.get(".ms-seg__label").text())).toEqual([
      "128",
      "192",
      "256",
      "320",
      "384",
    ]);
    expect(form.mesh.octreeResolution).toBeNull();
    expect(segments[2]!.attributes("aria-checked")).toBe("true");

    await segments[4]!.trigger("click");
    expect(form.mesh.octreeResolution).toBe(384);
    expect(
      wrapper
        .get("[data-test='mobile-mesh-octree']")
        .findAll("button")[4]!
        .attributes("aria-checked"),
    ).toBe("true");
  });

  it("drives the iso threshold from the advertised float control", async () => {
    const form = meshForm();
    const wrapper = mountMesh(form);

    const slider = wrapper.get<HTMLInputElement>("[data-test='mobile-mesh-threshold']");
    expect(slider.attributes("min")).toBe("0");
    expect(slider.attributes("max")).toBe("1");
    expect(slider.attributes("step")).toBe("0.01");
    expect(slider.attributes("disabled")).toBeUndefined();
    expect(slider.element.value).toBe("0.6");

    slider.element.value = "0.42";
    await slider.trigger("input");
    expect(form.mesh.threshold).toBe(0.42);
  });

  it("disables a fixed threshold and shows the host's own note", () => {
    const recipe = hunyuan3dRecipe();
    recipe.capabilities.mesh = {
      ...recipe.capabilities.mesh!,
      threshold: {
        default: 0.5,
        min: 0.5,
        max: 0.5,
        step: 0.01,
        mode: "fixed",
        note: "This build pins the iso surface at 0.50.",
      },
    };
    const wrapper = mountMesh(meshForm(), meshModel(recipe));

    const slider = wrapper.get("[data-test='mobile-mesh-threshold']");
    expect(slider.attributes("disabled")).toBeDefined();
    expect(wrapper.get("[data-test='mobile-mesh-threshold-note']").text()).toBe(
      "This build pins the iso surface at 0.50.",
    );
  });

  it("keeps target faces optional and names the advertised bounds for a budget outside them", async () => {
    const form = meshForm();
    const wrapper = mountMesh(form);

    const faces = wrapper.get<HTMLInputElement>("[data-test='mobile-mesh-target-faces']");
    expect(faces.element.value).toBe("");
    expect(faces.attributes("min")).toBe("100");
    expect(faces.attributes("max")).toBe("2000000");
    expect(form.mesh.targetFaces).toBeNull();
    expect(wrapper.find("[data-test='mobile-mesh-target-faces-error']").exists()).toBe(false);

    faces.element.value = "25000";
    await faces.trigger("change");
    expect(form.mesh.targetFaces).toBe(25_000);
    expect(wrapper.find("[data-test='mobile-mesh-target-faces-error']").exists()).toBe(false);

    // The typed value stands — it is not snapped behind the user's back; the
    // advisory names the bounds the host will hold it to.
    faces.element.value = "9000000";
    await faces.trigger("change");
    expect(form.mesh.targetFaces).toBe(9_000_000);
    expect(wrapper.get("[data-test='mobile-mesh-target-faces-error']").text()).toBe(
      "Use a whole number of faces from 100 to 2000000.",
    );
    expect(faces.attributes("aria-invalid")).toBe("true");

    faces.element.value = "";
    await faces.trigger("change");
    expect(form.mesh.targetFaces).toBeNull();
    expect(wrapper.find("[data-test='mobile-mesh-target-faces-error']").exists()).toBe(false);
  });

  // The group must never reach into the prop to create its own slot: a form
  // restored without one is initialised by the owner, and reading it here is
  // side-effect free.
  it("reads a form that carries no mesh slot without mutating it from a computed", () => {
    const form = meshForm();
    const legacy = form as unknown as { mesh?: unknown };
    delete legacy.mesh;
    const wrapper = mountMesh(form);
    expect(wrapper.find("[data-test='mobile-mesh-controls']").exists()).toBe(true);
    expect(legacy.mesh).toBeUndefined();
  });
});

/** The `.field` label wrapping one named control — the container a note has to
 * live inside to read as that control's own explanation. */
function fieldFor(wrapper: VueWrapper, label: string): DOMWrapper<HTMLElement> {
  const field = wrapper
    .findAll<HTMLElement>("label.field")
    .find((candidate) => candidate.get("span").text() === label);
  if (!field) throw new Error(`no field labelled ${label}`);
  return field;
}
