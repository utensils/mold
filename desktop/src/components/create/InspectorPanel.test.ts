import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";
import InspectorPanel from "./InspectorPanel.vue";
import AdvancedSettings from "./AdvancedSettings.vue";
import ModelPicker from "./ModelPicker.vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
import Stepper from "@ui/components/Stepper.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import PanelResizeHandle from "../shell/PanelResizeHandle.vue";
import TemplatesPanel from "../generate/TemplatesPanel.vue";
import { aspectIdFor } from "../../lib/resolutions";
import { buildRequest, newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useModelStore } from "../../stores/models";
import { useConnectionStore } from "../../stores/connection";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useLibraryPrefsStore } from "../../stores/libraryPrefs";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { ModelEntry } from "../../lib/api/types";
import { apiJsonTo } from "../../lib/api/client";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn(),
}));
vi.mock("../../lib/ipc", () => ({ ipc: {}, inTauri: () => false }));
// The File under group reads the Library's merged tags and collections, so the
// gallery store now fetches from Create. Serve those two listings from memory.
const libraryListings = vi.hoisted(() => ({
  collections: [] as { id: string; name: string; slug: string }[],
  tags: [] as { name: string; count: number }[],
}));
vi.mock("@studio/api/galleryOrganization", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  listCollections: vi.fn(() => Promise.resolve(libraryListings.collections)),
  listTags: vi.fn(() => Promise.resolve(libraryListings.tags)),
}));

beforeEach(() => {
  setActivePinia(createPinia());
  libraryListings.collections = [];
  libraryListings.tags = [];
});
afterEach(() => (document.body.innerHTML = ""));

function formFor(family: string): GenerateForm {
  return reactive({ ...newGenerateForm(), family });
}

describe("InspectorPanel — layout", () => {
  it("disables ineffective distilled guidance and re-enables a guided recipe", async () => {
    const form = formFor("ltx2");
    form.model = "ltx-2.3-22b-distilled:fp8";
    form.guidance = 6;
    useModelStore().all = [
      noteModel(
        form.model,
        "ltx2",
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
              ...noteCapabilities,
              guidance: { adjustable: true, supports_negative_prompt: true },
            },
            provenance: [],
          },
        ],
      ),
    ];
    const wrapper = mount(InspectorPanel, { props: { form } });
    const guidance = () =>
      wrapper
        .findAllComponents(SliderRow)
        .find((row) => row.props("label") === "Stick to my words")!;
    expect(guidance().props("disabled")).toBe(true);
    expect(guidance().props("modelValue")).toBe(1);
    // The sentence is the profile's own note, not inspector copy.
    expect(wrapper.get("[data-test='fixed-guidance-hint']").text()).toBe(DISTILLED_NOTE);
    form.pipeline = "two-stage";
    await flushPromises();
    expect(guidance().props("disabled")).toBe(false);
    expect(wrapper.find("[data-test='fixed-guidance-hint']").exists()).toBe(false);
  });

  it("renders the host's own note for a fixed H3 Turbo step count and guidance", () => {
    const name = "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step";
    useModelStore().all = [
      noteModel(
        name,
        "minimax-h3",
        { default: 9, min: 9, max: 9, step: 1, mode: "fixed", note: H3_TURBO_STEPS_NOTE },
        { default: 0, min: 0, max: 0, step: 0.1, mode: "fixed", note: H3_GUIDANCE_NOTE },
      ),
    ];
    const form = formFor("minimax-h3");
    form.model = name;
    form.steps = 9;
    form.guidance = 0;

    const wrapper = mount(InspectorPanel, { props: { form } });

    expect(wrapper.get("[data-test='fixed-steps-hint']").text()).toBe(H3_TURBO_STEPS_NOTE);
    expect(wrapper.get("[data-test='fixed-guidance-hint']").text()).toBe(H3_GUIDANCE_NOTE);
    // The old hard-coded sentence was false here: H3 pins guidance at 0 and
    // offers no Dev checkpoint to switch to.
    expect(wrapper.text()).not.toContain("Distilled recipe fixes CFG");
  });

  it("renders no note for adjustable controls, and none for a fixed one the host left silent", () => {
    useModelStore().all = [
      noteModel(
        "flux-dev:q8",
        "flux",
        { default: 20, min: 1, max: 100, step: 1, mode: "adjustable" },
        { default: 3.5, min: 0, max: 100, step: 0.1, mode: "adjustable" },
      ),
    ];
    const adjustable = formFor("flux");
    adjustable.model = "flux-dev:q8";
    const open = mount(InspectorPanel, { props: { form: adjustable } });
    expect(open.find("[data-test='fixed-steps-hint']").exists()).toBe(false);
    expect(open.find("[data-test='fixed-guidance-hint']").exists()).toBe(false);

    // An older host fixes the control and says nothing; invent no copy.
    useModelStore().all = [
      noteModel(
        "silent:fixed",
        "minimax-h3",
        { default: 9, min: 9, max: 9, step: 1, mode: "fixed" },
        { default: 0, min: 0, max: 0, step: 0.1, mode: "fixed" },
      ),
    ];
    const silent = formFor("minimax-h3");
    silent.model = "silent:fixed";
    silent.steps = 9;
    silent.guidance = 0;
    const quiet = mount(InspectorPanel, { props: { form: silent } });
    expect(quiet.find("[data-test='fixed-steps-hint']").exists()).toBe(false);
    expect(quiet.find("[data-test='fixed-guidance-hint']").exists()).toBe(false);
  });

  it("renders every primary generation control", () => {
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    expect(wrapper.findComponent(ShapePicker).exists()).toBe(true);
    expect(wrapper.findComponent(ResolutionSelector).exists()).toBe(true);
    // Detail + Stick to my words sliders.
    expect(wrapper.findAllComponents(SliderRow)).toHaveLength(2);
    // "Make N" moved to the composer's control row — the inspector must not
    // carry a second copy of it.
    expect(wrapper.findComponent(Stepper).exists()).toBe(false);
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
    expect(duration.findAll(".ms-slider__mark b").map((mark) => mark.text())).toEqual([
      "1×",
      "2×",
      "3×",
      "4×",
      "5×",
      "6×",
    ]);
    expect(wrapper.find('[data-test="generate-audio-control"]').exists()).toBe(true);
    wrapper.getComponent(SwitchToggle).vm.$emit("update:modelValue", true);
    expect(form.enableAudio).toBe(true);
    duration.vm.$emit("update:frames", 241);
    await flushPromises();
    expect(form.frames).toBe(241);
  });

  it("keeps the audio control visible for a video-only LTX-2.5 checkpoint", () => {
    const model = noteModel(
      "ltx-2.5-22b-distilled:q4",
      "ltx2",
      { default: 8, min: 8, max: 8, step: 1, mode: "fixed" },
      { default: 1, min: 1, max: 1, step: 0.1, mode: "fixed" },
    );
    model.supports_audio = false;
    useModelStore().all = [model];
    const form = formFor("ltx2");
    form.model = model.name;

    const wrapper = mount(InspectorPanel, { props: { form } });

    expect(wrapper.find('[data-test="generate-audio-control"]').exists()).toBe(true);
    expect(wrapper.getComponent(SwitchToggle).props("disabled")).toBe(true);
    expect(wrapper.text()).toContain("Audio assets are not included with this checkpoint");
  });

  it("keeps LTX-2.5 audio visible when the host recipe cannot deliver it", () => {
    const model = noteModel(
      "ltx-2.5-22b-distilled:q4",
      "ltx2",
      { default: 8, min: 8, max: 8, step: 1, mode: "fixed" },
      { default: 1, min: 1, max: 1, step: 0.1, mode: "fixed" },
    );
    model.supports_audio = true;
    useModelStore().all = [model];
    const form = formFor("ltx2");
    form.model = model.name;

    const wrapper = mount(InspectorPanel, { props: { form } });

    expect(wrapper.find('[data-test="generate-audio-control"]').exists()).toBe(true);
    expect(wrapper.getComponent(SwitchToggle).props("disabled")).toBe(true);
    expect(wrapper.text()).toContain("Generated audio is unavailable for this recipe");
  });

  it("does not expose the audio toggle for an H3 model restored without a family", () => {
    const form = formFor("");
    form.model = "minimax-h3-fl2va:official-bf16";
    const wrapper = mount(InspectorPanel, { props: { form } });
    expect(wrapper.find('[data-test="generate-audio-control"]').exists()).toBe(false);
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
  it("uses the explicit host's model profile instead of the fleet union's first row", () => {
    const name = "shared-model";
    const local = {
      name,
      family: "flux",
      downloaded: true,
      recommended_dimensions: [{ width: 1024, height: 1024 }],
    } as ModelEntry;
    const remote = {
      ...local,
      recommended_dimensions: [{ width: 1280, height: 720 }],
    } as ModelEntry;
    const connection = useConnectionStore();
    connection.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
    connection.status = "ready";
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "HAL 9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    useModelStore().all = [local];
    const hostModels = useHostModelsStore();
    hostModels.byHost.local = { entries: [local], fetchedAt: Date.now(), error: null };
    hostModels.byHost["hal9000-7680"] = {
      entries: [remote],
      fetchedAt: Date.now(),
      error: null,
    };
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    const form = formFor("flux");
    form.model = name;

    const wrapper = mount(InspectorPanel, { props: { form } });

    expect(wrapper.getComponent(ShapePicker).props("options")).toEqual([
      expect.objectContaining({ label: "16:9" }),
    ]);
  });

  it("marks the nearest aspect chip approximate for a custom Advanced size", () => {
    const model = {
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
                  presets: [{ id: "1344x768", width: 1344, height: 768, tier: "recommended" }],
                },
              ],
            },
            steps: { default: 21, min: 1, max: 100, step: 1, mode: "adjustable" },
            guidance: { default: 1, min: 0, max: 20, step: 0.1, mode: "fixed" },
            capabilities: {
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
            },
            provenance: [],
          },
        ],
      },
    } as unknown as ModelEntry;
    useModelStore().all = [model];
    const form = formFor("minimax-h3");
    form.model = model.name;
    form.width = 1024;
    form.height = 576;

    const wrapper = mount(InspectorPanel, { props: { form } });
    const shape = wrapper.getComponent(ShapePicker);
    expect(shape.props("modelValue")).toBe("16:9");
    expect(shape.props("approximate")).toBe(true);

    // The exact bucket clears the mark.
    form.width = 1344;
    form.height = 768;
    return wrapper.vm.$nextTick().then(() => {
      expect(shape.props("approximate")).toBe(false);
    });
  });

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
      expect.objectContaining({ id: "16:9", label: "16:9" }),
      expect.objectContaining({ id: "9:16", label: "9:16" }),
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

  it("exposes and applies Qwen Image aspect ratios on desktop", async () => {
    const form = formFor("qwen-image");
    form.model = "qwen-image:q4";
    form.width = 1328;
    form.height = 1328;

    const wrapper = mount(InspectorPanel, { props: { form } });
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
    await flushPromises();
    expect([form.width, form.height]).toEqual([1664, 928]);
  });

  it("applies a picked shape to the form dimensions at the current budget", async () => {
    const form = formFor("flux");
    form.width = 1024;
    form.height = 1024;
    const wrapper = mount(InspectorPanel, { props: { form } });
    expect(wrapper.findComponent(ShapePicker).props("modelValue")).toBe("1:1");
    wrapper.findComponent(ShapePicker).vm.$emit("update:modelValue", "16:9");
    await flushPromises();
    expect(aspectIdFor(form.width, form.height)).toBe("wide");
    expect(form.width % 16).toBe(0);
    expect(form.height % 16).toBe(0);
  });

  it("selects an exact size from the family's pixel ladder", async () => {
    const form = formFor("flux");
    form.width = 1024;
    form.height = 1024;
    const wrapper = mount(InspectorPanel, { props: { form } });
    const selector = wrapper.findComponent(ResolutionSelector);
    expect(selector.props("modelValue")).toBe("1024x1024");
    expect(selector.props("options")).toEqual([
      expect.objectContaining({ id: "768x768", label: "768×768" }),
      expect.objectContaining({ id: "1024x1024", label: "1024×1024" }),
    ]);
    selector.vm.$emit("update:modelValue", "768x768");
    await flushPromises();
    expect([form.width, form.height]).toEqual([768, 768]);
  });

  it("labels sizes by pixels and megapixels, never by list position", () => {
    const wrapper = mount(InspectorPanel, {
      props: { form: formFor("wuerstchen") },
    });
    expect(wrapper.findComponent(ResolutionSelector).props("options")).toEqual([
      expect.objectContaining({ label: "1024×1024", sub: "1 MP" }),
    ]);
  });
});

// "Make N" now lives on the composer's control row; its Stepper contract and
// the one-at-a-time lock are covered by `ComposerCard.test.ts`.

describe("InspectorPanel — tabs", () => {
  it("opens Starters and Recent as tabs beside Settings, never as popovers", async () => {
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    expect(wrapper.find('[data-test="inspector-starters"]').exists()).toBe(false);
    await wrapper.get('[data-test="inspector-tab-starters"]').trigger("click");
    expect(wrapper.emitted("update:tab")).toEqual([["starters"]]);
    await wrapper.get('[data-test="inspector-tab-recent"]').trigger("click");
    expect(wrapper.emitted("update:tab")?.at(-1)).toEqual(["recent"]);
  });

  it("loads a starting point from the Starters tab", () => {
    const wrapper = mount(InspectorPanel, {
      props: { form: formFor("flux"), tab: "starters" },
    });
    expect(wrapper.find('[data-test="inspector-starters"]').exists()).toBe(true);
    const template = { id: "t1", name: "River preset" } as never;
    wrapper.findComponent(TemplatesPanel).vm.$emit("load", template);
    expect(wrapper.emitted("load-template")).toEqual([[template]]);
  });

  it("hands a past prompt back from the Recent tab", async () => {
    const wrapper = mount(InspectorPanel, {
      props: { form: formFor("flux"), tab: "recent", history: ["a lighthouse", "a river"] },
    });
    const rows = wrapper
      .get('[data-test="inspector-recent"]')
      .findAll('[data-test="recent-prompt"]');
    expect(rows.map((r) => r.text())).toEqual([
      expect.stringContaining("a lighthouse"),
      expect.stringContaining("a river"),
    ]);
    await rows[1]!.trigger("click");
    expect(wrapper.emitted("use-prompt")).toEqual([["a river"]]);
  });

  it("names the machine the print runs on in the Settings tab", () => {
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    const row = wrapper.get('[data-test="inspector-host"]');
    expect(row.text()).toContain("Where it runs");
    expect(row.find('[data-test="host-chip"]').exists()).toBe(true);
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
  it("passes the selected generation machine to the one-shot LoRA picker", async () => {
    const connection = useConnectionStore();
    connection.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local" };
    connection.status = "ready";
    useHostsStore().extras.push({
      id: "plato-7680",
      label: "plato",
      url: "http://plato:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: "plato-instance",
    });
    useAppPrefsStore().settings = { generateTargetHost: "plato-7680" } as never;
    const form = formFor("z-image");
    form.model = "z-image-turbo:q8";
    const wrapper = mount(InspectorPanel, { props: { form } });

    await wrapper.get('[data-test="open-advanced"]').trigger("click");

    expect(wrapper.getComponent(AdvancedSettings).props("loraRoute")).toMatchObject({
      hostId: "plato-7680",
      target: { baseUrl: "http://plato:7680" },
    });
  });

  it("never counts the Sequence opening image — it is primary-form media, not Advanced", async () => {
    const draft = useSequenceDraftStore();
    draft.output = "sequence";
    draft.ensureClips(97);
    draft.openingImage = { filename: "opening.png", base64: "PARKED" };
    const form = formFor("ltx2");
    form.model = "ltx-2-19b-distilled:fp8";
    const wrapper = mount(InspectorPanel, { props: { form } });

    expect(wrapper.find("[data-test='advanced-count']").exists()).toBe(false);
    expect(draft.openingImage?.base64).toBe("PARKED");

    draft.clips[0]!.cameraControl = "dolly-in";
    await flushPromises();
    expect(wrapper.get("[data-test='advanced-count']").text()).toContain("1 on");
    expect(draft.openingImage?.base64).toBe("PARKED");
  });

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

    expect(wrapper.find('[data-test="generate-audio-control"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="sequence-section-opening-image"]').exists()).toBe(false);
    expect(wrapper.find('[data-test="sequence-section-negative"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="sequence-section-audio"]').exists()).toBe(false);
    expect(wrapper.find('[data-test="inline-advanced"]').exists()).toBe(false);
  });
});

describe("InspectorPanel — sequence opening image in the primary form", () => {
  function sequenceForm() {
    const draft = useSequenceDraftStore();
    draft.output = "sequence";
    draft.ensureClips(97);
    const form = formFor("ltx2");
    form.model = "ltx-2-19b-distilled:fp8";
    return form;
  }

  it("renders the opening image beside the other primary controls, never inside Advanced", async () => {
    const wrapper = mount(InspectorPanel, { props: { form: sequenceForm() } });

    const field = wrapper.get("[data-test='inspector-sequence-opening-image']");
    expect(field.find("[data-test='sequence-opening-image-well']").exists()).toBe(true);

    await wrapper.get('[data-test="open-advanced"]').trigger("click");
    const advanced = wrapper.get("[data-test='sequence-inline-advanced']");
    expect(advanced.find("[data-test='sequence-opening-image-well']").exists()).toBe(false);
    expect(advanced.find("[data-test='sequence-source-strength']").exists()).toBe(false);
    // Still exactly one well on the panel — it did not move, it was moved out.
    expect(wrapper.findAll("[data-test='sequence-opening-image-well']")).toHaveLength(1);
  });

  it("stands down for a checkpoint whose contract rejects a source image", async () => {
    const draft = useSequenceDraftStore();
    draft.output = "sequence";
    draft.ensureClips(97);
    draft.openingImage = { filename: "opening.png", base64: "PARKED" };
    const form = formFor("wan");
    form.model = "wan22-t2v-a14b";
    form.sourceImageCapability = "unsupported";
    const wrapper = mount(InspectorPanel, { props: { form } });

    expect(wrapper.find("[data-test='inspector-sequence-opening-image']").exists()).toBe(false);
    expect(draft.openingImage?.base64).toBe("PARKED");

    form.sourceImageCapability = "optional";
    await flushPromises();
    expect(wrapper.find("[data-test='inspector-sequence-opening-image']").exists()).toBe(true);
    expect(draft.openingImage?.base64).toBe("PARKED");
  });

  it("keeps the one-shot source well out of sequence mode", () => {
    const wrapper = mount(InspectorPanel, { props: { form: sequenceForm() } });
    expect(wrapper.find("[data-test='inspector-source-media']").exists()).toBe(false);
  });

  it("clears the opening image with the header Reset, like one-shot source media", async () => {
    const draft = useSequenceDraftStore();
    const form = sequenceForm();
    draft.openingImage = { filename: "opening.png", base64: "QUJD" };
    const wrapper = mount(InspectorPanel, { props: { form } });

    await wrapper.get('[data-test="settings-reset"]').trigger("click");

    expect(draft.openingImage).toBeNull();
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

  it("restores the model's defaults while preserving prompt/model and resetting Batch", async () => {
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
    expect(form.batchSize).toBe(1);
    expect(form.negativePrompt).toBe("");
    expect(form.seed).toBe("");
    expect(form.steps).toBe(30);
    expect(form.width).toBe(1024);
    expect(form.height).toBe(768);
  });

  it("returns the canvas authority to the model on reset (#1166)", async () => {
    const form = formFor("flux");
    form.sourceImage = "SRC";
    form.sourceImageWidth = 1024;
    form.sourceImageHeight = 1024;
    const wrapper = mount(InspectorPanel, {
      props: { form, canvasIntent: "source" },
    });

    await wrapper.get('[data-test="settings-reset"]').trigger("click");

    // Without this, the next model change would re-snap the reset canvas
    // back onto the attached source.
    expect(wrapper.emitted("canvas-intent")?.at(-1)).toEqual(["model-default"]);
  });

  it("resets sequence audio as part of the full Settings reset", async () => {
    const draft = useSequenceDraftStore();
    draft.output = "sequence";
    draft.enableAudio = true;
    const wrapper = mount(InspectorPanel, { props: { form: formFor("ltx2") } });

    await wrapper.get('[data-test="settings-reset"]').trigger("click");

    expect(draft.enableAudio).toBe(false);
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
    // The switch-back caption appears. (The batch lock itself is the
    // composer's chip now — see `ComposerCard.test.ts`.)
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

  it("shows a remote H3 download-only install with readable labels and its refusal", async () => {
    const h3 = {
      ...model,
      name: "minimax-h3-fl2va:comfy-pruned-nvfp4",
      family: "minimax-h3",
      runtime_available: false,
      runtime_unavailable_reason: "This H3 weight layout has no executable loader.",
    } as ModelEntry;
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "HAL 9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    useHostModelsStore().byHost["hal9000-7680"] = {
      entries: [h3],
      fetchedAt: Date.now(),
      error: null,
    };
    vi.mocked(apiJsonTo).mockResolvedValueOnce([h3]);
    const form = useGenerateFormStore().form;
    const wrapper = mount(InspectorPanel, { props: { form }, attachTo: document.body });

    await wrapper.get('[data-test="selected-model-name"]').trigger("click");

    expect(wrapper.get(".ms-model__group").text()).toBe("MiniMax H3");
    expect(wrapper.get('[data-test="model-option-name"]').text()).toBe("MiniMax H3 FL2VA · NVFP4");
    expect(wrapper.get('[data-test="model-disabled-reason"]').text()).toBe(
      "Download only — This H3 weight layout has no executable loader.",
    );
    expect(wrapper.get(".ms-model__option").attributes("disabled")).toBeDefined();
  });

  it("keeps an unavailable model disabled on the pinned host when another host can run it", () => {
    const name = "shared-runtime-model";
    const runnable = {
      ...model,
      name,
      runtime_available: true,
    } as ModelEntry;
    const unavailable = {
      ...runnable,
      runtime_available: false,
      runtime_unavailable_reason: "HAL cannot execute this H3 weight layout.",
    } as ModelEntry;
    const connection = useConnectionStore();
    connection.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
    connection.status = "ready";
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "HAL 9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    const hostModels = useHostModelsStore();
    hostModels.byHost.local = { entries: [runnable], fetchedAt: Date.now(), error: null };
    hostModels.byHost["hal9000-7680"] = {
      entries: [unavailable],
      fetchedAt: Date.now(),
      error: null,
    };
    useModelStore().all = [runnable];
    useAppPrefsStore().settings = { generateTargetHost: "hal9000-7680" } as never;
    const form = useGenerateFormStore().form;
    form.model = name;
    const wrapper = mount(InspectorPanel, { props: { form }, attachTo: document.body });

    const disabledReason = wrapper.getComponent(ModelPicker).props("disabledReason");
    expect(disabledReason).toBeTypeOf("function");
    if (!disabledReason) throw new Error("ModelPicker disabledReason prop is required");
    expect(disabledReason(unavailable)).toBe(
      "Download only — HAL cannot execute this H3 weight layout.",
    );
  });
});

describe("InspectorPanel — source media in the primary form", () => {
  const wanModel = (sourceImage?: string): ModelEntry =>
    ({
      name: "wan22-t2v-a14b",
      family: "wan",
      downloaded: true,
      ...(sourceImage === undefined ? {} : { source_image: sourceImage }),
    }) as ModelEntry;

  it("renders the source well for an image family without any Advanced toggle", () => {
    const wrapper = mount(InspectorPanel, { props: { form: formFor("sd15") } });
    const field = wrapper.get("[data-test='inspector-source-media']");
    expect(field.find("[data-test='source-well']").exists()).toBe(true);
  });

  it("follows the advertised per-model contract, exactly like resolutions", async () => {
    const form = formFor("wan");
    form.model = "wan22-t2v-a14b";
    useModelStore().all = [wanModel("unsupported")];
    const hidden = mount(InspectorPanel, { props: { form } });
    expect(hidden.find("[data-test='inspector-source-media']").exists()).toBe(false);
    hidden.unmount();

    useModelStore().all = [wanModel("required")];
    const required = mount(InspectorPanel, { props: { form } });
    const field = required.get("[data-test='inspector-source-media']");
    expect(field.find("[data-test='source-required-badge']").exists()).toBe(true);
    expect(field.find("[data-test='end-frame-well']").exists()).toBe(true);
  });

  it("hides source-derived controls while an image is parked, then restores them", async () => {
    const form = formFor("wan");
    form.model = "wan22-t2v-a14b";
    form.sourceImage = "PARKED";
    form.sourceImageName = "previous.png";
    form.sourceImageWidth = 1024;
    form.sourceImageHeight = 1024;
    form.width = 720;
    form.height = 1280;
    form.sourceImageCapability = "unsupported";
    useModelStore().all = [wanModel("unsupported")];

    const wrapper = mount(InspectorPanel, { props: { form } });

    expect(wrapper.getComponent(ShapePicker).props("options")).not.toEqual(
      expect.arrayContaining([expect.objectContaining({ id: "source" })]),
    );
    expect(wrapper.getComponent(ResolutionSelector).props("customLabel")).toBeUndefined();
    expect(wrapper.getComponent(ResolutionSelector).props("status")).not.toContain("source");
    expect(wrapper.find("[data-test='match-source-resolution']").exists()).toBe(false);
    expect(form.sourceImage).toBe("PARKED");
    expect(form.sourceImageName).toBe("previous.png");
    expect(form.sourceImageWidth).toBe(1024);
    expect(form.sourceImageHeight).toBe(1024);

    form.sourceImageCapability = "optional";
    useModelStore().all = [wanModel("optional")];
    await flushPromises();
    expect(wrapper.getComponent(ShapePicker).props("options")).toEqual(
      expect.arrayContaining([expect.objectContaining({ id: "source" })]),
    );
    expect(wrapper.getComponent(ResolutionSelector).props("status")).toContain("source");
    expect(wrapper.find("[data-test='match-source-resolution']").exists()).toBe(true);
  });

  it("renders H3 FL2VA boundaries as the same standard wells and applies a gallery pick", async () => {
    const form = formFor("minimax-h3");
    form.model = "minimax-h3-fl2va:comfy-pruned-int8";
    useModelStore().all = [
      {
        name: form.model,
        family: form.family,
        downloaded: true,
        source_image: "required",
      } as ModelEntry,
    ];
    const wrapper = mount(InspectorPanel, { props: { form }, attachTo: document.body });
    const field = wrapper.get("[data-test='inspector-source-media']");
    expect(field.find("[data-test='source-well']").exists()).toBe(true);
    expect(field.find("[data-test='source-required-badge']").exists()).toBe(true);
    // Reviewed first-frame-only runtime: no empty last-frame well.
    expect(field.find("[data-test='end-frame-well']").exists()).toBe(false);

    await field.get("[data-test='source-gallery']").trigger("click");
    const picker = wrapper
      .findAllComponents({ name: "ImagePickerModal" })
      .find((candidate) => candidate.props("title") === "First frame");
    if (!picker) throw new Error("H3 first-frame picker not found");
    expect(picker.props("open")).toBe(true);
    picker.vm.$emit("pick", [
      {
        filename: "opening.png",
        base64: "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAIAAAAmkwkpAAAAAElFTkSuQmCC",
      },
    ]);
    await flushPromises();
    expect(form.h3Authoring?.firstFrame).toMatchObject({
      filename: "opening.png",
      width: 7,
      height: 4,
    });
  });

  it("renders the identity photo well right after source media when qualified", () => {
    const form = formFor("flux");
    form.model = "flux-dev:q8";
    form.identitySupported = true;
    const wrapper = mount(InspectorPanel, { props: { form } });
    const field = wrapper.get("[data-test='inspector-identity']");
    expect(field.find("[data-test='identity-photo-well']").exists()).toBe(true);
    // Primary form, immediately below the source wells — never behind Advanced.
    const order = wrapper
      .findAll("[data-test='inspector-source-media'], [data-test='inspector-identity']")
      .map((node) => node.attributes("data-test"));
    expect(order).toEqual(["inspector-source-media", "inspector-identity"]);
  });

  it("hides identity entirely when the checkpoint has not said yes", async () => {
    const form = formFor("flux");
    form.model = "flux-dev:bf16";
    const wrapper = mount(InspectorPanel, { props: { form } });
    // Unread capability: absence is never evidence of support.
    expect(wrapper.find("[data-test='inspector-identity']").exists()).toBe(false);
    form.identitySupported = false;
    await flushPromises();
    expect(wrapper.find("[data-test='inspector-identity']").exists()).toBe(false);
  });

  it("parks a staged photo when the capability is lost, and restores it", async () => {
    const form = formFor("flux");
    form.model = "flux-dev:q8";
    form.identitySupported = true;
    form.identityImage = { filename: "ada.png", base64: "AAAA" };
    const wrapper = mount(InspectorPanel, { props: { form } });
    expect(wrapper.find("[data-test='inspector-identity']").exists()).toBe(true);

    // Switching to an unqualified checkpoint hides the well and retains the
    // photo — nothing is erased, nothing is refused, and `buildRequest` keeps
    // the partition off the wire.
    form.identitySupported = false;
    await flushPromises();
    expect(wrapper.find("[data-test='inspector-identity']").exists()).toBe(false);
    expect(form.identityImage).toEqual({ filename: "ada.png", base64: "AAAA" });
    expect(buildRequest(form).id_image).toBeUndefined();

    // Selecting a qualified checkpoint again brings the well back with the
    // photo still in it.
    form.identitySupported = true;
    await flushPromises();
    expect(wrapper.find("[data-test='inspector-identity']").exists()).toBe(true);
    expect(wrapper.find("[data-test='identity-remove']").exists()).toBe(true);
    expect(form.identityImage).toEqual({ filename: "ada.png", base64: "AAAA" });
  });

  it("keeps identity out of sequence mode", () => {
    useSequenceDraftStore().output = "sequence";
    const form = formFor("flux");
    form.model = "flux-dev:q8";
    form.identitySupported = true;
    const wrapper = mount(InspectorPanel, { props: { form } });
    expect(wrapper.find("[data-test='inspector-identity']").exists()).toBe(false);
  });

  it("exposes H3 Ref2VA references in the primary form", () => {
    const form = formFor("minimax-h3");
    form.model = "minimax-h3-ref2va:comfy-pruned-int8";
    useModelStore().all = [
      { name: form.model, family: form.family, downloaded: true } as ModelEntry,
    ];
    const wrapper = mount(InspectorPanel, { props: { form } });
    expect(wrapper.find("[data-test='inspector-source-media']").exists()).toBe(true);
    expect(wrapper.find("[data-test='h3-reference-files']").exists()).toBe(true);
  });

  it("stands down in sequence mode — the sequence composer owns its opening image", () => {
    useSequenceDraftStore().output = "sequence";
    const wrapper = mount(InspectorPanel, { props: { form: formFor("sd15") } });
    expect(wrapper.find("[data-test='inspector-source-media']").exists()).toBe(false);
  });

  it("hides a parked Sequence opening image from unsupported shape controls", () => {
    const form = formFor("wan");
    form.model = "wan22-t2v-a14b";
    form.sourceImageCapability = "unsupported";
    useModelStore().all = [wanModel("unsupported")];
    const draft = useSequenceDraftStore();
    draft.output = "sequence";
    draft.openingImage = {
      filename: "opening.png",
      base64: "PARKED",
      width: 1024,
      height: 1024,
    };

    const wrapper = mount(InspectorPanel, { props: { form } });

    expect(wrapper.getComponent(ShapePicker).props("options")).not.toEqual(
      expect.arrayContaining([expect.objectContaining({ id: "source" })]),
    );
    expect(wrapper.getComponent(ResolutionSelector).props("status")).not.toContain("source");
    expect(wrapper.find("[data-test='match-source-resolution']").exists()).toBe(false);
    expect(draft.openingImage?.base64).toBe("PARKED");
  });
});

describe("InspectorPanel — model aspect vs source tie", () => {
  it("defaults to the canonical aspect and preserves an explicit Source pick", async () => {
    const model = {
      name: "wan22-ti2v-5b:fp16",
      family: "wan",
      downloaded: true,
      default_width: 1280,
      default_height: 704,
    } as ModelEntry;
    useModelStore().all = [model];
    const form = formFor("wan");
    form.model = model.name;
    form.width = 1280;
    form.height = 704;
    form.sourceImage = "SRC";
    form.sourceImageWidth = 1280;
    form.sourceImageHeight = 704;
    const wrapper = mount(InspectorPanel, { props: { form } });
    const shape = wrapper.getComponent(ShapePicker);
    const canonical = (shape.props("options") as ReadonlyArray<{ id: string }>).find(
      (option) => option.id !== "source",
    )!.id;
    expect(shape.props("modelValue")).toBe(canonical);
    expect(form.width).toBe(1280);
    expect(form.height).toBe(704);

    shape.vm.$emit("update:modelValue", "source");
    await flushPromises();
    expect(wrapper.emitted("canvas-intent")?.at(-1)).toEqual(["source"]);

    // The parent owns the intent; once it flows back the Source chip wins
    // over the canonical family that matches the same canvas.
    await wrapper.setProps({ canvasIntent: "source" });
    expect(shape.props("modelValue")).toBe("source");

    wrapper.unmount();
    const remounted = mount(InspectorPanel, {
      props: { form, canvasIntent: "source" },
    });
    expect(remounted.getComponent(ShapePicker).props("modelValue")).toBe("source");
  });

  it("never badges a manual canvas as source-derived", async () => {
    const model = {
      name: "wan22-ti2v-5b:fp16",
      family: "wan",
      downloaded: true,
      default_width: 1280,
      default_height: 704,
      dimension_alignment: 16,
      recommended_dimensions: [
        { width: 1280, height: 704 },
        { width: 704, height: 704 },
      ],
    } as ModelEntry;
    useModelStore().all = [model];
    const form = formFor("wan");
    form.model = model.name;
    form.sourceImage = "SRC";
    form.sourceImageWidth = 1024;
    form.sourceImageHeight = 1024;
    form.width = 1024;
    form.height = 576;

    const wrapper = mount(InspectorPanel, {
      props: { form, canvasIntent: "manual" },
    });
    const selector = wrapper.getComponent(ResolutionSelector);
    expect(selector.props("customLabel")).toBe("Manual");
    expect(selector.props("status")).toContain("1024×576");
    expect(selector.props("status")).not.toContain("Matches source");
    expect(wrapper.find("[data-test='match-source-resolution']").exists()).toBe(true);

    wrapper.find("[data-test='match-source-resolution']").trigger("click");
    await flushPromises();
    expect(wrapper.emitted("canvas-intent")?.at(-1)).toEqual(["source-exact"]);
    await wrapper.setProps({ canvasIntent: "source-exact" });
    expect(selector.props("customLabel")).toBe("Source");
    expect(selector.props("status")).toContain("Matches source");
  });
});

describe("InspectorPanel — a restored model no machine has", () => {
  it("keeps the recorded model visible with a Not installed tag", async () => {
    useModelStore().all = [];
    const form = formFor("zimage");
    form.model = "z-image-turbo:q6";
    const wrapper = mount(InspectorPanel, { props: { form } });
    await flushPromises();

    expect(wrapper.get('[data-test="selected-model-name"]').text()).toBe("z-image-turbo:q6");
    expect(wrapper.get('[data-test="selected-model-missing"]').text()).toBe("Not installed");
  });

  it("offers the pull for that exact id when its picker row is chosen", async () => {
    useModelStore().all = [];
    const form = formFor("zimage");
    form.model = "z-image-turbo:q6";
    const wrapper = mount(InspectorPanel, { props: { form } });
    await flushPromises();

    await wrapper.get(".ms-model__button").trigger("click");
    await wrapper.get('[data-test="model-option-missing"]').trigger("click");

    expect(wrapper.emitted("pull-missing-model")).toEqual([["z-image-turbo:q6"]]);
    // The raw id is what the form and the request keep carrying.
    expect(form.model).toBe("z-image-turbo:q6");
  });

  it("shows Choose a model only when nothing is selected at all", () => {
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    expect(wrapper.get('[data-test="selected-model-name"]').text()).toBe("Choose a model");
    expect(wrapper.find('[data-test="selected-model-missing"]').exists()).toBe(false);
  });
});

describe("InspectorPanel — File under", () => {
  function connectHost(id: string, organize: boolean) {
    const hosts = useHostsStore();
    hosts.extras.push({
      id,
      label: id,
      url: `http://${id}:7680`,
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    hosts.capabilities[id] = { gallery: { can_delete: true, organize } } as never;
  }

  it("stays hidden while no machine reports gallery.organize", () => {
    connectHost("legacy-7680", false);
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    expect(wrapper.find('[data-test="file-under-group"]').exists()).toBe(false);
  });

  it("stays hidden when the capability snapshot has not been read", () => {
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "unknown-7680",
      label: "unknown",
      url: "http://unknown:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    expect(wrapper.find('[data-test="file-under-group"]').exists()).toBe(false);
  });

  it("renders between the essentials and Advanced once a machine can file", () => {
    connectHost("halcyon-7680", true);
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    const group = wrapper.get('[data-test="file-under-group"]');
    expect(group.text()).toContain("File under");
    const html = wrapper.html();
    const at = (hook: string) => html.indexOf(`data-test="${hook}"`);
    expect(at("seed-mode-random")).toBeLessThan(at("file-under-group"));
    expect(at("file-under-group")).toBeLessThan(at("open-advanced"));
  });

  it("drops the ghost chip when Settings ▸ Library turns auto-tagging off", async () => {
    connectHost("halcyon-7680", true);
    useLibraryPrefsStore().autoTagTitle = false;
    const form = formFor("flux");
    form.title = "Smurf Village";
    const wrapper = mount(InspectorPanel, { props: { form } });
    await flushPromises();
    expect(wrapper.find('[data-test="file-under-group"]').exists()).toBe(true);
    expect(wrapper.find('[data-test="file-under-ghost-tag"]').exists()).toBe(false);
  });

  it("hides when the PINNED machine cannot file, even if another can", async () => {
    connectHost("halcyon-7680", true);
    connectHost("legacy-7680", false);
    useAppPrefsStore().settings = { generateTargetHost: "legacy-7680" } as never;
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    await flushPromises();
    expect(wrapper.find('[data-test="file-under-group"]').exists()).toBe(false);
  });

  it("derives the ghost chip from the Create header title", async () => {
    connectHost("halcyon-7680", true);
    const form = formFor("flux");
    const wrapper = mount(InspectorPanel, { props: { form } });
    expect(wrapper.find('[data-test="file-under-ghost-tag"]').exists()).toBe(false);
    form.title = "Smurf Village";
    await flushPromises();
    expect(wrapper.get('[data-test="file-under-ghost-tag"]').text()).toContain("smurf-village");
  });

  it("keeps the form's collection match in step with the live title", async () => {
    connectHost("halcyon-7680", true);
    libraryListings.collections = [{ id: "c1", name: "Smurf Village", slug: "smurf-village" }];
    const form = formFor("flux");
    const wrapper = mount(InspectorPanel, { props: { form } });
    await flushPromises();
    expect(form.fileUnderMatch).toBeNull();
    form.title = "Smurf Village";
    await flushPromises();
    expect(form.fileUnderMatch).toMatchObject({ name: "Smurf Village", slug: "smurf-village" });
    expect(wrapper.get('[data-test="file-under-collection-match"]').text()).toContain(
      "matched to title",
    );
  });

  it("offers the fleet's own tags as suggestions", async () => {
    connectHost("halcyon-7680", true);
    libraryListings.tags = [{ name: "blue", count: 9 }];
    const wrapper = mount(InspectorPanel, { props: { form: formFor("flux") } });
    await flushPromises();
    await wrapper.get('[data-test="file-under-tag-input"]').setValue("bl");
    expect(wrapper.get('[data-test="file-under-tag-suggestion"]').text()).toContain("blue");
  });

  it("writes tag edits back onto the form", async () => {
    connectHost("halcyon-7680", true);
    const form = formFor("flux");
    const wrapper = mount(InspectorPanel, { props: { form } });
    const input = wrapper.get('[data-test="file-under-tag-input"]');
    await input.setValue("#kodak");
    await input.trigger("keydown.enter");
    expect(form.fileUnder.manualTags).toEqual(["kodak"]);
  });

  it("previews the filename the print will land as", async () => {
    connectHost("halcyon-7680", true);
    const form = formFor("flux");
    form.model = "flux-dev:q8";
    form.title = "Smurf Village";
    const wrapper = mount(InspectorPanel, { props: { form } });
    await flushPromises();
    expect(wrapper.get('[data-test="file-under-filename"]').text()).toMatch(
      /mold-flux-dev-q8-\d+~smurf-village\.png/,
    );
  });
});

const noteCapabilities = {
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

const DISTILLED_NOTE =
  "Distilled recipe fixes CFG at 1.0. Choose a Dev checkpoint with Auto or a guided pipeline to adjust it.";
const H3_GUIDANCE_NOTE =
  "MiniMax H3 does not use classifier-free guidance; guidance is fixed at 0.";
const H3_TURBO_STEPS_NOTE =
  "Fixed by the 8-step Turbo tier: 9 terminal-inclusive sampler grid points (8 denoise intervals).";

/** A minimal advertised v1 profile whose two numeric controls carry exactly
 * the mode and note under test. Defaults mirror the controls because the
 * client validator cross-checks them. */
function noteModel(
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
  } as unknown as ModelEntry;
}
