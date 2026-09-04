import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { reactive } from "vue";

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn().mockResolvedValue({}),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
  },
}));

vi.mock("@studio/api/chains", () => ({
  validateChain: vi.fn(),
}));

import SequenceComposer from "./SequenceComposer.vue";
import ImagePickerModal from "../generate/ImagePickerModal.vue";
import SeamEditor from "@ui/components/SeamEditor.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useContextMenuStore, type MenuItem } from "../../stores/contextMenu";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import type { ModelEntry } from "../../lib/api/types";
import { validateChain } from "@studio/api/chains";

const validateChainMock = vi.mocked(validateChain);

const model = { name: "ltx-video", family: "ltx-video" } as ModelEntry;

const limits: ChainLimits = {
  model: "ltx-video",
  frames_per_clip_cap: 97,
  frames_per_clip_recommended: 25,
  max_stages: 4,
  max_total_frames: 400,
  fade_frames_max: 24,
  transition_modes: ["smooth", "cut", "fade"],
  quantization_family: "fp8",
  supports_audio: false,
  supports_sequence: true,
};

function form(): GenerateForm {
  const f = reactive({ ...newGenerateForm(), family: "ltx-video", model: "ltx-video" });
  f.fps = 24;
  return f;
}

function seedDraft(prompts: string[] = ["clip one", "clip two"]) {
  const draft = useSequenceDraftStore();
  draft.output = "sequence";
  draft.ensureClips(25);
  prompts.forEach((prompt, i) => {
    if (draft.clips[i]) draft.clips[i]!.prompt = prompt;
  });
  return draft;
}

function mountComposer(overrides: Record<string, unknown> = {}) {
  return mount(SequenceComposer, {
    props: { form: form(), selectedModel: model, chainLimits: limits, ...overrides },
    attachTo: document.body,
  });
}

beforeEach(() => {
  validateChainMock.mockReset();
  setActivePinia(createPinia());
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "k" };
  conn.status = "ready";
  useHostsStore().initialized = true;
});
afterEach(() => (document.body.innerHTML = ""));

describe("SequenceComposer — rail", () => {
  it("renders the clip rail from the draft store and adds clips through it", async () => {
    const draft = seedDraft();
    const wrapper = mountComposer();
    expect(wrapper.findAll(".ms-clip")).toHaveLength(2);
    await wrapper.get(".ms-rail__add").trigger("click");
    expect(draft.clips).toHaveLength(3);
  });

  it("opens the seam editor on seam click and writes the transition to the store", async () => {
    const draft = seedDraft();
    const wrapper = mountComposer();
    expect(wrapper.findComponent(SeamEditor).exists()).toBe(false);
    await wrapper.get(".ms-seam").trigger("click");
    const editor = wrapper.getComponent(SeamEditor);
    editor.vm.$emit("update:transition", "cut");
    await flushPromises();
    expect(draft.clips[1]!.transition).toBe("cut");
  });
});

describe("SequenceComposer — active clip editor", () => {
  it("captions the active clip and binds its prompt and frames", async () => {
    const draft = seedDraft();
    draft.activeClipId = draft.clips[1]!.id;
    // A context-capable LTX-2 model: the seam reads "Smooth from clip 1"
    // (LTX-Video's zero motion tail would label it "Join" instead).
    const wrapper = mountComposer({
      selectedModel: { name: "ltx-2-19b-distilled:fp8", family: "ltx2" } as ModelEntry,
    });
    expect(wrapper.get("[data-test='active-clip-caption']").text()).toContain("Scene 2 of 2");
    expect(wrapper.get("[data-test='active-clip-meta']").text()).toContain("Smooth from scene 1");

    await wrapper.get("[data-test='clip-prompt']").setValue("a new beat");
    expect(draft.clips[1]!.prompt).toBe("a new beat");

    const frames = wrapper.get<HTMLSelectElement>("[data-test='clip-frames']");
    // 8n+1 options above the motion tail, capped by chain limits.
    expect(frames.element.options.length).toBeGreaterThan(0);
    await frames.setValue("41");
    expect(draft.clips[1]!.frames).toBe(41);
  });

  it("submits on ⌘↵ from the prompt editor", async () => {
    seedDraft();
    const wrapper = mountComposer();
    await wrapper
      .get("[data-test='clip-prompt']")
      .trigger("keydown", { key: "Enter", metaKey: true });
    expect(wrapper.emitted("submit")).toHaveLength(1);
  });

  it("keeps opening-image controls out of the clip editor", async () => {
    const draft = seedDraft();
    draft.activeClipId = draft.clips[0]!.id;
    const wrapper = mountComposer();
    expect(wrapper.find("[data-test='opening-image-attach']").exists()).toBe(false);
    draft.activeClipId = draft.clips[1]!.id;
    await flushPromises();
    expect(wrapper.find("[data-test='opening-image-attach']").exists()).toBe(false);
  });

  it("does not mount a duplicate opening-image picker", async () => {
    seedDraft();
    const wrapper = mountComposer();
    expect(wrapper.findComponent(ImagePickerModal).exists()).toBe(false);
  });
});

describe("SequenceComposer — footer", () => {
  it("keeps the preparing action responsive and emits cancel", async () => {
    seedDraft();
    const wrapper = mountComposer({ submitting: true });
    const button = wrapper.get("[data-test='generate-sequence']");
    expect(button.attributes("disabled")).toBeUndefined();
    expect(button.text()).toContain("Cancel");
    await button.trigger("click");
    expect(wrapper.emitted("cancel")).toHaveLength(1);
    expect(wrapper.emitted("submit")).toBeUndefined();
  });

  it("discards an in-flight validation when the rendering host changes", async () => {
    let resolveValidation!: (value: Awaited<ReturnType<typeof validateChain>>) => void;
    validateChainMock.mockReturnValue(
      new Promise((resolve) => {
        resolveValidation = resolve;
      }),
    );
    seedDraft();
    const wrapper = mountComposer({
      target: { baseUrl: "http://render-one:7680", apiKey: "one" },
    });
    await wrapper.get("[data-test='sequence-validate']").trigger("click");
    await wrapper.setProps({
      target: { baseUrl: "http://render-two:7680", apiKey: "two" },
    });
    resolveValidation({
      model: "ltx-video",
      width: 512,
      height: 512,
      fps: 24,
      motion_tail_frames: 0,
      stage_count: 2,
      estimated_total_frames: 50,
      estimated_duration_ms: 2_083,
      stages: [],
      warnings: [],
      vram_estimate: null,
    });
    await flushPromises();
    expect(wrapper.find("[data-test='sequence-validation-plan']").exists()).toBe(false);
  });

  it("validates the current sequence on its exact host and clears the plan after edits", async () => {
    validateChainMock.mockResolvedValue({
      model: "ltx-video",
      width: 512,
      height: 512,
      fps: 24,
      motion_tail_frames: 0,
      stage_count: 2,
      estimated_total_frames: 50,
      estimated_duration_ms: 2_083,
      stages: [
        {
          prompt: "clip one",
          frames: 25,
          output_frames: 25,
          transition: "smooth",
          fade_frames: null,
          has_source_image: true,
          has_negative_prompt: false,
        },
        {
          prompt: "clip two",
          frames: 25,
          output_frames: 25,
          transition: "smooth",
          fade_frames: null,
          has_source_image: true,
          has_negative_prompt: true,
        },
      ],
      warnings: ["Join normalized for this checkpoint."],
      vram_estimate: { worst_case_bytes: 12_884_901_888, fits: true },
    });
    const draft = seedDraft();
    draft.clips[1]!.negativePrompt = "camera shake";
    const target = { baseUrl: "http://render-box:7680", apiKey: "secret" };
    const wrapper = mountComposer({ target });

    await wrapper.get("[data-test='sequence-validate']").trigger("click");
    await flushPromises();

    expect(validateChainMock).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "ltx-video",
        stages: [
          expect.objectContaining({ prompt: "clip one" }),
          expect.objectContaining({ prompt: "clip two", negative_prompt: "camera shake" }),
        ],
      }),
      target,
    );
    expect(wrapper.get("[data-test='sequence-validation-plan']").text()).toContain(
      "Validated · 2 scenes · 50f · 2.1s",
    );
    expect(wrapper.get("[data-test='sequence-validation-plan']").text()).toContain("12.0 GiB");
    expect(wrapper.get("[data-test='sequence-validation-plan']").text()).toContain(
      "Join normalized",
    );
    const planText = wrapper
      .get("[data-test='sequence-validation-plan']")
      .text()
      .replace(/\s+/g, " ");
    expect(planText).toContain("Scene 1 · 25f in / 25f out · Join · Opening image");
    expect(planText).toContain("Scene 2 · 25f in / 25f out · Join · Source image");
    expect(wrapper.emitted("submit")).toBeUndefined();

    await wrapper.get("[data-test='clip-prompt']").setValue("edited opening");
    await flushPromises();
    expect(wrapper.find("[data-test='sequence-validation-plan']").exists()).toBe(false);
  });

  it("validates without parked images when the checkpoint does not support them", async () => {
    validateChainMock.mockResolvedValue({
      model: "wan22-t2v-a14b:q4",
      width: 1280,
      height: 720,
      fps: 24,
      motion_tail_frames: 17,
      stage_count: 2,
      estimated_total_frames: 50,
      estimated_duration_ms: 2_083,
      stages: [],
      warnings: [],
      vram_estimate: null,
    });
    const draft = seedDraft();
    draft.openingImage = { filename: "opening.png", base64: "OPENING" };
    draft.clips[1]!.sourceImage = { filename: "second.png", base64: "SECOND" };
    const unsupported = {
      name: "wan22-t2v-a14b:q4",
      family: "wan",
      source_image: "unsupported",
    } as ModelEntry;
    const unsupportedForm = form();
    unsupportedForm.model = unsupported.name;
    unsupportedForm.family = unsupported.family;
    unsupportedForm.sourceImageCapability = "unsupported";
    const target = { baseUrl: "http://render-box:7680", apiKey: "secret" };
    const wrapper = mountComposer({
      form: unsupportedForm,
      selectedModel: unsupported,
      target,
    });

    await wrapper.get("[data-test='sequence-validate']").trigger("click");
    await flushPromises();

    expect(validateChainMock.mock.calls[0]?.[0].stages).toEqual([
      expect.not.objectContaining({ source_image: expect.anything() }),
      expect.not.objectContaining({ source_image: expect.anything() }),
    ]);
    expect(draft.openingImage?.base64).toBe("OPENING");
    expect(draft.clips[1]!.sourceImage?.base64).toBe("SECOND");
  });

  it("disables Generate with the first validation message while a clip is blank", () => {
    seedDraft(["described", ""]);
    const wrapper = mountComposer();
    const button = wrapper.get("[data-test='generate-sequence']");
    expect(button.attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("Describe clip 2 before generating.");
  });

  it("shows the fit note when the sequence validates", () => {
    seedDraft();
    const wrapper = mountComposer();
    expect(wrapper.get("[data-test='sequence-fit']").text()).toContain("✓ fits");
    expect(wrapper.get("[data-test='sequence-fit']").text()).toContain("@ 24fps");
    expect(wrapper.get("[data-test='generate-sequence']").attributes("disabled")).toBeUndefined();
  });

  it("blocks sequence-incapable checkpoints with the server's reason", () => {
    seedDraft();
    const wrapper = mountComposer({
      chainLimits: {
        ...limits,
        supports_sequence: false,
        sequence_unsupported_reason: "Two-stage checkpoint — can't chain clips",
      },
    });
    expect(wrapper.get("[data-test='generate-sequence']").attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("Two-stage checkpoint");
  });

  it("keeps audio out of the composer footer", () => {
    seedDraft();
    expect(mountComposer().find("[data-test='sequence-audio']").exists()).toBe(false);
    document.body.innerHTML = "";
    const withAudio = mountComposer({ chainLimits: { ...limits, supports_audio: true } });
    expect(withAudio.find("[data-test='sequence-audio']").exists()).toBe(false);
  });
});

describe("SequenceComposer — edit sessions", () => {
  function startEditing() {
    const draft = seedDraft();
    draft.loadFromJob(
      {
        jobId: "abcdef1234567890",
        hostId: "local",
        baseline: draft.clips.map((clip) => ({ ...clip })),
        completedStages: 2,
      },
      draft.clips.map((clip) => ({ ...clip })),
      false,
    );
    return draft;
  }

  it("banners the edit with cached/re-render counts and relabels the button", async () => {
    const draft = startEditing();
    const wrapper = mountComposer();
    expect(wrapper.get("[data-test='edit-banner']").text()).toContain("Editing clip abcdef12");
    expect(wrapper.get("[data-test='edit-banner']").text()).toContain(
      "2 cached · 0 will re-render",
    );
    expect(wrapper.get("[data-test='generate-sequence']").text()).toContain("Generate");

    draft.clips[1]!.prompt = "changed beat";
    await flushPromises();
    expect(wrapper.get("[data-test='edit-banner']").text()).toContain(
      "1 cached · 1 will re-render",
    );
  });

  it("Duplicate as new emits duplicate; Discard restores the baseline and stops editing", async () => {
    const draft = startEditing();
    draft.openingImage = { filename: "original.png", base64: "ORIGINAL" };
    draft.enableAudio = true;
    draft.loadFromJob(
      {
        jobId: "abcdef1234567890",
        hostId: "local",
        baseline: draft.clips.map((clip) => ({ ...clip })),
        completedStages: 2,
      },
      draft.clips.map((clip) => ({ ...clip })),
      true,
      draft.openingImage,
    );
    const wrapper = mountComposer();
    await wrapper.get("[data-test='edit-duplicate']").trigger("click");
    expect(wrapper.emitted("duplicate")).toHaveLength(1);

    draft.clips[1]!.prompt = "changed beat";
    draft.openingImage = { filename: "replacement.png", base64: "REPLACEMENT" };
    draft.enableAudio = false;
    await wrapper.get("[data-test='edit-discard']").trigger("click");
    expect(draft.editing).toBeNull();
    expect(draft.clips[1]!.prompt).toBe("clip two");
    expect(draft.openingImage?.filename).toBe("original.png");
    expect(draft.enableAudio).toBe(true);
  });
});

describe("SequenceComposer — clear sequence", () => {
  it("confirms, then resets to two fresh clips and stays in Sequence", async () => {
    const draft = seedDraft(["clip one", "clip two"]);
    draft.addClip(25);
    draft.clips[2]!.prompt = "clip three";
    draft.enableAudio = true;
    const wrapper = mountComposer();

    await wrapper.get("[data-test='sequence-clear']").trigger("click");
    // The confirm dialog teleports to <body>; blunt copy names the count.
    const dialog = document.querySelector("[data-test='confirm-dialog']");
    expect(dialog?.textContent).toContain("Clear the clip?");
    expect(dialog?.textContent).toContain("Removes all 3 scenes");

    (document.querySelector("[data-test='confirm-accept']") as HTMLElement).click();
    await flushPromises();

    expect(draft.clips).toHaveLength(2);
    expect(draft.clips.every((clip) => clip.prompt === "")).toBe(true);
    expect(draft.enableAudio).toBe(false);
    expect(draft.output).toBe("sequence");
    expect(document.querySelector("[data-test='confirm-dialog']")).toBeNull();
  });

  it("cancel keeps every clip", async () => {
    const draft = seedDraft(["clip one", "clip two"]);
    const wrapper = mountComposer();
    await wrapper.get("[data-test='sequence-clear']").trigger("click");
    (document.querySelector("[data-test='confirm-cancel']") as HTMLElement).click();
    await flushPromises();
    expect(draft.clips.map((clip) => clip.prompt)).toEqual(["clip one", "clip two"]);
  });

  it("clearing during an edit session ends the session without emitting", async () => {
    const draft = seedDraft(["clip one", "clip two"]);
    draft.loadFromJob(
      {
        jobId: "job-1",
        hostId: "h1",
        baseline: draft.clips.map((c) => ({ ...c })),
        completedStages: 1,
      },
      draft.clips.map((c) => ({ ...c })),
      false,
    );
    const wrapper = mountComposer();
    await wrapper.get("[data-test='sequence-clear']").trigger("click");
    const dialog = document.querySelector("[data-test='confirm-dialog']");
    expect(dialog?.textContent).toContain("Ends the edit session");
    (document.querySelector("[data-test='confirm-accept']") as HTMLElement).click();
    await flushPromises();
    expect(draft.editing).toBeNull();
    expect(wrapper.emitted("submit")).toBeUndefined();
    expect(wrapper.emitted("duplicate")).toBeUndefined();
  });
});

describe("SequenceComposer — TOML import", () => {
  it("restores opening-image strength and clears stale maskless fit state", async () => {
    seedDraft();
    const liveForm = form();
    liveForm.strength = 0.75;
    liveForm.sourceFit = {
      mode: "upscale-then-fit",
      upscalerModel: "upscaler",
      fit: { mode: "crop-fill" },
    };
    const wrapper = mountComposer({ form: liveForm });
    const toml = [
      'schema = "mold.chain.v1"',
      "[chain]",
      'model = "ltx-video"',
      "strength = 0.4",
      "[[stage]]",
      'prompt = "opening"',
      "frames = 25",
      'source_image_b64 = "aGk="',
      "[[stage]]",
      'prompt = "ending"',
      "frames = 25",
    ].join("\n");
    const { importTomlText } = wrapper.vm as unknown as {
      importTomlText?: (text: string, filename?: string) => Promise<void>;
    };
    if (!importTomlText) throw new Error("SequenceComposer did not expose importTomlText");
    await importTomlText(toml);
    await flushPromises();

    expect(liveForm.strength).toBe(0.4);
    expect(liveForm.sourceFit).toEqual({ mode: "crop-fill" });
    expect(useSequenceDraftStore().openingImage?.base64).toBe("aGk=");
  });
});

/**
 * Wan's clip grid is its own: the family's VAE compresses time by 4, so its
 * clips are `4k+1` and the LTX `8k+1` ladder hid every value wan actually
 * routes to — including its 53-frame auto-chaining default (#783).
 */
describe("SequenceComposer — wan clip grid", () => {
  const wanModel = {
    name: "wan22-i2v-a14b:q5",
    family: "wan",
    source_image: "required",
  } as ModelEntry;

  it("offers wan durations on the 4k+1 grid", async () => {
    const draft = seedDraft();
    draft.activeClipId = draft.clips[1]!.id;
    const wanForm = reactive({
      ...newGenerateForm(),
      family: "wan",
      model: "wan22-i2v-a14b:q5",
    }) as GenerateForm;
    const wrapper = mountComposer({
      form: wanForm,
      selectedModel: wanModel,
      chainLimits: { ...limits, model: "wan22-i2v-a14b:q5", frames_per_clip_cap: 121 },
    });

    const frames = wrapper.get<HTMLSelectElement>("[data-test='clip-frames']");
    const values = Array.from(frames.element.options).map((option) => Number(option.value));
    expect(values).toContain(53);
    for (const value of values) expect((value - 1) % 4).toBe(0);
    await flushPromises();
  });
});

/**
 * Right-click on the bench: the clip pills get their own reorder/duplicate
 * menu and the rail background gets the bench actions. Both go through the
 * app-wide context-menu store, and both share `studio/lib/sequenceContextMenu`
 * with web so the two surfaces cannot drift.
 */
describe("SequenceComposer — context menus", () => {
  const rightClick = { clientX: 40, clientY: 60 };

  function menuLabels() {
    return useContextMenuStore()
      .entries.filter((entry): entry is MenuItem => !("separator" in entry))
      .map((entry) => entry.label);
  }

  function menuItem(label: string): MenuItem {
    const found = useContextMenuStore().entries.find(
      (entry): entry is MenuItem => !("separator" in entry) && entry.label === label,
    );
    if (!found) throw new Error(`no context-menu item labelled ${label}`);
    return found;
  }

  it("opens the clip menu on a clip pill and makes that clip active", async () => {
    const draft = seedDraft();
    draft.activeClipId = draft.clips[0]!.id;
    const wrapper = mountComposer();
    const menu = useContextMenuStore();

    await wrapper.findAll("[data-clip-id]")[1]!.trigger("contextmenu", rightClick);

    expect(menu.visible).toBe(true);
    expect(draft.activeClipId).toBe(draft.clips[1]!.id);
    expect(menuLabels()).toEqual([
      "Duplicate clip",
      "Insert clip before",
      "Insert clip after",
      "Move to start",
      "Move left",
      "Move right",
      "Move to end",
      "Remove clip",
    ]);
  });

  it("duplicates, inserts, moves, and removes through the clip menu", async () => {
    const draft = seedDraft(["one", "two"]);
    const wrapper = mountComposer();

    await wrapper.findAll("[data-clip-id]")[0]!.trigger("contextmenu", rightClick);
    menuItem("Duplicate clip").action!();
    await flushPromises();
    expect(draft.clips).toHaveLength(3);
    expect(draft.clips[1]!.prompt).toBe("one");

    await wrapper.findAll("[data-clip-id]")[0]!.trigger("contextmenu", rightClick);
    menuItem("Insert clip after").action!();
    await flushPromises();
    expect(draft.clips).toHaveLength(4);
    expect(draft.clips[1]!.prompt).toBe("");

    const moved = draft.clips[0]!.id;
    await wrapper.findAll("[data-clip-id]")[0]!.trigger("contextmenu", rightClick);
    menuItem("Move to end").action!();
    await flushPromises();
    expect(draft.clips[draft.clips.length - 1]!.id).toBe(moved);

    const removed = draft.clips[0]!.id;
    await wrapper.findAll("[data-clip-id]")[0]!.trigger("contextmenu", rightClick);
    menuItem("Remove clip").action!();
    await flushPromises();
    expect(draft.clips.some((clip) => clip.id === removed)).toBe(false);
  });

  it("disables Remove clip at the two-clip floor", async () => {
    seedDraft();
    const wrapper = mountComposer();
    await wrapper.findAll("[data-clip-id]")[0]!.trigger("contextmenu", rightClick);
    expect(menuItem("Remove clip").disabled).toBe(true);
    expect(menuItem("Remove clip").danger).toBe(true);
  });

  it("opens the rail menu on the bench background and adds a clip", async () => {
    const draft = seedDraft();
    const wrapper = mountComposer();
    const menu = useContextMenuStore();

    await wrapper.get(".ms-rail").trigger("contextmenu", rightClick);

    expect(menu.visible).toBe(true);
    expect(menuLabels()).toEqual([
      "Add clip",
      "Validate plan",
      "Import TOML…",
      "Export TOML",
      "Copy TOML",
      "Clear sequence",
    ]);
    menuItem("Add clip").action!();
    expect(draft.clips).toHaveLength(3);
  });

  it("leaves the prompt textarea and the seam pill alone", async () => {
    seedDraft();
    const wrapper = mountComposer();
    const menu = useContextMenuStore();

    await wrapper.get("[data-test='clip-prompt']").trigger("contextmenu", rightClick);
    expect(menu.visible).toBe(false);

    // The seam's own right-click opens the transition editor instead.
    await wrapper.get(".ms-seam").trigger("contextmenu", rightClick);
    expect(menu.visible).toBe(false);
  });
});
