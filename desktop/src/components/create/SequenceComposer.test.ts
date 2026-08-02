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
    expect(wrapper.get("[data-test='active-clip-caption']").text()).toContain("CLIP 2 OF 2");
    expect(wrapper.get("[data-test='active-clip-meta']").text()).toContain("Smooth from clip 1");

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
          has_source_image: false,
          has_negative_prompt: false,
        },
        {
          prompt: "clip two",
          frames: 25,
          output_frames: 25,
          transition: "smooth",
          fade_frames: null,
          has_source_image: false,
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
      "Validated · 2 clips · 50f · 2.1s",
    );
    expect(wrapper.get("[data-test='sequence-validation-plan']").text()).toContain("12.0 GiB");
    expect(wrapper.get("[data-test='sequence-validation-plan']").text()).toContain(
      "Join normalized",
    );
    expect(wrapper.emitted("submit")).toBeUndefined();

    await wrapper.get("[data-test='clip-prompt']").setValue("edited opening");
    await flushPromises();
    expect(wrapper.find("[data-test='sequence-validation-plan']").exists()).toBe(false);
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
    expect(wrapper.get("[data-test='edit-banner']").text()).toContain("Editing sequence abcdef12");
    expect(wrapper.get("[data-test='edit-banner']").text()).toContain(
      "2 cached · 0 will re-render",
    );
    expect(wrapper.get("[data-test='generate-sequence']").text()).toContain("Update sequence");

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
    expect(dialog?.textContent).toContain("Clear sequence?");
    expect(dialog?.textContent).toContain("Removes all 3 clips");

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
