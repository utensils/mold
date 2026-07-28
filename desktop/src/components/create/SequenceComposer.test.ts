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

import SequenceComposer from "./SequenceComposer.vue";
import SeamEditor from "@ui/components/SeamEditor.vue";
import { newGenerateForm, type GenerateForm } from "../../lib/generateForm";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import type { ModelEntry } from "../../lib/api/types";

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
    // (LTX-Video's zero motion tail would label it "Join clips" instead).
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

  it("offers the opening-image attach only on clip 1", async () => {
    const draft = seedDraft();
    draft.activeClipId = draft.clips[0]!.id;
    const wrapper = mountComposer();
    expect(wrapper.find("[data-test='opening-image-attach']").exists()).toBe(true);
    draft.activeClipId = draft.clips[1]!.id;
    await flushPromises();
    expect(wrapper.find("[data-test='opening-image-attach']").exists()).toBe(false);
  });
});

describe("SequenceComposer — footer", () => {
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

  it("gates the audio toggle on chain-limits support", () => {
    seedDraft();
    expect(mountComposer().find("[data-test='sequence-audio']").exists()).toBe(false);
    document.body.innerHTML = "";
    const withAudio = mountComposer({ chainLimits: { ...limits, supports_audio: true } });
    expect(withAudio.find("[data-test='sequence-audio']").exists()).toBe(true);
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
    const wrapper = mountComposer();
    await wrapper.get("[data-test='edit-duplicate']").trigger("click");
    expect(wrapper.emitted("duplicate")).toHaveLength(1);

    draft.clips[1]!.prompt = "changed beat";
    await wrapper.get("[data-test='edit-discard']").trigger("click");
    expect(draft.editing).toBeNull();
    expect(draft.clips[1]!.prompt).toBe("clip two");
  });
});
