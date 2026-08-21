import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia, type Pinia } from "pinia";
import SequenceComposer from "./SequenceComposer.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { parseChainScript } from "@studio/lib/chainToml";
import type { SequenceSharedParams } from "@studio/lib/sequenceForm";
import * as api from "../api";
import { settleConfirm } from "../lib/toasts";

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  fetchChainLimits: vi.fn(),
  validateChain: vi.fn(),
  listGallery: vi.fn(async () => []),
}));

const fetchChainLimitsMock = vi.mocked(api.fetchChainLimits);
const validateChainMock = vi.mocked(api.validateChain);

function ltx2Limits(overrides: Partial<api.ChainLimits> = {}): api.ChainLimits {
  return {
    model: "ltx-2-19b-distilled:fp8",
    frames_per_clip_cap: 97,
    frames_per_clip_recommended: 97,
    max_stages: 16,
    max_total_frames: 97 * 16,
    fade_frames_max: 32,
    transition_modes: ["smooth", "cut", "fade"],
    quantization_family: "fp8",
    supports_audio: true,
    supports_sequence: true,
    ...overrides,
  };
}

function shared(): SequenceSharedParams {
  return {
    model: "ltx-2-19b-distilled:fp8",
    family: "ltx2",
    width: 1216,
    height: 704,
    fps: 24,
    steps: 8,
    guidance: 3,
    strength: 1,
    seed: "",
  };
}

function mountComposer(overrides: Record<string, unknown> = {}) {
  return mount(SequenceComposer, {
    props: {
      model: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      shared: shared(),
      modelDefaultFrames: 97,
      ...overrides,
    },
    global: { plugins: [pinia] },
  });
}

let pinia: Pinia;

describe("SequenceComposer", () => {
  beforeEach(() => {
    localStorage.clear();
    pinia = createPinia();
    setActivePinia(pinia);
    fetchChainLimitsMock.mockReset();
    fetchChainLimitsMock.mockResolvedValue(ltx2Limits());
    validateChainMock.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("starts with two required clips and a disabled Generate sequence button", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    expect(store.clips).toHaveLength(2);
    const button = wrapper.get("[data-test='sequence-generate']");
    expect(button.text()).toBe("Generate sequence");
    expect(button.attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("Describe clip 1");
  });

  it("keeps the preparing action responsive and emits cancel", async () => {
    const store = useSequenceDraftStore();
    store.ensureClips(97);
    store.clips[0]!.prompt = "opening";
    store.clips[1]!.prompt = "ending";
    const wrapper = mountComposer({ submitting: true });
    await flushPromises();
    const button = wrapper.get("[data-test='sequence-generate']");
    expect(button.attributes("disabled")).toBeUndefined();
    expect(button.text()).toContain("Cancel · Preparing sequence");
    await button.trigger("click");
    expect(wrapper.emitted("cancel")).toHaveLength(1);
    expect(wrapper.emitted("submit")).toBeUndefined();
  });

  it("resets model-owned clip lengths and fetches limits at the active fps", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.clips[0]!.frames = 481;
    store.clips[1]!.frames = 53;

    fetchChainLimitsMock.mockResolvedValueOnce(
      ltx2Limits({
        model: "wan22-i2v-a14b:q5",
        frames_per_clip_cap: 257,
        frames_per_clip_recommended: 53,
      }),
    );
    await wrapper.setProps({
      model: "wan22-i2v-a14b:q5",
      family: "wan",
      modelDefaultFrames: 53,
      shared: {
        ...shared(),
        model: "wan22-i2v-a14b:q5",
        family: "wan",
        fps: 16,
      },
    });
    await flushPromises();

    expect(fetchChainLimitsMock).toHaveBeenLastCalledWith(
      "wan22-i2v-a14b:q5",
      undefined,
      16,
    );
    expect(store.clips.map((clip) => clip.frames)).toEqual([53, 53]);
  });

  it("detaches an amend session before switching its model authority", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.loadFromJob(
      {
        jobId: "job-model-a",
        hostId: "origin",
        baseline: store.clips.map((clip) => ({ ...clip })),
        completedStages: 0,
      },
      store.clips.map((clip) => ({ ...clip })),
      false,
    );

    fetchChainLimitsMock.mockResolvedValueOnce(
      ltx2Limits({
        model: "wan22-i2v-a14b:q5",
        frames_per_clip_cap: 257,
        frames_per_clip_recommended: 53,
      }),
    );
    await wrapper.setProps({
      model: "wan22-i2v-a14b:q5",
      family: "wan",
      modelDefaultFrames: 53,
      shared: {
        ...shared(),
        model: "wan22-i2v-a14b:q5",
        family: "wan",
        fps: 16,
      },
    });
    await flushPromises();

    expect(store.editing).toBeNull();
    expect(store.clips.map((clip) => clip.frames)).toEqual([53, 53]);
  });

  // An opening image conditions clip 1 and every later clip inherits the
  // previous clip's motion tail, so a promptless-capable family can render an
  // undescribed sequence — the same rule the one-shot composer applies.
  it("allows undescribed clips once an opening image conditions the sequence", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    expect(
      wrapper.get("[data-test='sequence-generate']").attributes("disabled"),
    ).toBeDefined();

    store.openingImage = { filename: "opening.png", base64: "QUJD" };
    await flushPromises();
    expect(
      wrapper.get("[data-test='sequence-generate']").attributes("disabled"),
    ).toBeUndefined();
  });

  it("keeps clip prompts required for a family that cannot render undescribed", async () => {
    const wrapper = mountComposer({ family: "ltx-video-unknown" });
    await flushPromises();
    const store = useSequenceDraftStore();
    store.openingImage = { filename: "opening.png", base64: "QUJD" };
    await flushPromises();
    expect(
      wrapper.get("[data-test='sequence-generate']").attributes("disabled"),
    ).toBeDefined();
  });

  it("gives the filmstrip popover an explicit full-width flex wrapper", async () => {
    const wrapper = mountComposer();
    await flushPromises();

    const railWrap = wrapper.find(".sq-filmstrip-wrap");
    expect(railWrap.find(".ms-popover").exists()).toBe(true);
    expect(railWrap.find(".ms-popover__trigger").exists()).toBe(true);
    expect(railWrap.find("[aria-label='Sequence filmstrip']").exists()).toBe(
      true,
    );
  });

  it("edits the ACTIVE clip's prompt through the textarea", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    await wrapper.get("[data-test='clip-prompt']").setValue("a heron lifts");
    expect(store.clips[0]?.prompt).toBe("a heron lifts");

    const second = store.clips[1];
    store.activeClipId = second?.id ?? null;
    await flushPromises();
    await wrapper.get("[data-test='clip-prompt']").setValue("it lands");
    expect(store.clips[1]?.prompt).toBe("it lands");
    expect(store.clips[0]?.prompt).toBe("a heron lifts");
    expect(wrapper.text()).toContain("CLIP 2 OF 2");
  });

  it("submits on ⌘↵ once every clip has a prompt", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.clips.forEach((clip, i) => (clip.prompt = `clip ${i + 1}`));
    await flushPromises();
    await wrapper
      .get("[data-test='clip-prompt']")
      .trigger("keydown", { key: "Enter", metaKey: true });
    expect(wrapper.emitted("submit")).toHaveLength(1);
  });

  it("opens the seam editor from the seam pill and applies a transition", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    await wrapper.get(".ms-seam").trigger("click");
    const editor = wrapper.getComponent({ name: "SeamEditor" });
    const cutRow = editor
      .findAll("button")
      .find((b) => b.text().includes("Cut"))!;
    await cutRow.trigger("click");
    expect(store.clips[1]?.transition).toBe("cut");
  });

  it("keeps sequence audio out of the composer footer", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    expect(wrapper.find("[data-test='sequence-enable-audio']").exists()).toBe(
      false,
    );
  });

  it("shows the sequence_unsupported_reason inline and disables Generate", async () => {
    fetchChainLimitsMock.mockResolvedValue(
      ltx2Limits({
        supports_sequence: false,
        sequence_unsupported_reason: "Two-stage dev checkpoints cannot chain.",
      }),
    );
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.clips.forEach((clip, i) => (clip.prompt = `clip ${i + 1}`));
    await flushPromises();
    expect(wrapper.text()).toContain("Two-stage dev checkpoints cannot chain.");
    expect(
      wrapper.get("[data-test='sequence-generate']").attributes("disabled"),
    ).toBeDefined();
  });

  it("shows the duration fit note from the live shared fps", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    // 97 + (97 − 17 tail) = 177 frames at 24 fps.
    expect(wrapper.get("[data-test='sequence-fit-note']").text()).toContain(
      "2 clips",
    );
    expect(wrapper.get("[data-test='sequence-fit-note']").text()).toContain(
      "177f · 7.4s",
    );
  });

  it("validates the current sequence on its exact host and renders the normalized plan", async () => {
    validateChainMock.mockResolvedValue({
      model: "ltx-2-19b-distilled:fp8",
      width: 1216,
      height: 704,
      fps: 24,
      motion_tail_frames: 17,
      stage_count: 2,
      estimated_total_frames: 177,
      estimated_duration_ms: 7_375,
      stages: [
        {
          prompt: "clip 1",
          frames: 97,
          output_frames: 97,
          transition: "smooth",
          fade_frames: null,
          has_source_image: true,
          has_negative_prompt: false,
        },
        {
          prompt: "clip 2",
          frames: 97,
          output_frames: 80,
          transition: "smooth",
          fade_frames: null,
          has_source_image: false,
          has_negative_prompt: true,
        },
      ],
      warnings: ["Opening transition normalized to Continue motion."],
      vram_estimate: {
        worst_case_bytes: 12_884_901_888,
        fits: true,
      },
    });
    const target = {
      baseUrl: "http://render-box:7680",
      apiKey: "secret",
    };
    const wrapper = mountComposer({ target });
    await flushPromises();
    const store = useSequenceDraftStore();
    store.clips.forEach((clip, i) => {
      clip.prompt = `clip ${i + 1}`;
      if (i === 1) clip.negativePrompt = "camera shake";
    });
    store.openingImage = { filename: "opening.png", base64: "AAAA" };
    await flushPromises();

    await wrapper.get("[data-test='sequence-validate']").trigger("click");
    await flushPromises();

    expect(validateChainMock).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "ltx-2-19b-distilled:fp8",
        width: 1216,
        height: 704,
        stages: [
          expect.objectContaining({
            prompt: "clip 1",
            source_image: "AAAA",
          }),
          expect.objectContaining({
            prompt: "clip 2",
            negative_prompt: "camera shake",
          }),
        ],
      }),
      target,
    );
    const plan = wrapper.get("[data-test='sequence-validation-plan']");
    expect(plan.text()).toContain("Validated · 2 clips · 177f · 7.4s");
    expect(plan.text()).toContain("Clip 1");
    expect(plan.text()).toContain("97f output");
    expect(plan.text()).toContain("Opening image");
    expect(plan.text()).toContain("VRAM");
    expect(plan.text()).toContain("12.0 GiB");
    expect(plan.text()).toContain("fits");
    expect(plan.text()).toContain("Opening transition normalized");
    expect(wrapper.emitted("submit")).toBeUndefined();
  });

  it("shows server validation errors inline and clears stale plans after edits", async () => {
    validateChainMock
      .mockResolvedValueOnce({
        model: "ltx-2-19b-distilled:fp8",
        width: 1216,
        height: 704,
        fps: 24,
        motion_tail_frames: 17,
        stage_count: 2,
        estimated_total_frames: 177,
        estimated_duration_ms: 7_375,
        stages: [
          {
            prompt: "clip 1",
            frames: 97,
            output_frames: 97,
            transition: "smooth",
            fade_frames: null,
            has_source_image: false,
            has_negative_prompt: false,
          },
          {
            prompt: "clip 2",
            frames: 97,
            output_frames: 80,
            transition: "smooth",
            fade_frames: null,
            has_source_image: false,
            has_negative_prompt: false,
          },
        ],
        warnings: [],
        vram_estimate: null,
      })
      .mockRejectedValueOnce(new Error("motion tail exceeds clip 2"));
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.clips.forEach((clip, i) => (clip.prompt = `clip ${i + 1}`));
    await flushPromises();

    await wrapper.get("[data-test='sequence-validate']").trigger("click");
    await flushPromises();
    expect(
      wrapper.find("[data-test='sequence-validation-plan']").exists(),
    ).toBe(true);

    await wrapper.get("[data-test='clip-prompt']").setValue("edited opening");
    await flushPromises();
    expect(
      wrapper.find("[data-test='sequence-validation-plan']").exists(),
    ).toBe(false);

    await wrapper.get("[data-test='sequence-validate']").trigger("click");
    await flushPromises();
    expect(
      wrapper.get("[data-test='sequence-validation-error']").text(),
    ).toContain("motion tail exceeds clip 2");
    expect(wrapper.emitted("submit")).toBeUndefined();
  });

  it("discards an in-flight result when a same-sized source payload changes", async () => {
    let resolveValidation!: (value: api.ChainValidationResponse) => void;
    validateChainMock.mockReturnValue(
      new Promise((resolve) => {
        resolveValidation = resolve;
      }),
    );
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.clips.forEach((clip, i) => (clip.prompt = `clip ${i + 1}`));
    store.clips[0]!.sourceImage = {
      filename: "opening.png",
      base64: "AAAA",
    };
    await flushPromises();

    await wrapper.get("[data-test='sequence-validate']").trigger("click");
    store.clips[0]!.sourceImage = {
      filename: "opening.png",
      base64: "BBBB",
    };
    await flushPromises();
    resolveValidation({
      model: "ltx-2-19b-distilled:fp8",
      width: 1216,
      height: 704,
      fps: 24,
      motion_tail_frames: 17,
      stage_count: 2,
      estimated_total_frames: 177,
      estimated_duration_ms: 7_375,
      stages: [],
      warnings: [],
      vram_estimate: null,
    });
    await flushPromises();

    expect(
      wrapper.find("[data-test='sequence-validation-plan']").exists(),
    ).toBe(false);
  });

  it("copies TOML built from the LIVE shared params", async () => {
    const writeText = vi.fn(async (_text: string) => undefined);
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.clips.forEach((clip, i) => (clip.prompt = `clip ${i + 1}`));
    await wrapper.get("[data-test='sequence-file-tools']").trigger("click");
    // The file-tools menu lives in a popover teleported to <body>.
    document
      .querySelector<HTMLElement>("[data-test='sequence-copy-toml']")!
      .click();
    await flushPromises();
    const toml = writeText.mock.calls[0]?.[0] ?? "";
    expect(toml).toContain("width = 1216");
    expect(toml).toContain('model = "ltx-2-19b-distilled:fp8"');
    const parsed = parseChainScript(toml);
    expect(parsed.stages).toHaveLength(2);
  });

  it("imports a TOML script into the draft and emits the shared params", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const toml = [
      'schema = "mold.chain.v1"',
      "[chain]",
      'model = "ltx-video"',
      "width = 768",
      "height = 512",
      "fps = 24",
      "steps = 30",
      "guidance = 5.0",
      "motion_tail_frames = 0",
      "[[stage]]",
      'prompt = "first"',
      "frames = 25",
      "[[stage]]",
      'prompt = "second"',
      "frames = 25",
      'transition = "cut"',
    ].join("\n");
    const file = new File([toml], "chain.toml", { type: "application/toml" });
    await wrapper
      .getComponent({ name: "SequenceComposer" })
      .vm.importTomlText(await file.text());
    await flushPromises();
    const store = useSequenceDraftStore();
    expect(store.clips).toHaveLength(2);
    expect(store.clips[0]?.prompt).toBe("first");
    expect(store.clips[1]?.transition).toBe("cut");
    const emitted = wrapper.emitted("import-shared");
    expect(emitted).toHaveLength(1);
    expect(emitted?.[0]?.[0]).toMatchObject({ width: 768, steps: 30 });
  });

  it("switches to Update sequence with edit-session recovery actions while editing", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.clips.forEach((clip, i) => (clip.prompt = `clip ${i + 1}`));
    store.loadFromJob(
      {
        jobId: "job-9",
        hostId: "origin",
        baseline: store.clips.map((c) => ({ ...c })),
        completedStages: 1,
      },
      store.clips.map((c) => ({ ...c })),
      false,
    );
    await flushPromises();
    expect(wrapper.get("[data-test='sequence-generate']").text()).toBe(
      "Update sequence",
    );
    expect(wrapper.find("[data-test='sequence-edit-banner']").exists()).toBe(
      true,
    );
    await wrapper.get("[data-test='sequence-duplicate']").trigger("click");
    expect(wrapper.emitted("duplicate-as-new")).toHaveLength(1);
    await wrapper.get("[data-test='sequence-discard']").trigger("click");
    expect(wrapper.emitted("discard-edit")).toHaveLength(1);
  });

  it("offers only frame counts strictly above the motion tail", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const options = wrapper
      .get("[data-test='clip-frames']")
      .findAll("option")
      .map((o) => Number(o.element.value));
    expect(options.length).toBeGreaterThan(0);
    expect(Math.min(...options)).toBeGreaterThan(17);
    expect(Math.max(...options)).toBeLessThanOrEqual(97);
  });

  it("keeps opening-image controls out of the clip editor", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    expect(wrapper.find("[data-test='opening-image-attach']").exists()).toBe(
      false,
    );
  });
});

describe("clear sequence", () => {
  beforeEach(() => {
    localStorage.clear();
    pinia = createPinia();
    setActivePinia(pinia);
    fetchChainLimitsMock.mockReset();
    fetchChainLimitsMock.mockResolvedValue(ltx2Limits());
  });

  it("clears to two fresh clips after the app-frame confirm", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.output = "sequence";
    store.addClip(97);
    store.clips.forEach((clip, i) => (clip.prompt = `clip ${i + 1}`));

    await wrapper.get("[data-test='sequence-clear']").trigger("click");
    // Still intact until the dialog is answered.
    expect(store.clips).toHaveLength(3);
    settleConfirm(true);
    await flushPromises();

    expect(store.clips).toHaveLength(2);
    expect(store.clips.every((clip) => clip.prompt === "")).toBe(true);
    expect(store.output).toBe("sequence");
  });

  it("a declined confirm keeps every clip", async () => {
    fetchChainLimitsMock.mockResolvedValue(ltx2Limits());
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.clips.forEach((clip, i) => (clip.prompt = `clip ${i + 1}`));

    await wrapper.get("[data-test='sequence-clear']").trigger("click");
    settleConfirm(false);
    await flushPromises();

    expect(store.clips.map((clip) => clip.prompt)).toEqual([
      "clip 1",
      "clip 2",
    ]);
  });
});

/**
 * Wan's clip grid and seam are both per checkpoint (#783). The composer took
 * neither: it enumerated the LTX `8k+1` durations for every family — so wan's
 * own 53-frame routing default was not even selectable — and read the motion
 * tail from bare name/family strings, which made an image-conditioned
 * checkpoint's real handoff look like LTX-Video's "Join".
 */
describe("SequenceComposer — wan", () => {
  beforeEach(() => {
    localStorage.clear();
    pinia = createPinia();
    setActivePinia(pinia);
    fetchChainLimitsMock.mockReset();
    validateChainMock.mockReset();
  });
  afterEach(() => vi.restoreAllMocks());

  function wanLimits(): api.ChainLimits {
    return ltx2Limits({
      model: "wan22-i2v-a14b:q5",
      frames_per_clip_cap: 121,
      frames_per_clip_recommended: 53,
      supports_audio: false,
    });
  }

  function mountWan(sourceImage: string | null) {
    fetchChainLimitsMock.mockResolvedValue(wanLimits());
    return mountComposer({
      model: "wan22-i2v-a14b:q5",
      family: "wan",
      sourceImage,
      modelDefaultFrames: 53,
      shared: {
        ...shared(),
        model: "wan22-i2v-a14b:q5",
        family: "wan",
        fps: 16,
      },
    });
  }

  it("offers wan clip durations on its own 4k+1 grid", async () => {
    const wrapper = mountWan("required");
    await flushPromises();
    const frames = wrapper
      .get("[data-test='clip-frames']")
      .findAll("option")
      .map((option) => Number(option.attributes("value")));
    expect(frames).toContain(53);
    for (const value of frames) expect((value - 1) % 4).toBe(0);
  });

  it("names the seam from the checkpoint's own conditioning contract", async () => {
    const conditioned = mountWan("required");
    await flushPromises();
    expect(conditioned.get(".ms-seam").attributes("aria-label")).toBe(
      "Transition: Smooth",
    );

    // A text-to-video checkpoint genuinely joins end to end.
    const unconditioned = mountWan("unsupported");
    await flushPromises();
    expect(unconditioned.get(".ms-seam").attributes("aria-label")).toBe(
      "Transition: Join",
    );
  });

  it("resets stale off-grid durations when the model authority changes", async () => {
    fetchChainLimitsMock.mockResolvedValue(ltx2Limits());
    const store = useSequenceDraftStore();
    store.clearSequence(53);
    store.clips[0]!.frames = 53;
    store.clips[1]!.frames = 53;
    store.activeClipId = store.clips[0]!.id;

    const wrapper = mountComposer();
    await flushPromises();
    const frames = wrapper
      .get("[data-test='clip-frames']")
      .findAll("option")
      .map((option) => Number(option.attributes("value")));
    expect(store.clips.map((clip) => clip.frames)).toEqual([97, 97]);
    expect(frames).not.toContain(53);
    expect(frames).toEqual([...frames].sort((a, b) => a - b));
    expect(frames).toContain(97);
  });
});

/**
 * Right-click on the bench. Web has no app-wide menu component, so the
 * composer renders its own inline `role="menu"` panel — the entries and their
 * disabled rules are the desktop ones, from the shared builder.
 */
describe("SequenceComposer — context menus", () => {
  const rightClick = { clientX: 40, clientY: 60 };

  beforeEach(() => {
    localStorage.clear();
    pinia = createPinia();
    setActivePinia(pinia);
    fetchChainLimitsMock.mockReset();
    fetchChainLimitsMock.mockResolvedValue(ltx2Limits());
    validateChainMock.mockReset();
  });
  afterEach(() => vi.restoreAllMocks());

  function itemLabels(wrapper: ReturnType<typeof mountComposer>) {
    return wrapper
      .findAll("[data-test='sequence-context-item']")
      .map((item) => item.text());
  }

  async function clickItem(
    wrapper: ReturnType<typeof mountComposer>,
    label: string,
  ) {
    const item = wrapper
      .findAll("[data-test='sequence-context-item']")
      .find((candidate) => candidate.text() === label);
    if (!item) throw new Error(`no context-menu item labelled ${label}`);
    await item.trigger("click");
    await flushPromises();
  }

  it("opens the clip menu on a clip pill and makes that clip active", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.activeClipId = store.clips[0]!.id;

    await wrapper
      .findAll("[data-clip-id]")[1]!
      .trigger("contextmenu", rightClick);

    expect(wrapper.find("[data-test='sequence-context-menu']").exists()).toBe(
      true,
    );
    expect(store.activeClipId).toBe(store.clips[1]!.id);
    expect(itemLabels(wrapper)).toEqual([
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
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    store.clips[0]!.prompt = "one";

    await wrapper
      .findAll("[data-clip-id]")[0]!
      .trigger("contextmenu", rightClick);
    await clickItem(wrapper, "Duplicate clip");
    expect(store.clips).toHaveLength(3);
    expect(store.clips[1]!.prompt).toBe("one");
    // Acting on an item closes the panel.
    expect(wrapper.find("[data-test='sequence-context-menu']").exists()).toBe(
      false,
    );

    await wrapper
      .findAll("[data-clip-id]")[0]!
      .trigger("contextmenu", rightClick);
    await clickItem(wrapper, "Insert clip after");
    expect(store.clips).toHaveLength(4);
    expect(store.clips[1]!.prompt).toBe("");

    const moved = store.clips[0]!.id;
    await wrapper
      .findAll("[data-clip-id]")[0]!
      .trigger("contextmenu", rightClick);
    await clickItem(wrapper, "Move to end");
    expect(store.clips[store.clips.length - 1]!.id).toBe(moved);

    const removed = store.clips[0]!.id;
    await wrapper
      .findAll("[data-clip-id]")[0]!
      .trigger("contextmenu", rightClick);
    await clickItem(wrapper, "Remove clip");
    expect(store.clips.some((clip) => clip.id === removed)).toBe(false);
  });

  it("disables Remove clip at the two-clip floor", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    await wrapper
      .findAll("[data-clip-id]")[0]!
      .trigger("contextmenu", rightClick);
    const remove = wrapper
      .findAll("[data-test='sequence-context-item']")
      .find((item) => item.text() === "Remove clip");
    expect(remove?.attributes("disabled")).toBeDefined();
    expect(remove?.classes()).toContain("sq-context__danger");
  });

  it("opens the rail menu on the bench background and adds a clip", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();

    await wrapper.get(".ms-rail").trigger("contextmenu", rightClick);
    expect(itemLabels(wrapper)).toEqual([
      "Add clip",
      "Validate plan",
      "Import TOML…",
      "Export TOML",
      "Copy TOML",
      "Clear sequence",
    ]);

    await clickItem(wrapper, "Add clip");
    expect(store.clips).toHaveLength(3);
  });

  it("closes the menu on Escape", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    await wrapper.get(".ms-rail").trigger("contextmenu", rightClick);
    expect(wrapper.find("[data-test='sequence-context-menu']").exists()).toBe(
      true,
    );
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await flushPromises();
    expect(wrapper.find("[data-test='sequence-context-menu']").exists()).toBe(
      false,
    );
  });

  it("leaves the prompt textarea and the seam pill alone", async () => {
    const wrapper = mountComposer();
    await flushPromises();

    await wrapper
      .get("[data-test='clip-prompt']")
      .trigger("contextmenu", rightClick);
    expect(wrapper.find("[data-test='sequence-context-menu']").exists()).toBe(
      false,
    );

    await wrapper.get(".ms-seam").trigger("contextmenu", rightClick);
    expect(wrapper.find("[data-test='sequence-context-menu']").exists()).toBe(
      false,
    );
  });
});
