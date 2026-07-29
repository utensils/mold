import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia, type Pinia } from "pinia";
import SequenceComposer from "./SequenceComposer.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { parseChainScript } from "@studio/lib/chainToml";
import type { SequenceSharedParams } from "@studio/lib/sequenceForm";
import * as api from "../api";

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  fetchChainLimits: vi.fn(),
  listGallery: vi.fn(async () => []),
}));

const fetchChainLimitsMock = vi.mocked(api.fetchChainLimits);

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

  it("gates the audio toggle on chain-limits supports_audio", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    expect(wrapper.find("[data-test='sequence-enable-audio']").exists()).toBe(
      true,
    );

    fetchChainLimitsMock.mockResolvedValue(
      ltx2Limits({ supports_audio: false }),
    );
    const store = useSequenceDraftStore();
    store.enableAudio = true;
    await wrapper.setProps({
      model: "ltx-video-distilled",
      family: "ltx-video",
    });
    await flushPromises();
    expect(wrapper.find("[data-test='sequence-enable-audio']").exists()).toBe(
      false,
    );
    expect(store.enableAudio).toBe(false);
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
      "177 frames",
    );
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
    await wrapper.get("[data-test='sequence-copy-toml']").trigger("click");
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

  it("attaches an opening image to clip 1 only", async () => {
    const wrapper = mountComposer();
    await flushPromises();
    const store = useSequenceDraftStore();
    expect(wrapper.find("[data-test='opening-image-attach']").exists()).toBe(
      true,
    );
    store.activeClipId = store.clips[1]?.id ?? null;
    await flushPromises();
    expect(wrapper.find("[data-test='opening-image-attach']").exists()).toBe(
      false,
    );
  });
});
