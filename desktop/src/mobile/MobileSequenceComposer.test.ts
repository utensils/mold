import { mount, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import SeamPill from "@ui/components/SeamPill.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";
import type { ModelEntry } from "../lib/api/types";
import MobileSeamSheet from "./MobileSeamSheet.vue";
import MobileSequenceComposer from "./MobileSequenceComposer.vue";

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

let wrapper: VueWrapper | null = null;

function mountComposer(props: Record<string, unknown> = {}): VueWrapper {
  wrapper = mount(MobileSequenceComposer, {
    props: {
      selectedModel: ltx2,
      chainLimits: limits,
      target: { baseUrl: "http://studio:7680", apiKey: "secret" },
      fps: 24,
      submitting: false,
      error: "",
      ...props,
    },
  });
  return wrapper;
}

function clipCards() {
  return wrapper!.findAll("[data-test='mobile-sequence-clip']");
}

beforeEach(() => {
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
        chainLimits: limits,
        target: { baseUrl: "http://studio:7680", apiKey: "secret" },
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

    expect(draft.clips[0]!.sourceImage).toEqual({
      filename: "opening.jpg",
      base64: "wire-bytes",
    });
    expect(draft.clips[1]!.sourceImage).toBeNull();
    expect(
      wrapper!.get("[data-test='mobile-sequence-source-preview']").attributes("src"),
    ).toContain("wire-bytes");
  });

  it("summarizes the timeline at the form's own frame rate", () => {
    mountComposer({ fps: 12 });
    // Two 97-frame clips with LTX-2's 17-frame smooth tail: 97 + 80 = 177.
    const summary = wrapper!.get("[data-test='mobile-sequence-duration']").text();
    expect(summary).toContain("177f · 14.8s");
    expect(summary).toContain("14.8s");
  });
});
