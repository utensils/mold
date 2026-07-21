import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import AdvancedDrawer from "./AdvancedDrawer.vue";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import type { GenerateFormState } from "../../types";

function baseForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  __testing__.resetForTest();
  const state = useGenerateForm().state.value;
  return { ...state, ...overrides };
}

function factory(
  family: string,
  overrides: Partial<GenerateFormState> = {},
  extra: Record<string, unknown> = {},
) {
  return mount(AdvancedDrawer, {
    props: { open: true, modelValue: baseForm(overrides), family, ...extra },
    global: {
      stubs: {
        LoraPicker: { template: "<div data-test='lora-picker-stub' />" },
        PlacementPanel: { template: "<div data-test='placement-stub' />" },
      },
    },
  });
}

function sections(family: string, extra: Record<string, unknown> = {}) {
  const wrapper = factory(family, {}, extra);
  const has = (key: string) =>
    wrapper.find(`[data-test='section-${key}']`).exists();
  return {
    scheduler: has("scheduler"),
    negative: has("negative"),
    source: has("source"),
    lora: has("lora"),
    upscale: has("upscale"),
    output: has("output"),
    video: has("video"),
    placement: has("placement"),
  };
}

describe("AdvancedDrawer capability matrix", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  it("sdxl exposes scheduler, negative, source and output", () => {
    const s = sections("sdxl");
    expect(s.scheduler).toBe(true);
    expect(s.negative).toBe(true);
    expect(s.source).toBe(true);
    expect(s.output).toBe(true);
    expect(s.video).toBe(false);
  });

  it("flux hides scheduler and negative but keeps source, lora and output", () => {
    const s = sections("flux");
    expect(s.scheduler).toBe(false);
    expect(s.negative).toBe(false);
    expect(s.source).toBe(true);
    expect(s.lora).toBe(true);
    expect(s.output).toBe(true);
  });

  it("qwen-image-edit hides the single-image source section", () => {
    expect(sections("qwen-image-edit").source).toBe(false);
  });

  it("ltx2 exposes the video section", () => {
    expect(sections("ltx2").video).toBe(true);
  });

  it("shows GPU placement only when GPUs are reported", () => {
    expect(sections("flux").placement).toBe(false);
    expect(
      sections("flux", { placementGpus: [{ ordinal: 0, name: "GPU 0" }] })
        .placement,
    ).toBe(true);
  });
});

describe("AdvancedDrawer default-open section", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  const expanded = (
    wrapper: ReturnType<typeof factory>,
    key: string,
  ): string | undefined =>
    wrapper
      .find(`[data-test='section-${key}'] .ms-acc__head`)
      .attributes("aria-expanded");

  it("opens on the first available section, never a hidden one", () => {
    // flux hides scheduler/negative, so the drawer must open on Source —
    // not sit fully collapsed on a scheduler section that isn't rendered.
    const w = factory("flux");
    expect(expanded(w, "source")).toBe("true");
  });

  it("reveals the LoRA picker when openTo is 'lora'", () => {
    const w = factory("flux", {}, { openTo: "lora" });
    expect(expanded(w, "lora")).toBe("true");
    expect(w.find("[data-test='lora-picker-stub']").exists()).toBe(true);
  });

  it("ignores openTo when that section is not available for the family", () => {
    // sd3.5 has no LoRA section → openTo is dropped and the first visible
    // section (Scheduler & sampling, for CFG++) opens instead.
    const w = factory("sd3.5", {}, { openTo: "lora" });
    expect(w.find("[data-test='section-lora']").exists()).toBe(false);
    expect(expanded(w, "scheduler")).toBe("true");
  });
});

describe("AdvancedDrawer ControlNet block", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  async function openSource(family: string, overrides = {}) {
    const wrapper = factory(family, overrides);
    const head = wrapper.find("[data-test='section-source'] .ms-acc__head");
    if (head.exists()) await head.trigger("click");
    return wrapper;
  }

  it("shows the ControlNet block for a controlnet family (sd15)", async () => {
    const wrapper = await openSource("sd15");
    expect(wrapper.find("[data-test='controlnet-block']").exists()).toBe(true);
  });

  it("hides the ControlNet block for flux", async () => {
    const wrapper = await openSource("flux");
    expect(wrapper.find("[data-test='controlnet-block']").exists()).toBe(false);
  });

  it("only surfaces control model + scale once an image is attached", async () => {
    const bare = await openSource("sd15");
    expect(bare.find("[data-test='control-attach']").exists()).toBe(true);
    expect(bare.find("[data-test='control-model']").exists()).toBe(false);

    const withImage = await openSource("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
    });
    expect(withImage.find("[data-test='control-model']").exists()).toBe(true);
  });

  it("round-trips the control model text field", async () => {
    const wrapper = await openSource("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
    });
    const input = wrapper.get("[data-test='control-model']");
    (input.element as HTMLInputElement).value = "control_v11p_sd15_canny";
    await input.trigger("input");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.controlModel).toBe("control_v11p_sd15_canny");
  });

  it("round-trips the control scale slider", async () => {
    const wrapper = await openSource("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
    });
    const range = wrapper.get("[data-test='control-scale'] input");
    (range.element as HTMLInputElement).value = "1.5";
    await range.trigger("input");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.controlScale).toBe(1.5);
  });

  it("removes the control image", async () => {
    const wrapper = await openSource("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
      controlModel: "control_v11p_sd15_canny",
    });
    await wrapper.get("[data-test='control-remove']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.controlImage).toBe(null);
  });
});

describe("AdvancedDrawer video suite", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  async function openVideo(family: string, overrides = {}) {
    const wrapper = factory(family, overrides);
    await wrapper
      .get("[data-test='section-video'] .ms-acc__head")
      .trigger("click");
    return wrapper;
  }

  it("shows the LTX-2 suite for ltx2 but not for ltx-video", async () => {
    const ltx2 = await openVideo("ltx2");
    expect(ltx2.find("[data-test='ltx2-suite']").exists()).toBe(true);
    const ltxVideo = await openVideo("ltx-video");
    expect(ltxVideo.find("[data-test='ltx2-suite']").exists()).toBe(false);
    // The GIF preview toggle is shared by every video family.
    expect(ltxVideo.find("[data-test='video-gif-preview']").exists()).toBe(
      true,
    );
  });

  it("hides the video suite entirely for flux", () => {
    expect(sections("flux").video).toBe(false);
  });

  it("toggles the GIF preview", async () => {
    const wrapper = await openVideo("ltx2");
    await wrapper.get("[data-test='video-gif-preview']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.gifPreview).toBe(true);
  });

  it("round-trips the LTX-2 pipeline select", async () => {
    const wrapper = await openVideo("ltx2");
    const select = wrapper.get("[data-test='ltx2-pipeline']");
    (select.element as HTMLSelectElement).value = "two-stage-hq";
    await select.trigger("change");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.pipeline).toBe("two-stage-hq");
  });

  it("toggles LTX-2 audio decode off (null default reads as on)", async () => {
    const wrapper = await openVideo("ltx2");
    await wrapper.get("[data-test='ltx2-enable-audio']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.enableAudio).toBe(false);
  });

  it("edits an existing keyframe's frame", async () => {
    const wrapper = await openVideo("ltx2", {
      keyframes: [
        {
          frame: 0,
          image: { kind: "upload", filename: "k.png", base64: "AA" },
        },
      ],
    });
    const input = wrapper.get("[data-test='ltx2-keyframe-frame']");
    (input.element as HTMLInputElement).value = "48";
    await input.trigger("input");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.keyframes[0]?.frame).toBe(48);
  });

  it("Reset also clears the video suite", async () => {
    const wrapper = factory("ltx2", {
      pipeline: "two-stage",
      gifPreview: true,
      spatialUpscale: "x2",
    });
    await wrapper.get("[data-test='advanced-reset']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.pipeline).toBe(null);
    expect(next.gifPreview).toBe(false);
    expect(next.spatialUpscale).toBe(null);
  });
});

describe("AdvancedDrawer interactions", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  it("opens one section at a time", async () => {
    const wrapper = factory("sdxl");
    // scheduler starts open; open negative and the scheduler body should close.
    await wrapper
      .get("[data-test='section-negative'] .ms-acc__head")
      .trigger("click");
    expect(wrapper.find("[data-test='negative-input']").exists()).toBe(true);
    expect(wrapper.find("[data-test='scheduler-select']").exists()).toBe(false);
  });

  it("adds negative-prompt quick chips", async () => {
    const wrapper = factory("sdxl", { negativePrompt: "" });
    await wrapper
      .get("[data-test='section-negative'] .ms-acc__head")
      .trigger("click");
    await wrapper.get("[data-test='neg-chip-blurry']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.negativePrompt).toBe("blurry");
  });

  it("snaps video frames to 8n+1", async () => {
    const wrapper = factory("ltx2");
    await wrapper
      .get("[data-test='section-video'] .ms-acc__head")
      .trigger("click");
    const input = wrapper.get("[data-test='video-frames']");
    (input.element as HTMLInputElement).value = "100";
    await input.trigger("change");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.frames).toBe(97);
  });

  it("Reset clears advanced fields but preserves the prompt", async () => {
    const wrapper = factory("sdxl", {
      prompt: "a lighthouse in a storm",
      negativePrompt: "blurry",
      loras: [{ path: "a", scale: 1 }],
      upscaleModel: "real-esrgan-x4plus",
    });
    await wrapper.get("[data-test='advanced-reset']").trigger("click");
    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.prompt).toBe("a lighthouse in a storm");
    expect(next.negativePrompt).toBe("");
    expect(next.loras).toEqual([]);
    expect(next.upscaleModel).toBe("");
  });

  it("emits close from Done", async () => {
    const wrapper = factory("sdxl");
    await wrapper.get("[data-test='advanced-done']").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });
});
