import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AdvancedDrawer from "./AdvancedDrawer.vue";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import type { GenerateFormState, ModelInfoExtended } from "../../types";

// The upscale section reads the host's model list. Only upscalers matter here.
const UPSCALERS: ModelInfoExtended[] = [
  {
    name: "real-esrgan-x4plus:fp16",
    family: "upscaler",
    size_gb: 0.03,
    is_loaded: false,
    last_used: null,
    hf_repo: "",
    downloaded: true,
    default_steps: 0,
    default_guidance: 0,
    default_width: 0,
    default_height: 0,
    description: "Real-ESRGAN x4+ FP16",
  } as ModelInfoExtended,
];
vi.mock("../../api", () => ({
  fetchModels: vi.fn(async () => UPSCALERS),
  fetchCatalogInstalled: vi.fn(async () => ({
    entries: [],
    page: 1,
    page_size: 0,
    total: 0,
  })),
}));

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
        RouterLink: { template: "<a><slot /></a>" },
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

// The canonical Advanced section order shared with the desktop inspector
// (which has no placement section — desktop owns GPU placement in Settings).
const SECTION_ORDER = [
  "scheduler",
  "negative",
  "source",
  "lora",
  "upscale",
  "output",
  "video",
  "placement",
] as const;

function sectionIds(wrapper: ReturnType<typeof factory>): string[] {
  return wrapper
    .findAll("[data-test^='section-']")
    .map((node) => node.attributes("data-test")!.replace("section-", ""));
}

describe("AdvancedDrawer section ordering contract", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  const gpus = { placementGpus: [{ ordinal: 0, name: "GPU 0" }] };

  it("renders still-image sections in the canonical order", () => {
    expect(sectionIds(factory("sdxl", {}, gpus))).toEqual([
      "scheduler",
      "negative",
      "source",
      "lora",
      "upscale",
      "output",
      "placement",
    ]);
  });

  it("renders video sections in the canonical order", () => {
    expect(sectionIds(factory("ltx2", {}, gpus))).toEqual([
      "negative",
      "source",
      "lora",
      "output",
      "video",
      "placement",
    ]);
  });

  it("keeps every family's rendered sections a subsequence of the canon", () => {
    const families = [
      "sdxl",
      "sd15",
      "sd3.5",
      "flux",
      "qwen-image-edit",
      "ltx-video",
      "ltx2",
    ];
    for (const family of families) {
      const rendered = sectionIds(factory(family, {}, gpus));
      expect(rendered.length).toBeGreaterThan(0);
      expect(rendered).toEqual(
        SECTION_ORDER.filter((id) => rendered.includes(id)),
      );
    }
  });
});

describe("AdvancedDrawer capability matrix", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  it("matches desktop with always-open icon sections and no nested disclosure", () => {
    const wrapper = factory("sdxl");
    const sections = wrapper.findAll(".adv__sections > .ms-acc");

    expect(sections.length).toBeGreaterThan(0);
    expect(
      wrapper.findAll(".adv__sections > .ms-acc .ms-acc__plate svg"),
    ).toHaveLength(sections.length);
    expect(wrapper.find(".adv__sections .ms-acc__chevron").exists()).toBe(
      false,
    );
    expect(wrapper.find(".adv__sections button.ms-acc__head").exists()).toBe(
      false,
    );
    expect(wrapper.find("[data-test='negative-input']").exists()).toBe(true);
    expect(wrapper.find("[data-test='exact-width']").exists()).toBe(true);
  });

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

  it("qwen-image-edit exposes its required edit-image attachment section", async () => {
    const wrapper = factory("qwen-image-edit");
    expect(sections("qwen-image-edit").source).toBe(true);
    expect(wrapper.find("[data-test='source-attach']").exists()).toBe(true);
    expect(wrapper.text()).toContain("Edit images");
  });

  it("ltx2 exposes the video section", () => {
    expect(sections("ltx2").video).toBe(true);
  });

  it("offers post-generate upscale for stills but not for video", () => {
    // The server skips the post-upscaler whenever the response is a video
    // (queue.rs), so the section must not pretend otherwise.
    expect(sections("flux").upscale).toBe(true);
    expect(sections("ltx2").upscale).toBe(false);
  });

  it("shows GPU placement only when GPUs are reported", () => {
    expect(sections("flux").placement).toBe(false);
    expect(
      sections("flux", { placementGpus: [{ ordinal: 0, name: "GPU 0" }] })
        .placement,
    ).toBe(true);
  });

  it("renders the real placement child with the selected placement", async () => {
    const placement = { text_encoders: { kind: "cpu" as const } };
    const wrapper = factory(
      "flux",
      { placement },
      { placementGpus: [{ ordinal: 0, name: "GPU 0" }] },
    );
    await wrapper
      .get("[data-test='section-placement'] .ms-acc__head")
      .trigger("click");
    expect(
      wrapper.getComponent({ name: "PlacementPanel" }).props("modelValue"),
    ).toEqual(placement);
  });

  it("offers all five source-fit policies", async () => {
    const wrapper = factory("sdxl", {
      imageAttachments: [
        { kind: "upload", filename: "source.png", base64: "AA" },
      ],
    });
    await wrapper
      .get("[data-test='section-source'] .ms-acc__head")
      .trigger("click");
    const labels = wrapper
      .findAll("[aria-label='Fit to canvas'] button")
      .map((button) => button.text());
    expect(labels).toEqual([
      "Contain",
      "Cover",
      "Pad + repaint",
      "Lanczos",
      "Upscale + fit",
    ]);
  });
});

describe("AdvancedDrawer always-open sections", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => __testing__.resetForTest());

  it("shows every available control without aria disclosure state", () => {
    const w = factory("flux");
    expect(w.find("[data-test='source-attach']").exists()).toBe(true);
    expect(w.find("[data-test='lora-picker-stub']").exists()).toBe(true);
    expect(w.find("[data-test='exact-width']").exists()).toBe(true);
    expect(w.find("[aria-expanded]").exists()).toBe(false);
  });

  it("still hides sections unavailable to the selected family", () => {
    const w = factory("sd3.5");
    expect(w.find("[data-test='section-lora']").exists()).toBe(false);
    expect(w.find("[data-test='cfg-plus']").exists()).toBe(true);
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

  it("suggests installed ControlNet models while retaining custom input", async () => {
    const wrapper = await openSource("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
    });
    await wrapper.setProps({
      models: [
        {
          ...UPSCALERS[0],
          name: "controlnet-canny-sd15:fp16",
          family: "controlnet",
        },
      ],
    });
    expect(wrapper.get("[data-test='control-model']").attributes("list")).toBe(
      "installed-controlnet-models",
    );
    expect(
      wrapper.get("#installed-controlnet-models option").attributes("value"),
    ).toBe("controlnet-canny-sd15:fp16");
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

  it("keeps scheduler and negative controls visible together", () => {
    const wrapper = factory("sdxl");
    expect(wrapper.find("[data-test='negative-input']").exists()).toBe(true);
    expect(wrapper.find("[data-test='scheduler-select']").exists()).toBe(true);
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

  it("exact-size inputs snap to 16px and swap orientation", async () => {
    const wrapper = factory("flux");
    await wrapper
      .get("[data-test='section-output'] .ms-acc__head")
      .trigger("click");
    const width = wrapper.get("[data-test='exact-width']");
    await width.setValue("1000");
    await width.trigger("change");
    let next = wrapper.emitted("update:modelValue")?.at(-1)?.[0] as {
      width: number;
    };
    expect(next.width).toBe(1008); // 1000 → nearest multiple of 16

    await wrapper.setProps({
      modelValue: { ...baseForm({ width: 1216, height: 704 }) },
    });
    await wrapper.get("[data-test='exact-swap']").trigger("click");
    const swapped = wrapper.emitted("update:modelValue")?.at(-1)?.[0] as {
      width: number;
      height: number;
    };
    expect(swapped.width).toBe(704);
    expect(swapped.height).toBe(1216);
  });

  it("round-trips 'Upscale after generate' into the generate request", async () => {
    // The always-visible picker must survive into the wire request and the
    // explicit Off option must remove it again.
    const form = useGenerateForm();
    const wrapper = mount(AdvancedDrawer, {
      props: { open: true, modelValue: form.state.value, family: "flux" },
      global: {
        stubs: {
          LoraPicker: { template: "<div />" },
          PlacementPanel: { template: "<div />" },
          RouterLink: { template: "<a><slot /></a>" },
        },
      },
    });
    await flushPromises();

    await wrapper
      .get("[data-test='upscale-model']")
      .setValue("real-esrgan-x4plus:fp16");
    let next = wrapper.emitted("update:modelValue")!.at(-1)![0] as
      GenerateFormState | undefined;
    Object.assign(form.state.value, next);
    expect(form.state.value.upscaleModel).toBe("real-esrgan-x4plus:fp16");
    expect(form.toRequest().upscale_model).toBe("real-esrgan-x4plus:fp16");

    await wrapper.setProps({ modelValue: { ...form.state.value } });
    await wrapper.get("[data-test='upscale-model']").setValue("");
    next = wrapper.emitted("update:modelValue")!.at(-1)![0] as
      GenerateFormState | undefined;
    Object.assign(form.state.value, next);
    expect(form.state.value.upscaleModel).toBe("");
    expect(form.toRequest().upscale_model).toBeUndefined();
  });

  it("emits close from Done in the phone sheet", async () => {
    // Inline (desktop) has no Done — it's always visible. The Done button
    // lives only in the phone Advanced sheet (mobile: true).
    const wrapper = factory("sdxl", {}, { mobile: true });
    await wrapper.get("[data-test='advanced-done']").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });
});
