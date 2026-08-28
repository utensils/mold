import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import SourceMediaPanel from "./SourceMediaPanel.vue";
import {
  useGenerateForm,
  __testing__,
} from "../../composables/useGenerateForm";
import type { GenerateFormState, ModelInfoExtended } from "../../types";

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
  return mount(SourceMediaPanel, {
    props: { modelValue: baseForm(overrides), family, ...extra },
  });
}

beforeEach(() => localStorage.clear());
afterEach(() => __testing__.resetForTest());

describe("SourceMediaPanel — the model dictates the wells", () => {
  it("renders the single-source well for image families in the primary form", () => {
    const wrapper = factory("sdxl");
    expect(wrapper.find("[data-test='source-media-panel']").exists()).toBe(
      true,
    );
    expect(wrapper.find("[data-test='source-well']").exists()).toBe(true);
  });

  it("renders the attachment strip entry for Qwen edit and FLUX.2 Dev", () => {
    const qwen = factory("qwen-image-edit");
    expect(qwen.find("[data-test='source-well']").exists()).toBe(true);
    expect(qwen.text()).toContain("Edit images");

    const flux2 = factory("flux2", { model: "flux2-dev" });
    expect(flux2.text()).toContain("Reference images");
  });

  it("routes the Qwen Target well through the standard gallery picker", async () => {
    const wrapper = factory("qwen-image-edit");
    await wrapper.get("[data-test='source-gallery']").trigger("click");
    expect(wrapper.emitted("open-target-picker")).toHaveLength(1);
  });

  it("routes gallery, and clear per slot to the page's pickers", async () => {
    const wrapper = factory("sdxl");
    await wrapper.get("[data-test='source-gallery']").trigger("click");
    expect(wrapper.emitted("open-picker")).toHaveLength(1);
  });
});

describe("SourceMediaPanel — per-model source-image contract (#772)", () => {
  function wanModel(source_image?: string | null): ModelInfoExtended {
    return {
      name: "wan22-ti2v-5b:fp16",
      family: "wan",
      size_gb: 10,
      is_loaded: false,
      last_used: null,
      hf_repo: "",
      downloaded: true,
      default_steps: 20,
      default_guidance: 3.5,
      default_width: 1280,
      default_height: 704,
      default_frames: 81,
      default_fps: 24,
      description: "Wan 2.2 TI2V 5B",
      ...(source_image === undefined ? {} : { source_image }),
    } as ModelInfoExtended;
  }

  function wan(
    source_image?: string | null,
    overrides: Partial<GenerateFormState> = {},
  ) {
    return factory(
      "wan",
      {
        model: "wan22-ti2v-5b:fp16",
        modelFamily: "wan",
        frames: 81,
        fps: 24,
        sourceImageCapability: source_image ?? null,
        ...overrides,
      },
      { models: [wanModel(source_image)] },
    );
  }

  it("keeps today's source well when the server advertises nothing", () => {
    const wrapper = wan(undefined);
    expect(wrapper.find("[data-test='source-media-panel']").exists()).toBe(
      true,
    );
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(
      false,
    );
    // An older server rejects wan keyframes outright, so absence hides the well.
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(false);
  });

  it("hides the whole panel for a text-to-video checkpoint", () => {
    const wrapper = wan("unsupported");
    expect(wrapper.find("[data-test='source-media-panel']").exists()).toBe(
      false,
    );
  });

  it("marks the source required and names why the request would be refused", () => {
    const wrapper = wan("required");
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(
      true,
    );
    expect(wrapper.get("[data-test='source-conditioning-error']").text()).toBe(
      "This checkpoint is image-to-video only. Attach a source image to use as the first frame.",
    );
  });

  // #783: the inline notice has to read a continuation the way submit and
  // admission do (`request_carries_source_frames`), or the well contradicts a
  // Generate button that accepts the draft.
  it("clears the inline message for a continuation with no attached image", () => {
    const wrapper = wan("required", {
      extendVideo: { kind: "upload", filename: "clip.mp4", base64: "Q0xJUA==" },
    });
    expect(
      wrapper.find("[data-test='source-conditioning-error']").exists(),
    ).toBe(false);
  });

  it("clears the inline message once the opening frame is attached", () => {
    const wrapper = wan("required", {
      imageAttachments: [
        { kind: "upload", filename: "open.png", base64: "FIRST" },
      ],
    });
    expect(
      wrapper.find("[data-test='source-conditioning-error']").exists(),
    ).toBe(false);
  });

  it("offers the optional End frame well and its gallery/remove affordances", async () => {
    const wrapper = wan("optional");
    expect(wrapper.get("[data-test='end-frame-hint']").text()).toContain(
      "first/last-frame clip",
    );
    await wrapper.get("[data-test='end-frame-gallery']").trigger("click");
    expect(wrapper.emitted("open-end-frame-picker")).toHaveLength(1);

    const attached = wan("optional", {
      imageAttachments: [
        { kind: "upload", filename: "open.png", base64: "FIRST" },
      ],
      endFrame: { kind: "upload", filename: "close.png", base64: "TEFTVA==" },
    });
    await attached.get("[data-test='end-frame-remove']").trigger("click");
    expect(attached.emitted("clear-end-frame")).toHaveLength(1);
  });

  it("explains an end frame with no first frame", () => {
    const wrapper = wan("optional", {
      endFrame: { kind: "upload", filename: "close.png", base64: "TEFTVA==" },
    });
    expect(wrapper.get("[data-test='source-conditioning-error']").text()).toBe(
      "An end frame needs a first frame. Attach a source image, or remove the end frame.",
    );
  });

  it("never offers an End frame well outside wan", () => {
    const wrapper = factory("ltx2", {
      model: "ltx-2-19b:fp8",
      modelFamily: "ltx2",
      sourceImageCapability: "optional",
    });
    expect(wrapper.find("[data-test='source-well']").exists()).toBe(true);
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(false);
  });
});

describe("SourceMediaPanel — source fit", () => {
  it("offers all five source-fit policies once an image is attached", () => {
    const wrapper = factory("sdxl", {
      imageAttachments: [
        { kind: "upload", filename: "source.png", base64: "AA" },
      ],
    });
    const labels = wrapper
      .findAll("[aria-label='Fit to canvas'] button")
      .map((button) => button.text());
    expect(labels).toEqual([
      "Fit + repaint borders",
      "Crop to fill",
      "Fit with borders",
      "Stretch to fill",
      "Upscale, then crop",
    ]);
  });

  it("offers maskless fit choices for a Qwen edit Target", () => {
    const wrapper = factory("qwen-image-edit", {
      imageAttachments: [
        { kind: "upload", filename: "target.png", base64: "AA" },
      ],
    });
    const labels = wrapper
      .findAll("[aria-label='Fit to canvas'] button")
      .map((button) => button.text());
    expect(labels).toEqual([
      "Crop to fill",
      "Fit with borders",
      "Stretch to fill",
      "Upscale, then crop",
    ]);
    expect(wrapper.get("[data-test='source-fit-help']").text()).toContain(
      "conditioning limit: 1 MP from this model",
    );
  });
});

describe("SourceMediaPanel — ControlNet block", () => {
  it("shows the ControlNet block for a controlnet family (sd15) only", () => {
    expect(
      factory("sd15").find("[data-test='controlnet-block']").exists(),
    ).toBe(true);
    expect(
      factory("flux").find("[data-test='controlnet-block']").exists(),
    ).toBe(false);
  });

  it("only surfaces control model + scale once an image is attached", () => {
    const bare = factory("sd15");
    expect(bare.find("[data-test='control-attach']").exists()).toBe(true);
    expect(bare.find("[data-test='control-model']").exists()).toBe(false);

    const withImage = factory("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
    });
    expect(withImage.find("[data-test='control-model']").exists()).toBe(true);
  });

  it("round-trips the control model text field", async () => {
    const wrapper = factory("sd15", {
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
    const wrapper = factory("sd15", {
      controlImage: { kind: "upload", filename: "canny.png", base64: "AA" },
    });
    await wrapper.setProps({
      models: [
        {
          name: "controlnet-canny-sd15:fp16",
          family: "controlnet",
          downloaded: true,
        } as ModelInfoExtended,
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
    const wrapper = factory("sd15", {
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
    const wrapper = factory("sd15", {
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

describe("SourceMediaPanel — MiniMax H3 FL2VA boundaries", () => {
  const fl2va = {
    name: "minimax-h3-fl2va:comfy-pruned-int8",
    family: "minimax-h3",
    downloaded: true,
    source_image: "required",
  } as ModelInfoExtended;

  it("renders the same standard wells and delegates gallery picks to the page", async () => {
    const wrapper = factory(
      "minimax-h3",
      { model: fl2va.name, modelFamily: fl2va.family },
      { models: [fl2va] },
    );
    expect(wrapper.text()).toContain("Frame endpoints");
    expect(wrapper.find("[data-test='source-well']").exists()).toBe(true);
    expect(wrapper.find("[data-test='source-required-badge']").exists()).toBe(
      true,
    );
    // Reviewed first-frame-only runtime: no empty last-frame well.
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(false);

    await wrapper.get("[data-test='source-gallery']").trigger("click");
    expect(wrapper.emitted("open-h3-first-frame-picker")).toHaveLength(1);
  });

  it("offers both boundary wells when no endpoint is required", async () => {
    const open = { ...fl2va, source_image: undefined } as ModelInfoExtended;
    const wrapper = factory(
      "minimax-h3",
      { model: open.name, modelFamily: open.family },
      { models: [open] },
    );
    expect(wrapper.find("[data-test='end-frame-well']").exists()).toBe(true);
    await wrapper.get("[data-test='end-frame-gallery']").trigger("click");
    expect(wrapper.emitted("open-h3-last-frame-picker")).toHaveLength(1);
  });

  it("exposes H3 Ref2VA ordered references through the shared picker event", async () => {
    const ref2va = {
      name: "minimax-h3-ref2va:comfy-pruned-int8",
      family: "minimax-h3",
      downloaded: true,
    } as ModelInfoExtended;
    const wrapper = factory(
      "minimax-h3",
      { model: ref2va.name, modelFamily: ref2va.family },
      { models: [ref2va] },
    );
    expect(wrapper.find("[data-test='source-media-panel']").exists()).toBe(
      true,
    );
    expect(wrapper.text()).toContain("Ordered references");
    expect(wrapper.find("[data-test='h3-reference-files']").exists()).toBe(
      true,
    );
    await wrapper.get("[data-test='h3-reference-library']").trigger("click");
    expect(wrapper.emitted("open-h3-reference-picker")).toHaveLength(1);
  });
});

describe("SourceMediaPanel — H3 reference crop", () => {
  it("relays the shared panel's Crop action to the page-level editor host", async () => {
    const ref2va = {
      name: "minimax-h3-ref2va:comfy-pruned-int8",
      family: "minimax-h3",
      downloaded: true,
    } as ModelInfoExtended;
    const wrapper = factory(
      "minimax-h3",
      {
        model: ref2va.name,
        modelFamily: ref2va.family,
        h3Authoring: {
          firstFrame: null,
          lastFrame: null,
          references: [
            {
              reference: {
                kind: "image",
                media: { authority: "inline", data: "SU1BR0U=" },
                provenance: { name: "subject.png", sha256: "a".repeat(64) },
                mime_type: "image/png",
                width: 1024,
                height: 768,
              },
            },
          ],
        },
      },
      { models: [ref2va] },
    );
    await wrapper.get("[data-test='h3-reference-crop-0']").trigger("click");
    expect(wrapper.emitted("crop-h3-reference")).toEqual([[0]]);
  });
});

describe("SourceMediaPanel — direct file uploads", () => {
  const PNG_7x4 =
    "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAIAAAAmkwkpAAAAAElFTkSuQmCC";

  it("decodes dimensions for a dropped PNG so source-matched shapes keep working", async () => {
    const wrapper = factory("sdxl");
    const bytes = Uint8Array.from(atob(PNG_7x4), (c) => c.charCodeAt(0));
    const file = new File([bytes], "still.png", { type: "image/png" });
    await wrapper
      .get("[data-test='source-well']")
      .trigger("drop", { dataTransfer: { files: [file] } });
    await new Promise((resolve) => setTimeout(resolve, 0));

    const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
      GenerateFormState,
    ];
    expect(next.imageAttachments[0]).toMatchObject({
      filename: "still.png",
      base64: PNG_7x4,
      width: 7,
      height: 4,
    });
    expect(next.sourceFitPolicy).toEqual({ mode: "crop-fill" });
  });

  it("refuses a dropped non-PNG/JPEG file with a visible error", async () => {
    const wrapper = factory("sdxl");
    const file = new File(["webp"], "still.webp", { type: "image/webp" });
    await wrapper
      .get("[data-test='source-well']")
      .trigger("drop", { dataTransfer: { files: [file] } });
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
    expect(
      wrapper.get("[data-test='source-conditioning-error']").text(),
    ).toContain("Only PNG or JPEG");
  });
});
