import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import Composer from "./Composer.vue";
import type { GenerateFormState } from "../types";

function makeForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  return {
    version: 2,
    model: "qwen-image-edit:q4",
    modelFamily: "qwen-image-edit",
    prompt: "change the shirt",
    negativePrompt: "",
    width: 1024,
    height: 1024,
    steps: 20,
    guidance: 3.5,
    seedMode: "random",
    seed: null,
    batchSize: 1,
    strength: 0.75,
    imageAttachments: [
      { kind: "upload", filename: "target.png", base64: "TARGET" },
      { kind: "upload", filename: "ref.png", base64: "REF" },
    ],
    scheduler: null,
    frames: null,
    fps: null,
    cfgPlus: false,
    outputFormat: "png",
    expand: { enabled: false, variations: 1, familyOverride: null },
    maskImage: null,
    controlImage: null,
    controlModel: "",
    controlScale: 1,
    upscaleModel: "",
    gifPreview: false,
    audioFile: null,
    audioFilePath: "",
    sourceVideo: null,
    sourceVideoPath: "",
    keyframes: [],
    pipeline: null,
    retakeRange: null,
    spatialUpscale: null,
    temporalUpscale: null,
    placement: null,
    loras: [],
    enableAudio: null,
    ...overrides,
  };
}

function mountComposer(
  form: GenerateFormState = makeForm(),
  family = form.modelFamily,
) {
  return mount(Composer, {
    props: {
      modelValue: form,
      mode: "single",
      queueDepth: null,
      queueCapacity: null,
      gpus: null,
      expandActive: false,
      family,
      placementGpus: [],
      chainDecision: { kind: "single" },
    },
  });
}

describe("Composer image attachments", () => {
  it("labels Qwen edit attachments as target and ordered references", () => {
    const w = mountComposer();

    expect(w.find("[data-test='attachment-role-0']").text()).toBe("Target");
    expect(w.find("[data-test='attachment-title-0']").text()).toBe("Picture 1");
    expect(w.find("[data-test='attachment-role-1']").text()).toBe("Reference");
    expect(w.find("[data-test='attachment-title-1']").text()).toBe("Picture 2");
  });

  it("removes and reorders attachments by emitting updated form state", async () => {
    const w = mountComposer();

    await w.find("[data-test='move-attachment-down-0']").trigger("click");
    let last = w.emitted("update:modelValue")?.at(-1)?.[0] as
      | GenerateFormState
      | undefined;
    expect(last?.imageAttachments.map((a) => a.base64)).toEqual([
      "REF",
      "TARGET",
    ]);

    await w.setProps({ modelValue: last });
    await w.find("[data-test='remove-attachment-1']").trigger("click");
    last = w.emitted("update:modelValue")?.at(-1)?.[0] as
      | GenerateFormState
      | undefined;
    expect(last?.imageAttachments.map((a) => a.base64)).toEqual(["REF"]);
  });

  it("drag reorders Qwen edit attachments", async () => {
    const w = mountComposer(
      makeForm({
        imageAttachments: [
          { kind: "upload", filename: "target.png", base64: "TARGET" },
          { kind: "upload", filename: "ref-a.png", base64: "REF_A" },
          { kind: "upload", filename: "ref-b.png", base64: "REF_B" },
        ],
      }),
    );
    const data = new Map<string, string>();
    const dataTransfer = {
      effectAllowed: "",
      setData: (key: string, value: string) => data.set(key, value),
      getData: (key: string) => data.get(key) ?? "",
    };

    await w
      .find("[data-test='attachment-card-2']")
      .trigger("dragstart", { dataTransfer });
    await w
      .find("[data-test='attachment-card-0']")
      .trigger("drop", { dataTransfer });

    const last = w.emitted("update:modelValue")?.at(-1)?.[0] as
      | GenerateFormState
      | undefined;
    expect(last?.imageAttachments.map((a) => a.base64)).toEqual([
      "REF_B",
      "TARGET",
      "REF_A",
    ]);
  });

  it("shows a source chip label for non-edit image families", () => {
    const w = mountComposer(makeForm({ modelFamily: "sdxl" }), "sdxl");

    expect(w.find("[data-test='attachment-role-0']").text()).toBe("Source");
    expect(w.find("[data-test='attachment-title-0']").text()).toBe("Image");
    expect(w.find("[data-test='attachment-role-1']").exists()).toBe(false);
  });
});
