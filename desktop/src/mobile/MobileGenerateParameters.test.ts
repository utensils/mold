import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { defineComponent, reactive } from "vue";
import { describe, expect, it, vi } from "vitest";
import { buildRequest, newGenerateForm, type GenerateForm } from "../lib/generateForm";
import type { Ltx2CameraControlInfo, Ltx2ControlAdapterInfo, ModelEntry } from "../lib/api/types";
import MobileGenerateParameters from "./MobileGenerateParameters.vue";

const { fileToBase64 } = vi.hoisted(() => ({ fileToBase64: vi.fn() }));

vi.mock("../lib/image", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/image")>()),
  fileToBase64,
}));

const upscaler: ModelEntry = {
  name: "real-esrgan-x4plus:fp16",
  family: "upscaler",
  size_gb: 0.07,
  is_loaded: false,
  hf_repo: "example/upscaler",
  default_steps: 1,
  default_guidance: 0,
  default_width: 1024,
  default_height: 1024,
  description: "4x upscaler",
  downloaded: false,
};
const cameraControl: Ltx2CameraControlInfo = {
  id: "dolly-in",
  label: "Dolly in",
  size_bytes: 327_309_208,
  installed: false,
  download_model: "ltx2-camera-control-dolly-in-19b",
  download_repo: "Lightricks/camera",
  download_filename: "dolly-in.safetensors",
  download_sha256: "a".repeat(64),
};

function formFor(family: string, model = `${family}:test`): GenerateForm {
  return { ...newGenerateForm(), family, model };
}

function mountParameters(
  initial: GenerateForm,
  upscalers: ModelEntry[] = [],
  audioOutputSupported = true,
  controlAdapters: Ltx2ControlAdapterInfo[] = [],
  cameraControls: Ltx2CameraControlInfo[] = [cameraControl],
  selectedModel: ModelEntry | null = null,
): { wrapper: VueWrapper; form: GenerateForm } {
  const form = reactive(initial) as GenerateForm;
  const Harness = defineComponent({
    components: { MobileGenerateParameters },
    setup: () => ({
      audioOutputSupported,
      cameraControls,
      controlAdapters,
      form,
      selectedModel,
      upscalers,
    }),
    template: `<MobileGenerateParameters :form="form" :upscalers="upscalers" :selected-model="selectedModel" :audio-output-supported="audioOutputSupported" :control-adapters="controlAdapters" :camera-controls="cameraControls" camera-controls-loaded />`,
  });
  return { wrapper: mount(Harness), form };
}

async function attachFile(wrapper: VueWrapper, selector: string, file: File): Promise<void> {
  const input = wrapper.get(selector);
  Object.defineProperty(input.element, "files", { configurable: true, value: [file] });
  await input.trigger("change");
  await flushPromises();
}

describe("MobileGenerateParameters", () => {
  it("resets model-owned controls when the recipe changes", async () => {
    const model = {
      name: "ltx2:test",
      family: "ltx2",
      default_width: 1024,
      default_height: 576,
      default_steps: 30,
      default_guidance: 3,
      size_gb: 1,
      is_loaded: false,
      hf_repo: "fixture",
      description: "",
      downloaded: true,
    } as ModelEntry;
    const initial = formFor("ltx2", model.name);
    Object.assign(initial, { width: 640, height: 640, steps: 7, guidance: 8 });
    const { wrapper, form } = mountParameters(initial, [], true, [], [], model);
    await wrapper.get("[data-test='mobile-ltx2-disclosure']").trigger("click");
    await wrapper.get("[data-test='mobile-ltx2-pipeline']").setValue("two-stage");
    expect(form).toMatchObject({
      pipeline: "two-stage",
      width: 1024,
      height: 576,
      steps: 30,
      guidance: 3,
    });
  });

  it("offers a duration slider while retaining exact video fields", async () => {
    const model = {
      name: "ltx-2-19b-distilled:fp8",
      family: "ltx2",
      default_frames: 97,
      default_fps: 24,
      max_runtime_seconds: 20,
      max_frames_absolute: 604,
      frame_step: 8,
    } as ModelEntry;
    const { wrapper, form } = mountParameters(
      { ...formFor("ltx2", model.name), frames: 97, fps: 24 },
      [],
      true,
      [],
      [cameraControl],
      model,
    );
    expect(wrapper.get("[data-test='mobile-advanced-duration']").text()).toContain("4.0s");
    await wrapper.get("[data-test='mobile-advanced-duration'] input[type='range']").setValue("241");
    expect(form.frames).toBe(241);
    expect(wrapper.find("[data-test='mobile-frames']").exists()).toBe(true);
    expect(wrapper.find("[data-test='mobile-fps']").exists()).toBe(true);
  });

  it("renders host-provided reference controls and guide copy", async () => {
    const adapters: Ltx2ControlAdapterInfo[] = [
      {
        id: "detailer",
        label: "Detailer",
        guide: "Use the source clip.",
        size_bytes: 2_617_401_920,
        installed: false,
        download_model: "ltx2-control-detailer-19b",
        download_repo: "Lightricks/control",
        download_filename: "control.safetensors",
        download_sha256: "a".repeat(64),
      },
    ];
    const { wrapper, form } = mountParameters(
      formFor("ltx2", "ltx-2-19b-distilled:fp8"),
      [],
      true,
      adapters,
    );
    await wrapper.get("[data-test='mobile-ltx2-reference-control']").setValue("detailer");
    expect(form.icLoraControl).toBe("detailer");
    expect(form.pipeline).toBe("ic-lora");
    expect(wrapper.get("[data-test='mobile-ltx2-reference-guide']").text()).toContain(
      "source clip",
    );
  });

  it("offers an editable unbounded batch stepper and image-only post upscale", async () => {
    const { wrapper, form } = mountParameters(formFor("flux"), [upscaler]);

    expect(wrapper.get("[data-test='mobile-batch-value']").attributes("value")).toBe("1");
    expect(wrapper.find("[data-test='mobile-scheduler']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-cfg-plus']").exists()).toBe(false);
    expect(wrapper.get("[data-test='mobile-upscale']").text()).toContain("downloads on first use");

    for (let index = 0; index < 10; index += 1) {
      await wrapper.get("[data-test='mobile-batch-increment']").trigger("click");
    }
    expect(form.batchSize).toBe(11);
    expect(wrapper.get("[data-test='mobile-batch-increment']").attributes()).not.toHaveProperty(
      "disabled",
    );

    await wrapper.get("[data-test='mobile-batch-value']").setValue("250");
    await wrapper.get("[data-test='mobile-batch-value']").trigger("change");
    expect(form.batchSize).toBe(250);

    await wrapper.get("[data-test='mobile-batch-value']").setValue("999999999999");
    await wrapper.get("[data-test='mobile-batch-value']").trigger("change");
    expect(form.batchSize).toBe(10_000);

    await wrapper.get("[data-test='mobile-batch-value']").setValue("250");
    await wrapper.get("[data-test='mobile-batch-value']").trigger("change");
    await wrapper.get("[data-test='mobile-batch-decrement']").trigger("click");
    expect(form.batchSize).toBe(249);
    await wrapper.get("[data-test='mobile-upscale']").setValue(upscaler.name);
    expect(form.upscaleModel).toBe(upscaler.name);
  });

  it("capability-gates scheduler and CFG++, and locks edit-model batches to one", async () => {
    const sd15 = mountParameters(formFor("sd15"));
    expect(sd15.wrapper.find("[data-test='mobile-scheduler']").exists()).toBe(true);
    await sd15.wrapper.get("[data-test='mobile-scheduler']").setValue("ddim");
    expect(sd15.form.scheduler).toBe("ddim");
    sd15.wrapper.unmount();

    const sd3 = mountParameters(formFor("sd3.5"));
    expect(sd3.wrapper.find("[data-test='mobile-cfg-plus']").exists()).toBe(true);
    await sd3.wrapper.get("[data-test='mobile-cfg-plus']").setValue(true);
    expect(sd3.form.cfgPlus).toBe(true);
    sd3.wrapper.unmount();

    const editForm = formFor("qwen-image-edit");
    editForm.batchSize = 6;
    const edit = mountParameters(editForm);
    await flushPromises();
    expect(edit.form.batchSize).toBe(1);
    expect(edit.wrapper.get("[data-test='mobile-batch-locked']").text()).toContain(
      "render one at a time",
    );
    expect(edit.wrapper.get("[data-test='mobile-batch-increment']").attributes()).toHaveProperty(
      "disabled",
    );
  });

  it("swaps the scheduler row for wan's sampler recipe", async () => {
    const { wrapper, form } = mountParameters(formFor("wan", "wan22-t2v-a14b:q5"));

    // One control per wire field: the solver row replaces the generic
    // scheduler row rather than adding a second writer of `scheduler`.
    expect(wrapper.find("[data-test='mobile-scheduler']").exists()).toBe(false);
    const solver = wrapper.get("[data-test='mobile-wan-solver']");
    expect(solver.findAll("option").map((option) => option.text())).toEqual([
      "Default",
      "UniPC",
      "Euler",
      "DPM++",
    ]);
    await solver.setValue("euler");
    expect(form.scheduler).toBe("euler");

    const shift = wrapper.get("[data-test='mobile-wan-sample-shift']");
    expect((shift.element as HTMLInputElement).value).toBe("");
    await shift.setValue("12");
    expect(form.wanRecipe.sampleShift).toBe(12);
    await shift.setValue("");
    expect(form.wanRecipe.sampleShift).toBeNull();

    await wrapper.get("[data-test='mobile-wan-distill-high']").setValue("1.8");
    expect(form.wanRecipe.distillStrengthHigh).toBe(1.8);
    expect(wrapper.get("[data-test='mobile-wan-recipe-count']").text()).toBe("1");

    await wrapper.get("[data-test='mobile-wan-distill-low']").setValue("9");
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-wan-recipe-error']").text()).toBe(
      "Low-noise distill strength must be at most 4.",
    );
  });

  it("holds the develop gate while a wan recipe value is out of band", async () => {
    const { wrapper, form } = mountParameters(formFor("wan", "wan22-t2v-a14b:q4"));
    const child = wrapper.getComponent(MobileGenerateParameters);
    expect(child.emitted("validity-change")?.at(-1)).toEqual([true]);

    // An invalid value must invalidate the form, not be silently dropped
    // from the wire while the button stays live (codex review).
    await wrapper.get("[data-test='mobile-wan-sample-shift']").setValue("-3");
    await flushPromises();
    expect(child.emitted("validity-change")?.at(-1)).toEqual([false]);

    await wrapper.get("[data-test='mobile-wan-sample-shift']").setValue("12");
    await flushPromises();
    expect(child.emitted("validity-change")?.at(-1)).toEqual([true]);
    expect(form.wanRecipe.sampleShift).toBe(12);
  });

  it("hides the wan distill strengths on a tier that ships none, and the recipe off-family", () => {
    const quality = mountParameters(formFor("wan", "wan22-ti2v-5b:fp16"));
    expect(quality.wrapper.find("[data-test='mobile-wan-sample-shift']").exists()).toBe(true);
    expect(quality.wrapper.find("[data-test='mobile-wan-distill-high']").exists()).toBe(false);
    quality.wrapper.unmount();

    const sd15 = mountParameters(formFor("sd15"));
    expect(sd15.wrapper.find("[data-test='mobile-wan-solver']").exists()).toBe(false);
    expect(sd15.wrapper.find("[data-test='mobile-wan-sample-shift']").exists()).toBe(false);
  });

  it("validates video frames and explains automatic long-video routing", async () => {
    const chain = mountParameters(formFor("ltx2", "ltx-2-19b-distilled:fp8"));
    const child = chain.wrapper.getComponent(MobileGenerateParameters);

    const frames = chain.wrapper.get("[data-test='mobile-frames']");
    (frames.element as HTMLInputElement).value = "100";
    await frames.trigger("input");
    expect(chain.wrapper.get("[data-test='mobile-frames-error']").text()).toContain("8n+1");
    expect(child.emitted("validity-change")?.at(-1)).toEqual([false]);

    await chain.wrapper.get("[data-test='mobile-frames']").setValue("177");
    expect(chain.wrapper.get("[data-test='mobile-chain-cue']").text()).toContain("2 chained clips");
    expect(child.emitted("validity-change")?.at(-1)).toEqual([true]);
    chain.wrapper.unmount();

    // Every ltx2 checkpoint chains now, dev included, so the remaining reject
    // case is the server's sixteen-stage ceiling.
    const unsupported = mountParameters(formFor("ltx2", "ltx-2-19b-dev:fp8"));
    await unsupported.wrapper.get("[data-test='mobile-frames']").setValue("1305");
    expect(unsupported.wrapper.get("[data-test='mobile-chain-reject']").text()).toContain(
      "at most 1297 frames",
    );
    expect(
      unsupported.wrapper.getComponent(MobileGenerateParameters).emitted("validity-change")?.at(-1),
    ).toEqual([false]);
  });

  it("accepts wan's 4n+1 frame counts and snaps its exact field on that grid", async () => {
    const model = {
      name: "wan22-i2v-a14b:q5",
      family: "wan",
      frame_step: 4,
    } as ModelEntry;
    const wan = mountParameters(
      { ...formFor("wan", model.name), frames: 45 },
      [],
      true,
      [],
      [cameraControl],
      model,
    );
    const child = wan.wrapper.getComponent(MobileGenerateParameters);

    expect(wan.wrapper.find("[data-test='mobile-frames-error']").exists()).toBe(false);
    expect(child.emitted("validity-change")?.at(-1)).toEqual([true]);
    const frames = wan.wrapper.get("[data-test='mobile-frames']");
    expect(frames.attributes("step")).toBe("4");
    await frames.setValue("46");
    await frames.trigger("change");
    expect(wan.form.frames).toBe(45);
  });

  it("keeps temporal x2 single-pass through 257 frames and enforces chain stage limits", async () => {
    const temporal = formFor("ltx2", "ltx-2-19b-dev:fp8");
    temporal.frames = 257;
    temporal.temporalUpscale = "x2";
    temporal.outputFormat = "mp4";
    const single = mountParameters(temporal);
    expect(single.wrapper.find("[data-test='mobile-chain-cue']").exists()).toBe(false);
    expect(single.wrapper.find("[data-test='mobile-chain-reject']").exists()).toBe(false);
    expect(
      single.wrapper.getComponent(MobileGenerateParameters).emitted("validity-change")?.at(-1),
    ).toEqual([true]);
    single.wrapper.unmount();

    const overLimit = formFor("ltx2", "ltx-2-19b-distilled:fp8");
    overLimit.frames = 1305;
    overLimit.outputFormat = "mp4";
    const chained = mountParameters(overLimit);
    expect(chained.wrapper.get("[data-test='mobile-chain-reject']").text()).toContain(
      "at most 1297 frames",
    );
  });

  it("keeps a long request single-shot when chaining would lose selected settings", async () => {
    const form = formFor("ltx2", "ltx-2-19b-dev:fp8");
    form.frames = 177;
    form.negativePrompt = "flicker";
    const chain = mountParameters(form);
    const child = chain.wrapper.getComponent(MobileGenerateParameters);

    expect(chain.wrapper.find("[data-test='mobile-chain-cue']").exists()).toBe(false);
    expect(chain.wrapper.get("[data-test='mobile-single-shot-preservation-cue']").text()).toContain(
      "one 177-frame clip to preserve negative prompt",
    );
    expect(child.emitted("validity-change")?.at(-1)).toEqual([true]);

    chain.form.negativePrompt = "";
    await flushPromises();
    expect(chain.wrapper.get("[data-test='mobile-chain-cue']").text()).toContain("2 chained clips");
    expect(child.emitted("validity-change")?.at(-1)).toEqual([true]);
  });

  it("validates LTX-2 pipeline prerequisites and audio output", async () => {
    const initial = formFor("ltx2", "ltx-2-19b-distilled:fp8");
    initial.outputFormat = "mp4";
    const { wrapper, form } = mountParameters(initial);
    const child = wrapper.getComponent(MobileGenerateParameters);
    await wrapper.get("[data-test='mobile-ltx2-disclosure']").trigger("click");

    await wrapper.get("[data-test='mobile-ltx2-pipeline']").setValue("a2-vid");
    expect(wrapper.get("[data-test='mobile-ltx2-validation-error']").text()).toContain(
      "conditioning audio file",
    );
    expect(child.emitted("validity-change")?.at(-1)).toEqual([false]);
    form.audioFile = { filename: "voice.wav", base64: "AUDIO" };
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-ltx2-validation-error']").exists()).toBe(false);

    await wrapper.get("[data-test='mobile-ltx2-pipeline']").setValue("retake");
    expect(wrapper.get("[data-test='mobile-ltx2-validation-error']").text()).toContain(
      "source video",
    );
    form.sourceVideo = { filename: "take.mp4", base64: "VIDEO" };
    form.retakeRange = { start_seconds: 2, end_seconds: 1 };
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-ltx2-validation-error']").text()).toContain(
      "greater than",
    );
    form.retakeRange.end_seconds = 3;
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-ltx2-validation-error']").exists()).toBe(false);

    await wrapper.get("[data-test='mobile-ltx2-pipeline']").setValue("keyframe");
    expect(wrapper.get("[data-test='mobile-ltx2-validation-error']").text()).toContain(
      "at least two",
    );
    form.keyframes = [
      { frame: 0, image: { filename: "start.png", base64: "START" } },
      { frame: 24, image: { filename: "end.png", base64: "END" } },
    ];
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-ltx2-validation-error']").exists()).toBe(false);

    form.pipeline = null;
    form.keyframes[1]!.frame = 0;
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-ltx2-validation-error']").text()).toContain("unique");
    form.keyframes[1]!.frame = form.frames;
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-ltx2-validation-error']").text()).toContain(
      `0 to ${form.frames - 1}`,
    );
    form.keyframes = [];

    await wrapper.get("[data-test='mobile-ltx2-pipeline']").setValue("ic-lora");
    expect(wrapper.get("[data-test='mobile-ltx2-validation-error']").text()).toContain("LoRA");
    form.loras = [{ path: "style.safetensors", name: "Style", scale: 1, trainedWords: [] }];
    await flushPromises();
    expect(wrapper.find("[data-test='mobile-ltx2-validation-error']").exists()).toBe(false);

    form.pipeline = null;
    form.enableAudio = true;
    form.outputFormat = "png";
    await flushPromises();
    expect(wrapper.get("[data-test='mobile-audio-format-error']").text()).toContain("MP4");
    expect(child.emitted("validity-change")?.at(-1)).toEqual([false]);
  });

  it("disables generated audio for video-only LTX-2 checkpoints", () => {
    const initial = formFor("ltx2", "cv:3143864");
    initial.enableAudio = false;
    const { wrapper } = mountParameters(initial, [], false);

    expect(wrapper.get("[data-test='mobile-enable-audio']").attributes()).toHaveProperty(
      "disabled",
    );
    expect(wrapper.text()).toContain("Audio assets are not included with this checkpoint");
  });

  it("requires a real custom camera LoRA path", async () => {
    const initial = formFor("ltx2", "ltx-2.3-22b-distilled:fp8");
    initial.outputFormat = "mp4";
    const { wrapper } = mountParameters(initial);
    const child = wrapper.getComponent(MobileGenerateParameters);

    await wrapper.get("[data-test='mobile-camera-motion']").setValue("custom");
    expect(wrapper.get("[data-test='mobile-camera-motion-error']").text()).toContain(
      ".safetensors",
    );
    expect(child.emitted("validity-change")?.at(-1)).toEqual([false]);
    await wrapper.get("[data-test='mobile-camera-motion-custom']").setValue("camera.bin");
    expect(wrapper.get("[data-test='mobile-camera-motion-error']").text()).toContain(
      ".safetensors",
    );
    await wrapper
      .get("[data-test='mobile-camera-motion-custom']")
      .setValue("/models/camera.safetensors");
    expect(wrapper.find("[data-test='mobile-camera-motion-error']").exists()).toBe(false);
    expect(child.emitted("validity-change")?.at(-1)).toEqual([true]);
  });

  it("rejects oversized source video and conditioning audio before reading base64", async () => {
    fileToBase64.mockReset();
    const initial = formFor("ltx2", "ltx-2-19b-distilled:fp8");
    initial.outputFormat = "mp4";
    const { wrapper } = mountParameters(initial);
    await wrapper.get("[data-test='mobile-ltx2-disclosure']").trigger("click");

    const oversizedVideo = new File(["v"], "large.mov", { type: "video/quicktime" });
    Object.defineProperty(oversizedVideo, "size", { value: 64 * 1024 * 1024 + 1 });
    await attachFile(wrapper, "[data-test='mobile-ltx2-source-video']", oversizedVideo);
    expect(wrapper.get("[data-test='mobile-ltx2-media-error']").text()).toContain("64 MiB");
    expect(fileToBase64).not.toHaveBeenCalled();

    await wrapper.get("[data-test='mobile-ltx2-pipeline']").setValue("a2-vid");
    const oversizedAudio = new File(["a"], "large.wav", { type: "audio/wav" });
    Object.defineProperty(oversizedAudio, "size", { value: 64 * 1024 * 1024 + 1 });
    await attachFile(wrapper, "[data-test='mobile-ltx2-audio-file']", oversizedAudio);
    expect(wrapper.get("[data-test='mobile-ltx2-media-error']").text()).toContain("64 MiB");
    expect(fileToBase64).not.toHaveBeenCalled();
  });

  it("edits every LTX-2 advanced field and ingests conditioning files", async () => {
    fileToBase64
      .mockReset()
      .mockImplementation((file: File) => Promise.resolve(`b64:${file.name}`));
    const initial = formFor("ltx2", "ltx-2-19b-distilled:fp8");
    initial.outputFormat = "mp4";
    const { wrapper, form } = mountParameters(initial);

    expect(wrapper.find("[data-test='mobile-enable-audio']").exists()).toBe(true);
    await wrapper.get("[data-test='mobile-enable-audio']").setValue(true);
    await wrapper.get("[data-test='mobile-camera-motion']").setValue("dolly-in");
    expect(form.enableAudio).toBe(true);
    expect(form.cameraControl).toBe("dolly-in");
    expect(form.loras).toEqual([
      {
        path: "camera-control:dolly-in",
        name: "Dolly in camera control",
        scale: 1,
        trainedWords: [],
      },
    ]);
    form.loras[0]!.scale = 0.45;
    await wrapper.get("[data-test='mobile-camera-motion']").setValue("custom");
    await wrapper
      .get("[data-test='mobile-camera-motion-custom']")
      .setValue("/models/camera/custom.safetensors");
    expect(form.cameraControl).toBe("/models/camera/custom.safetensors");
    expect(form.loras).toEqual([
      {
        path: "/models/camera/custom.safetensors",
        name: "/models/camera/custom.safetensors",
        scale: 0.45,
        trainedWords: [],
      },
    ]);

    const disclosure = wrapper.get("[data-test='mobile-ltx2-disclosure']");
    expect(disclosure.attributes("aria-expanded")).toBe("false");
    await disclosure.trigger("click");
    expect(disclosure.attributes("aria-expanded")).toBe("true");

    await wrapper.get("[data-test='mobile-ltx2-pipeline']").setValue("retake");
    await wrapper.get("[data-test='mobile-ltx2-retake-start']").setValue("0.5");
    await wrapper.get("[data-test='mobile-ltx2-retake-end']").setValue("1.75");
    await wrapper.get("[data-test='mobile-ltx2-spatial']").setValue("x2");
    await wrapper.get("[data-test='mobile-ltx2-temporal']").setValue("x2");
    expect(form.retakeRange).toEqual({ start_seconds: 0.5, end_seconds: 1.75 });
    expect(form.spatialUpscale).toBe("x2");
    expect(form.temporalUpscale).toBe("x2");

    await attachFile(
      wrapper,
      "[data-test='mobile-ltx2-source-video']",
      new File(["v"], "take.mov"),
    );
    expect(form.sourceVideo).toEqual({ filename: "take.mov", base64: "b64:take.mov" });

    await attachFile(
      wrapper,
      "[data-test='mobile-ltx2-keyframe-file']",
      new File(["i"], "start.png"),
    );
    expect(form.keyframes).toEqual([
      { frame: 0, image: { filename: "start.png", base64: "b64:start.png" } },
    ]);
    await wrapper.get("[data-test='mobile-ltx2-keyframe-frame-0']").setValue("24");
    expect(form.keyframes[0]?.frame).toBe(24);

    await wrapper.get("[data-test='mobile-ltx2-pipeline']").setValue("a2-vid");
    expect(form.retakeRange).toBeNull();
    await attachFile(wrapper, "[data-test='mobile-ltx2-audio-file']", new File(["a"], "voice.wav"));
    expect(form.audioFile).toEqual({ filename: "voice.wav", base64: "b64:voice.wav" });
  });

  it("disables new camera choices when all four LoRA slots are occupied", () => {
    const initial = formFor("ltx2", "ltx-2-19b-distilled:fp8");
    initial.loras = ["one", "two", "three", "four"].map((path) => ({
      path,
      name: path,
      scale: 1,
      trainedWords: [],
    }));
    const { wrapper } = mountParameters(initial);
    const options = wrapper.get("[data-test='mobile-camera-motion']").findAll("option");
    expect(options[1]?.attributes()).toHaveProperty("disabled");
    expect(options[2]?.attributes()).toHaveProperty("disabled");
  });

  it("edits guidance overrides and blocks invalid values before submission", async () => {
    const initial = formFor("ltx2", "ltx-2-19b-distilled:fp8");
    initial.outputFormat = "mp4";
    const { wrapper, form } = mountParameters(initial);
    const child = wrapper.getComponent(MobileGenerateParameters);
    await wrapper.get("[data-test='mobile-ltx2-disclosure']").trigger("click");

    await wrapper.get("[data-test='mobile-ltx2-stg-scale']").setValue("1.25");
    await wrapper.get("[data-test='mobile-ltx2-stg-blocks']").setValue("28, nope");
    expect(form.guidanceOverrides.stgScale).toBe(1.25);
    expect(wrapper.get("[data-test='mobile-ltx2-stg-blocks-error']").text()).toContain(
      "not a block index",
    );
    expect(wrapper.get("[data-test='mobile-ltx2-guidance-count']").text()).toBe("2");
    expect(child.emitted("validity-change")?.at(-1)).toEqual([false]);

    await wrapper.get("[data-test='mobile-ltx2-stg-blocks']").setValue("28, 29");
    expect(child.emitted("validity-change")?.at(-1)).toEqual([true]);
  });
});

describe("MobileGenerateParameters — continue a video", () => {
  const extendModel = {
    name: "ltx-2-19b-distilled:fp8",
    family: "ltx2",
    supports_extend: true,
    extend_default_overlap_frames: 17,
  } as ModelEntry;

  // The LTX-2 advanced block is collapsed until it holds a value, so open it
  // explicitly before asking whether the control is present.
  async function openAdvanced(wrapper: VueWrapper): Promise<void> {
    await wrapper.get("[data-test='mobile-ltx2-disclosure']").trigger("click");
  }

  // A host that predates continuation omits `supports_extend`, so showing the
  // control would only let the user build a request the host rejects.
  it("stays hidden until the selected model advertises continuation", async () => {
    const legacy = mountParameters(formFor("ltx2", "ltx-2-19b-distilled:fp8"));
    await openAdvanced(legacy.wrapper);
    expect(legacy.wrapper.find("[data-test='mobile-ltx2-extend-video']").exists()).toBe(false);

    const capable = mountParameters(
      formFor("ltx2", "ltx-2-19b-distilled:fp8"),
      [],
      true,
      [],
      [cameraControl],
      extendModel,
    );
    await openAdvanced(capable.wrapper);
    expect(capable.wrapper.find("[data-test='mobile-ltx2-extend-video']").exists()).toBe(true);
  });

  it("reports how many frames the continuation appends", () => {
    const form = formFor("ltx2", "ltx-2-19b-distilled:fp8");
    form.frames = 97;
    form.extendVideo = { filename: "clip.mp4", base64: "AAAA" };
    const { wrapper } = mountParameters(form, [], true, [], [cameraControl], extendModel);

    // 97 rendered frames minus the 17 that re-render the source tail.
    expect(wrapper.get("[data-test='mobile-ltx2-extend-summary']").text()).toContain(
      "80 new frames",
    );
  });

  it("explains a conflicting source image before submission", () => {
    const form = formFor("ltx2", "ltx-2-19b-distilled:fp8");
    form.frames = 97;
    form.extendVideo = { filename: "clip.mp4", base64: "AAAA" };
    form.sourceImage = "AAAA";
    const { wrapper } = mountParameters(form, [], true, [], [cameraControl], extendModel);

    expect(wrapper.get("[data-test='mobile-ltx2-extend-error']").text()).toContain("source image");
  });

  it("resets the overlap when the attached video is cleared", async () => {
    const form = formFor("ltx2", "ltx-2-19b-distilled:fp8");
    form.frames = 97;
    form.extendVideo = { filename: "clip.mp4", base64: "AAAA" };
    form.extendOverlapFrames = 33;
    const { wrapper } = mountParameters(form, [], true, [], [cameraControl], extendModel);

    await wrapper.get("[data-test='mobile-ltx2-extend-clear']").trigger("click");
    expect(form.extendVideo).toBeNull();
    expect(form.extendOverlapFrames).toBeNull();
  });

  /**
   * Wan continues too (#783), seeding the render with the source clip's final
   * frame — but the field lived inside the LTX-2 pipeline disclosure, which
   * no wan checkpoint renders.
   */
  it("offers continuation for an image-conditioned wan checkpoint", () => {
    const wanExtendModel = {
      name: "wan22-i2v-a14b:q5",
      family: "wan",
      source_image: "required",
      supports_extend: true,
      extend_default_overlap_frames: 17,
    } as ModelEntry;
    const form = formFor("wan", "wan22-i2v-a14b:q5");
    form.frames = 53;
    form.extendVideo = { filename: "clip.mp4", base64: "AAAA" };
    const { wrapper } = mountParameters(form, [], true, [], [cameraControl], wanExtendModel);

    expect(wrapper.find("[data-test='mobile-ltx2-extend-video']").exists()).toBe(false);
    expect(wrapper.find("[data-test='mobile-ltx2-extend-clear']").exists()).toBe(true);
    // The LTX-2 pipeline disclosure stays away with it.
    expect(wrapper.find("[data-test='mobile-ltx2-disclosure']").exists()).toBe(false);

    // Wan's engine accepts exactly the one frame it carries, so the 8k+1
    // ladder — and the server's family-wide 17 — would both be rejected.
    const select = wrapper.get("[data-test='mobile-ltx2-extend-overlap']");
    expect(select.findAll("option").map((option) => option.text())).toEqual(["1"]);
    expect((select.element as HTMLSelectElement).value).toBe("1");
    expect(wrapper.get("[data-test='mobile-ltx2-extend-summary']").text()).toContain(
      "52 new frames",
    );
  });

  it("stays hidden for a text-to-video wan checkpoint", () => {
    const { wrapper } = mountParameters(
      formFor("wan", "wan22-t2v-a14b:q5"),
      [],
      true,
      [],
      [cameraControl],
      {
        name: "wan22-t2v-a14b:q5",
        family: "wan",
        source_image: "unsupported",
        supports_extend: false,
      } as ModelEntry,
    );
    expect(wrapper.find("[data-test='mobile-ltx2-extend-video']").exists()).toBe(false);
  });

  /**
   * The clamp is only worth anything if the clamped value is the one that
   * leaves the phone. Wan's select carries a single option, so `@change` never
   * fires and `extendOverlapFrames` stays null — an absent wire field handed
   * the host its own family-wide default of 17, which `extend_inner` refuses.
   */
  it("submits the overlap it is showing for an untouched wan continuation", () => {
    const form = formFor("wan", "wan22-i2v-a14b:q5");
    form.frames = 53;
    form.extendVideo = { filename: "clip.mp4", base64: "AAAA" };
    const { wrapper, form: live } = mountParameters(form, [], true, [], [cameraControl], {
      name: "wan22-i2v-a14b:q5",
      family: "wan",
      source_image: "required",
      supports_extend: true,
      extend_default_overlap_frames: 17,
    } as ModelEntry);

    expect(live.extendOverlapFrames).toBeNull();
    const shown = Number(
      (wrapper.get("[data-test='mobile-ltx2-extend-overlap']").element as HTMLSelectElement).value,
    );
    expect(shown).toBe(1);
    expect(buildRequest(live).extend_overlap_frames).toBe(shown);
  });
});
