import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Ltx2VideoControls from "./Ltx2VideoControls.vue";
import {
  useGenerateForm,
  __testing__,
} from "../../../composables/useGenerateForm";
import type { GenerateFormState } from "../../../types";

function baseForm(
  overrides: Partial<GenerateFormState> = {},
): GenerateFormState {
  __testing__.resetForTest();
  const state = useGenerateForm().state.value;
  return { ...state, ...overrides };
}

function factory(
  overrides: Partial<GenerateFormState> = {},
  audioOutputSupported = true,
) {
  return mount(Ltx2VideoControls, {
    props: { modelValue: baseForm(overrides), audioOutputSupported },
  });
}

function lastPatch(wrapper: ReturnType<typeof factory>): GenerateFormState {
  const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
    GenerateFormState,
  ];
  return next;
}

describe("Ltx2VideoControls", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => [],
      }),
    );
  });
  afterEach(() => {
    __testing__.resetForTest();
    vi.unstubAllGlobals();
  });

  it("reads a null enableAudio as on and writes false when toggled", async () => {
    const wrapper = factory({ enableAudio: null });
    await wrapper.get("[data-test='ltx2-enable-audio']").trigger("click");
    expect(lastPatch(wrapper).enableAudio).toBe(false);
  });

  it("disables audio with an actionable reason for video-only checkpoints", async () => {
    const wrapper = factory({ enableAudio: false }, false);
    const toggle = wrapper.get("[data-test='ltx2-enable-audio']");
    expect(toggle.attributes("disabled")).toBeDefined();
    expect(toggle.attributes("aria-checked")).toBe("false");
    expect(wrapper.text()).toContain("Audio assets are not included");
    await toggle.trigger("click");
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
  });

  it("selects a pipeline mode and can return to auto", async () => {
    const wrapper = factory();
    const select = wrapper.get("[data-test='ltx2-pipeline']");
    (select.element as HTMLSelectElement).value = "distilled";
    await select.trigger("change");
    expect(lastPatch(wrapper).pipeline).toBe("distilled");

    (select.element as HTMLSelectElement).value = "";
    await select.trigger("change");
    expect(lastPatch(wrapper).pipeline).toBe(null);
  });

  it("renders host-provided controls and their guide copy", async () => {
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => [
        {
          id: "motion-track",
          label: "Motion track",
          guide: "A trajectory-overlay guide video.",
          size_bytes: 327_309_314,
          installed: false,
          download_model: "ltx2-control-motion-track-23",
          download_repo: "Lightricks/control",
          download_filename: "control.safetensors",
          download_sha256: "a".repeat(64),
        },
      ],
    } as Response);
    const wrapper = factory({ model: "ltx-2.3-22b-distilled:fp8" });
    await vi.waitFor(() =>
      expect(
        wrapper.get("[data-test='ltx2-reference-control']").findAll("option"),
      ).toHaveLength(2),
    );
    await wrapper
      .get("[data-test='ltx2-reference-control']")
      .setValue("motion-track");
    expect(lastPatch(wrapper).pipeline).toBe("ic-lora");
    expect(lastPatch(wrapper).icLoraControl).toBe("motion-track");
    await wrapper.setProps({
      modelValue: {
        ...wrapper.props("modelValue"),
        pipeline: "ic-lora",
        icLoraControl: "motion-track",
      },
    });
    expect(wrapper.get("[data-test='ltx2-reference-guide']").text()).toContain(
      "trajectory-overlay",
    );
  });

  it("offers the seven LTX-2 camera presets and writes the selected motion", async () => {
    const wrapper = factory({ model: "ltx-2-19b-distilled:fp8" });
    const select = wrapper.get("[data-test='ltx2-camera-motion']");
    expect(select.findAll("option").map((option) => option.text())).toEqual([
      "None",
      "Dolly in",
      "Dolly left",
      "Dolly out",
      "Dolly right",
      "Jib down",
      "Jib up",
      "Static",
      "Custom LoRA path…",
    ]);
    await select.setValue("jib-up");
    expect(lastPatch(wrapper).cameraControl).toBe("jib-up");
  });

  it("keeps presets disabled for LTX-2.3 and accepts a custom camera LoRA", async () => {
    const wrapper = factory({ model: "ltx-2.3-22b-distilled:fp8" });
    const select = wrapper.get("[data-test='ltx2-camera-motion']");
    const presets = select.findAll("option").slice(1, 8);
    expect(
      presets.every((option) => option.attributes("disabled") !== undefined),
    ).toBe(true);

    await select.setValue("custom");
    await wrapper
      .get("[data-test='ltx2-camera-motion-custom']")
      .setValue("/models/camera/pan.safetensors");
    expect(lastPatch(wrapper).cameraControl).toBe(
      "/models/camera/pan.safetensors",
    );
  });

  it("maps the spatial segmented control to the upscale field", async () => {
    const wrapper = factory();
    const buttons = wrapper.get("[data-test='ltx2-spatial']").findAll("button");
    await buttons[2]!.trigger("click"); // Native / 1.5× / 2×
    expect(lastPatch(wrapper).spatialUpscale).toBe("x2");
  });

  it("maps the temporal segmented control to the upscale field", async () => {
    const wrapper = factory();
    const buttons = wrapper
      .get("[data-test='ltx2-temporal']")
      .findAll("button");
    await buttons[1]!.trigger("click"); // Native / 2×
    expect(lastPatch(wrapper).temporalUpscale).toBe("x2");
  });

  it("builds a retake range from the start/end inputs", async () => {
    const wrapper = factory();
    const start = wrapper.get("[data-test='ltx2-retake-start']");
    (start.element as HTMLInputElement).value = "1.5";
    await start.trigger("input");
    expect(lastPatch(wrapper).retakeRange).toEqual({
      start_seconds: 1.5,
      end_seconds: 1,
    });
    const end = wrapper.get("[data-test='ltx2-retake-end']");
    (end.element as HTMLInputElement).value = "3";
    await end.trigger("input");
    expect(lastPatch(wrapper).retakeRange?.end_seconds).toBe(3);
  });

  it("round-trips the server audio path field", async () => {
    const wrapper = factory();
    const input = wrapper.get("[data-test='ltx2-audio-path']");
    (input.element as HTMLInputElement).value = "/srv/audio.wav";
    await input.trigger("input");
    expect(lastPatch(wrapper).audioFilePath).toBe("/srv/audio.wav");
  });

  it("disables the audio path when an audio file is attached", () => {
    const wrapper = factory({
      audioFile: { kind: "upload", filename: "a.wav", base64: "AA" },
    });
    const input = wrapper.get("[data-test='ltx2-audio-path']");
    expect((input.element as HTMLInputElement).disabled).toBe(true);
  });

  it("clears an attached audio file", async () => {
    const wrapper = factory({
      audioFile: { kind: "upload", filename: "a.wav", base64: "AA" },
    });
    await wrapper.get("[data-test='ltx2-audio-clear']").trigger("click");
    expect(lastPatch(wrapper).audioFile).toBe(null);
  });

  it("edits and removes keyframes", async () => {
    const wrapper = factory({
      keyframes: [
        {
          frame: 0,
          image: { kind: "upload", filename: "k.png", base64: "AA" },
        },
      ],
    });
    const frame = wrapper.get("[data-test='ltx2-keyframe-frame']");
    (frame.element as HTMLInputElement).value = "72";
    await frame.trigger("input");
    expect(lastPatch(wrapper).keyframes[0]?.frame).toBe(72);

    await wrapper.get("[data-test='ltx2-keyframe-remove']").trigger("click");
    expect(lastPatch(wrapper).keyframes).toEqual([]);
  });
});
