import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it } from "vitest";
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

function factory(overrides: Partial<GenerateFormState> = {}) {
  return mount(Ltx2VideoControls, {
    props: { modelValue: baseForm(overrides) },
  });
}

function lastPatch(wrapper: ReturnType<typeof factory>): GenerateFormState {
  const [next] = wrapper.emitted("update:modelValue")!.at(-1) as [
    GenerateFormState,
  ];
  return next;
}

describe("Ltx2VideoControls", () => {
  afterEach(() => __testing__.resetForTest());

  it("reads a null enableAudio as on and writes false when toggled", async () => {
    const wrapper = factory({ enableAudio: null });
    await wrapper.get("[data-test='ltx2-enable-audio']").trigger("click");
    expect(lastPatch(wrapper).enableAudio).toBe(false);
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
