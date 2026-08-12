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
  extra: { canExtend?: boolean; extendDefaultOverlapFrames?: number } = {},
) {
  return mount(Ltx2VideoControls, {
    props: {
      modelValue: baseForm(overrides),
      audioOutputSupported,
      ...extra,
    },
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

  it("routes the lip-dub adapter to its own pipeline and explains the timing", async () => {
    vi.mocked(fetch).mockImplementation(
      async (input) =>
        ({
          ok: true,
          json: async () =>
            String(input).includes("ltx2-control-adapters")
              ? [
                  {
                    id: "lipdub",
                    label: "Lip dub",
                    guide: "A reference video with speech.",
                    size_bytes: 2_466_665_072,
                    installed: false,
                    gated: true,
                    download_model: "ltx2-control-lipdub-23",
                    download_repo: "Lightricks/LTX-2.3-22b-IC-LoRA-DubIt",
                    download_filename:
                      "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors",
                    download_sha256: "a".repeat(64),
                  },
                ]
              : [],
        }) as Response,
    );
    const wrapper = factory({ model: "ltx-2.3-22b-distilled:fp8" });
    await vi.waitFor(() =>
      expect(
        wrapper.get("[data-test='ltx2-reference-control']").findAll("option"),
      ).toHaveLength(2),
    );

    await wrapper
      .get("[data-test='ltx2-reference-control']")
      .setValue("lipdub");

    // Not `ic-lora`: that pipeline drops the adapter before stage 2 and never
    // conditions on the reference voice, and the server rejects the pairing.
    expect(lastPatch(wrapper).pipeline).toBe("lip-dub");
    expect(lastPatch(wrapper).icLoraControl).toBe("lipdub");

    await wrapper.setProps({ modelValue: lastPatch(wrapper) });
    expect(wrapper.get("[data-test='ltx2-lip-dub-hint']").text()).toContain(
      "reference video",
    );
  });

  it("renders host-provided controls and their guide copy", async () => {
    vi.mocked(fetch).mockImplementation(
      async (input) =>
        ({
          ok: true,
          json: async () =>
            String(input).includes("ltx2-control-adapters")
              ? [
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
                ]
              : [],
        }) as Response,
    );
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
      modelValue: lastPatch(wrapper),
    });
    expect(wrapper.get("[data-test='ltx2-reference-guide']").text()).toContain(
      "trajectory-overlay",
    );
  });

  it("keeps IC options and camera state when an older host lacks the camera endpoint", async () => {
    vi.mocked(fetch).mockImplementation(async (input) => {
      if (String(input).includes("ltx2-camera-controls")) {
        return { ok: false, status: 404 } as Response;
      }
      return {
        ok: true,
        json: async () => [
          {
            id: "pose",
            label: "Pose control",
            guide: "A pose guide.",
            size_bytes: 1,
            installed: true,
            download_model: "pose",
            download_repo: "Lightricks/pose",
            download_filename: "pose.safetensors",
            download_sha256: "a".repeat(64),
          },
        ],
      } as Response;
    });
    const wrapper = factory({
      model: "ltx-2-19b-distilled:fp8",
      cameraControl: "dolly-in",
    });
    await vi.waitFor(() =>
      expect(
        wrapper.get("[data-test='ltx2-reference-control']").findAll("option"),
      ).toHaveLength(2),
    );
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
    expect(
      wrapper.find("[data-test='ltx2-camera-motion-19b-hint']").exists(),
    ).toBe(false);
  });

  it("offers compatible camera presets with download copy and writes the selected motion", async () => {
    vi.mocked(fetch).mockImplementation(
      async (input) =>
        ({
          ok: true,
          json: async () =>
            String(input).includes("ltx2-camera-controls")
              ? [
                  {
                    id: "jib-up",
                    label: "Jib up",
                    size_bytes: 327_309_208,
                    installed: false,
                    download_model: "ltx2-camera-control-jib-up-19b",
                    download_repo: "Lightricks/camera",
                    download_filename: "jib-up.safetensors",
                    download_sha256: "a".repeat(64),
                  },
                ]
              : [],
        }) as Response,
    );
    const wrapper = factory({ model: "ltx-2-19b-distilled:fp8" });
    await vi.waitFor(() =>
      expect(
        wrapper.get("[data-test='ltx2-camera-motion']").findAll("option"),
      ).toHaveLength(3),
    );
    const select = wrapper.get("[data-test='ltx2-camera-motion']");
    expect(select.findAll("option").map((option) => option.text())).toEqual([
      "None",
      "Jib up · downloads on first use",
      "Custom LoRA path…",
    ]);
    await select.setValue("jib-up");
    expect(lastPatch(wrapper).cameraControl).toBe("jib-up");
    expect(lastPatch(wrapper).loras).toEqual([
      { path: "camera-control:jib-up", scale: 1, trainedWords: [] },
    ]);

    await wrapper.setProps({
      modelValue: {
        ...lastPatch(wrapper),
        loras: [
          { path: "camera-control:jib-up", scale: 0.45, trainedWords: [] },
        ],
      },
    });
    await select.setValue("custom");
    await wrapper.setProps({ modelValue: lastPatch(wrapper) });
    await wrapper
      .get("[data-test='ltx2-camera-motion-custom']")
      .setValue("/models/camera/custom.safetensors");
    expect(lastPatch(wrapper).loras).toEqual([
      {
        path: "/models/camera/custom.safetensors",
        scale: 0.45,
        trainedWords: [],
      },
    ]);
  });

  it("disables camera choices when all four LoRA slots are occupied", async () => {
    vi.mocked(fetch).mockImplementation(
      async (input) =>
        ({
          ok: true,
          json: async () =>
            String(input).includes("ltx2-camera-controls")
              ? [{ id: "dolly-in", label: "Dolly in", installed: false }]
              : [],
        }) as Response,
    );
    const wrapper = factory({
      model: "ltx-2-19b-distilled:fp8",
      loras: ["one", "two", "three", "four"].map((path) => ({
        path,
        scale: 1,
        trainedWords: [],
      })),
    });
    await vi.waitFor(() =>
      expect(
        wrapper.get("[data-test='ltx2-camera-motion']").findAll("option"),
      ).toHaveLength(3),
    );
    const options = wrapper
      .get("[data-test='ltx2-camera-motion']")
      .findAll("option");
    expect(options[1]?.attributes()).toHaveProperty("disabled");
    expect(options[2]?.attributes()).toHaveProperty("disabled");
  });

  it("shows 19B guidance without disabled rows and accepts a custom camera LoRA", async () => {
    const wrapper = factory({ model: "ltx-2.3-22b-distilled:fp8" });
    await vi.waitFor(() =>
      expect(
        wrapper.find("[data-test='ltx2-camera-motion-19b-hint']").exists(),
      ).toBe(true),
    );
    const select = wrapper.get("[data-test='ltx2-camera-motion']");
    expect(select.findAll("option").map((option) => option.text())).toEqual([
      "None",
      "Custom LoRA path…",
    ]);
    expect(
      wrapper.get("[data-test='ltx2-camera-motion-19b-hint']").text(),
    ).toContain("LTX-2 19B");

    await select.setValue("custom");
    await wrapper
      .get("[data-test='ltx2-camera-motion-custom']")
      .setValue("/models/camera/pan.safetensors");
    expect(lastPatch(wrapper).cameraControl).toBe(
      "/models/camera/pan.safetensors",
    );
    expect(lastPatch(wrapper).loras).toEqual([
      {
        path: "/models/camera/pan.safetensors",
        scale: 1,
        trainedWords: [],
      },
    ]);
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

  it("leaves guidance overrides absent until a control is used", async () => {
    const wrapper = factory();
    const stg = wrapper.get("[data-test='ltx2-stg-scale']");
    expect((stg.element as HTMLInputElement).value).toBe("");
    expect(wrapper.find("[data-test='ltx2-guidance-count']").exists()).toBe(
      false,
    );

    (stg.element as HTMLInputElement).value = "1.5";
    await stg.trigger("input");
    expect(lastPatch(wrapper).guidanceOverrides).toMatchObject({
      stgScale: 1.5,
      rescaleScale: null,
    });
  });

  it("clears one override back to the pipeline default", async () => {
    const wrapper = factory({
      guidanceOverrides: {
        stgScale: 1.5,
        stgBlocks: "",
        rescaleScale: null,
        modalityScale: null,
        skipStep: null,
      },
    });
    expect(wrapper.get("[data-test='ltx2-guidance-count']").text()).toBe("1");

    const stg = wrapper.get("[data-test='ltx2-stg-scale']");
    (stg.element as HTMLInputElement).value = "";
    await stg.trigger("input");
    expect(lastPatch(wrapper).guidanceOverrides?.stgScale).toBeNull();
  });

  it("reports an unusable STG block list inline", async () => {
    const wrapper = factory({
      guidanceOverrides: {
        stgScale: null,
        stgBlocks: "28,twenty-nine",
        rescaleScale: null,
        modalityScale: null,
        skipStep: null,
      },
    });
    expect(wrapper.get("[data-test='ltx2-stg-blocks-error']").text()).toContain(
      "not a block index",
    );
  });

  it("resets every override at once", async () => {
    const wrapper = factory({
      guidanceOverrides: {
        stgScale: 1.5,
        stgBlocks: "29",
        rescaleScale: 0.5,
        modalityScale: 2,
        skipStep: 1,
      },
    });
    await wrapper.get("[data-test='ltx2-guidance-reset']").trigger("click");
    expect(lastPatch(wrapper).guidanceOverrides).toEqual({
      stgScale: null,
      stgBlocks: "",
      rescaleScale: null,
      modalityScale: null,
      skipStep: null,
    });
  });

  it("reports a fractional skip stride inline", async () => {
    const wrapper = factory({
      guidanceOverrides: {
        stgScale: null,
        stgBlocks: "",
        rescaleScale: null,
        modalityScale: null,
        skipStep: 1.5,
      },
    });
    expect(
      wrapper.get("[data-test='ltx2-guidance-skip-step-error']").text(),
    ).toContain("whole number");
  });

  it("renders without the field for templates saved before it existed", () => {
    const wrapper = factory({ guidanceOverrides: undefined });
    expect(
      (wrapper.get("[data-test='ltx2-stg-scale']").element as HTMLInputElement)
        .value,
    ).toBe("");
  });
});
