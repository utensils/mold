import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import MinimaxH3AuthoringPanel from "./MinimaxH3AuthoringPanel.vue";
import type { MinimaxH3AuthoringState } from "../lib/minimaxH3Authoring";
import {
  h264AacMp4Fixture,
  pcmWavFixture,
} from "../lib/minimaxH3MediaProbe.testFixtures";

function state(): MinimaxH3AuthoringState {
  return {
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
      {
        reference: {
          kind: "video",
          media: { authority: "upload", handle: "video-handle" },
          provenance: { name: "motion.mp4", sha256: "b".repeat(64) },
          mime_type: "video/mp4",
          width: 1280,
          height: 720,
          frame_count: 96,
          duration_ms: 4_000,
          fps: 24,
          has_audio: true,
          audio_duration_ms: 4_000,
          audio_sample_count: 192_000,
          audio_sample_rate: 48_000,
          audio_channels: 2,
        },
      },
      {
        reference: {
          kind: "audio",
          media: { authority: "descriptor" },
          provenance: { name: "voice.wav", sha256: "c".repeat(64) },
          mime_type: "audio/wav",
          duration_ms: 3_000,
          sample_rate: 48_000,
          channels: 2,
          sample_count: 144_000,
        },
      },
    ],
  };
}

function fileBuffer(bytes: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

describe("MinimaxH3AuthoringPanel", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn(async () => ({ width: 1_024, height: 768, close: vi.fn() })),
    );
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: vi.fn(),
    } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, "toDataURL").mockReturnValue(
      "data:image/jpeg;base64,BOUNDED",
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders semantic resynthesis, one-based mixed order, bounded thumbnails, soundtrack association, and budgets", async () => {
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: state() },
    });
    expect(wrapper.text()).toContain("Reference-guided semantic resynthesis");
    expect(wrapper.text()).toContain("not pixel-aligned");
    expect(wrapper.text()).toContain("subject.png");
    expect(wrapper.text()).toContain("motion.mp4");
    expect(wrapper.text()).toContain("soundtrack attached");
    expect(wrapper.text()).toContain("Reattach original media");
    await vi.waitFor(() => {
      expect(wrapper.get(".h3-authoring__preview img").attributes("src")).toBe(
        "data:image/jpeg;base64,BOUNDED",
      );
    });
    expect(HTMLCanvasElement.prototype.toDataURL).toHaveBeenCalledWith(
      "image/jpeg",
      0.78,
    );
    expect(createImageBitmap).toHaveBeenCalledWith(expect.any(Blob), {
      resizeWidth: 112,
      resizeHeight: 84,
      resizeQuality: "high",
    });
    expect(wrapper.get('[data-test="h3-reference-budget"]').text()).toContain(
      "3/12",
    );
  });

  it("opens the established image picker without duplicating gallery logic", async () => {
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: state(), imagePickerAvailable: true },
    });
    await wrapper.get('[data-test="h3-reference-library"]').trigger("click");
    expect(wrapper.emitted("open-image-picker")).toHaveLength(1);
  });

  it("offers an explicit same-slot reattach control for redacted provenance", () => {
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: state() },
    });

    const input = wrapper.get('[data-test="h3-reference-reattach-2"]');
    expect(input.attributes("accept")).toBe(
      ".wav,audio/wav,audio/x-wav,audio/wave",
    );
    expect(input.element.closest("label")?.getAttribute("aria-label")).toBe(
      "Reattach reference 3",
    );
  });

  it("keeps unavailable image copy and its complete action row as separate layout regions", () => {
    const value = state();
    value.references = [
      {
        reference: {
          kind: "image",
          media: { authority: "descriptor" },
          provenance: { name: "missing-subject.png", sha256: "d".repeat(64) },
          mime_type: "image/png",
          width: 1_024,
          height: 768,
        },
      },
    ];
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: value },
    });

    const row = wrapper.get('[data-test="h3-reference-0"]');
    expect(row.get(".h3-authoring__reference-copy").text()).toContain(
      "Reattach original media before generating.",
    );
    expect(row.get('[aria-label="Reference 1 controls"]')).toBeTruthy();
    expect(row.get('[data-test="h3-reference-reattach-0"]')).toBeTruthy();
    expect(
      row.get('[data-test="h3-reference-crop-0"]').attributes("disabled"),
    ).toBeDefined();
    expect(row.get('[data-test="h3-reference-remove-0"]')).toBeTruthy();
  });

  it("offers accessible 44pt reorder alternatives without regrouping kinds", async () => {
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: state(), touchFriendly: true },
    });
    const later = wrapper.get('[data-test="h3-reference-down-0"]');
    expect(later.attributes("aria-label")).toBe("Move reference 1 later");
    await later.trigger("click");
    const emitted = wrapper.emitted(
      "update:modelValue",
    )?.[0]?.[0] as MinimaxH3AuthoringState;
    expect(emitted.references.map((draft) => draft.reference.kind)).toEqual([
      "video",
      "image",
      "audio",
    ]);
  });

  it("canonicalizes audio/x-wav without resampling or rounding its sample timeline", async () => {
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: {
        modelValue: { firstFrame: null, lastFrame: null, references: [] },
      },
    });
    const bytes = pcmWavFixture({
      sampleRate: 44_100,
      channels: 2,
      sampleCount: 44_101,
    });
    const file = new File([fileBuffer(bytes)], "fractional.wav", {
      type: "audio/x-wav",
    });
    const input = wrapper.get('[data-test="h3-reference-files"]');
    Object.defineProperty(input.element, "files", {
      configurable: true,
      value: [file],
    });

    await input.trigger("change");
    await vi.waitFor(() => {
      expect(wrapper.emitted("update:modelValue")).toHaveLength(1);
    });

    const emitted = wrapper.emitted(
      "update:modelValue",
    )?.[0]?.[0] as MinimaxH3AuthoringState;
    expect(emitted.references[0]?.reference).toMatchObject({
      kind: "audio",
      mime_type: "audio/wav",
      duration_ms: 1_001,
      sample_rate: 44_100,
      channels: 2,
      sample_count: 44_101,
      provenance: { name: "fractional.wav" },
    });
  });

  it("authors H.264/AAC upload hints without browser audio resampling", async () => {
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: {
        modelValue: { firstFrame: null, lastFrame: null, references: [] },
      },
    });
    const bytes = h264AacMp4Fixture();
    const file = new File([fileBuffer(bytes)], "motion.mp4", {
      type: "video/mp4",
    });
    const input = wrapper.get('[data-test="h3-reference-files"]');
    Object.defineProperty(input.element, "files", {
      configurable: true,
      value: [file],
    });

    await input.trigger("change");
    await vi.waitFor(() => {
      expect(wrapper.emitted("update:modelValue")).toHaveLength(1);
    });

    const emitted = wrapper.emitted(
      "update:modelValue",
    )?.[0]?.[0] as MinimaxH3AuthoringState;
    expect(emitted.references[0]?.reference).toMatchObject({
      kind: "video",
      mime_type: "video/mp4",
      width: 1_280,
      height: 720,
      frame_count: 48,
      duration_ms: 2_000,
      fps: 24,
      has_audio: true,
      audio_duration_ms: 2_021,
      audio_sample_count: 89_088,
      audio_sample_rate: 44_100,
      audio_channels: 2,
      provenance: { name: "motion.mp4" },
    });
  });
});

describe("MinimaxH3AuthoringPanel — reference crop", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn(async () => ({ width: 1_024, height: 768, close: vi.fn() })),
    );
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: vi.fn(),
    } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, "toDataURL").mockReturnValue(
      "data:image/jpeg;base64,BOUNDED",
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("offers Crop on attached image rows only and asks the host to open the editor", async () => {
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: state() },
    });
    const crop = wrapper.get('[data-test="h3-reference-crop-0"]');
    expect(crop.attributes("aria-label")).toBe("Crop reference 1");
    expect(wrapper.find('[data-test="h3-reference-crop-1"]').exists()).toBe(
      false,
    );
    expect(wrapper.find('[data-test="h3-reference-crop-2"]').exists()).toBe(
      false,
    );
    await crop.trigger("click");
    expect(wrapper.emitted("crop-reference")).toEqual([[0]]);
  });

  it("disables Crop until a redacted image's bytes are reattached", () => {
    const value = state();
    value.references[0]!.reference.media = { authority: "descriptor" };
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: value },
    });
    expect(
      wrapper.get('[data-test="h3-reference-crop-0"]').attributes("disabled"),
    ).toBeDefined();
  });

  it("draws the pending crop's outline over the thumbnail and names its size", () => {
    const value = state();
    value.references[0]!.crop = { x: 256, y: 0, width: 512, height: 768 };
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: value },
    });
    const outline = wrapper.get('[data-test="h3-reference-crop-outline-0"]');
    expect(outline.attributes("style")).toContain("left: 25%");
    expect(outline.attributes("style")).toContain("width: 50%");
    expect(wrapper.text()).toContain("cropped to 512×768");
    expect(
      wrapper.find('[data-test="h3-reference-crop-outline-1"]').exists(),
    ).toBe(false);
  });

  it("clears a saved crop with an inline disclosure when a different original is reattached", async () => {
    const value = state();
    value.references[0]!.reference.media = { authority: "descriptor" };
    value.references[0]!.crop = { x: 256, y: 0, width: 512, height: 768 };
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: value },
    });
    const input = wrapper.get('[data-test="h3-reference-reattach-0"]');
    Object.defineProperty(input.element, "files", {
      configurable: true,
      value: [
        new File([new Uint8Array([1, 2, 3])], "other.png", {
          type: "image/png",
        }),
      ],
    });
    await input.trigger("change");
    await vi.waitFor(() => {
      expect(wrapper.emitted("update:modelValue")).toHaveLength(1);
    });
    const emitted = wrapper.emitted(
      "update:modelValue",
    )?.[0]?.[0] as MinimaxH3AuthoringState;
    expect(emitted.references[0]).not.toHaveProperty("crop");
    expect(emitted.references[0]?.reference.provenance?.name).toBe("other.png");
    expect(wrapper.get('[data-test="h3-reference-notice"]').text()).toContain(
      "saved crop was cleared",
    );
  });
});
