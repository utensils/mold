import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import MinimaxH3AuthoringPanel from "./MinimaxH3AuthoringPanel.vue";
import type { MinimaxH3AuthoringState } from "../lib/minimaxH3Authoring";
import { emptyMinimaxH3AuthoringState } from "../lib/minimaxH3Authoring";
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
          media: { authority: "inline", data: "IMAGE" },
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
  it("presents the reviewed first-frame-only contract", () => {
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: {
        modelValue: emptyMinimaxH3AuthoringState(),
        task: "fl2va",
        requiredEndpoint: "first",
      },
    });

    expect(wrapper.get("[data-test='h3-mode']").text()).toBe(
      "first-frame-required",
    );
    expect(wrapper.text()).toContain("Required");
    expect(wrapper.text()).toContain("accepts no other endpoint mode");
    expect(wrapper.find("[data-test='h3-last-file']").exists()).toBe(false);
  });

  it("keeps an incompatible restored last frame removable", () => {
    const restored = emptyMinimaxH3AuthoringState();
    restored.lastFrame = {
      filename: "old-last.png",
      mimeType: "image/png",
      width: 1344,
      height: 768,
      data: "LAST",
    };
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: {
        modelValue: restored,
        task: "fl2va",
        requiredEndpoint: "first",
      },
    });

    expect(wrapper.find("[data-test='h3-last-file']").exists()).toBe(true);
    expect(
      wrapper.get("[data-test='h3-last-file']").attributes("disabled"),
    ).toBe("");
    expect(wrapper.get("[data-test='h3-mode']").text()).toBe(
      "first-frame-required",
    );
    expect(wrapper.text()).toContain("incompatible; remove");
    expect(wrapper.get("[data-test='h3-last-remove']").text()).toBe("Remove");
  });

  it("never presents a restored first-plus-last pair as supported", () => {
    const restored = emptyMinimaxH3AuthoringState();
    restored.firstFrame = {
      filename: "first.png",
      mimeType: "image/png",
      width: 1344,
      height: 768,
      data: "FIRST",
    };
    restored.lastFrame = {
      filename: "old-last.png",
      mimeType: "image/png",
      width: 1344,
      height: 768,
      data: "LAST",
    };
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: {
        modelValue: restored,
        task: "fl2va",
        requiredEndpoint: "first",
      },
    });

    expect(wrapper.get("[data-test='h3-mode']").text()).toBe(
      "first-frame-to-audio-video",
    );
    expect(wrapper.text()).not.toContain("first-and-last-frame-to-audio-video");
  });

  it("renders semantic resynthesis, one-based mixed order, soundtrack association, and budgets", () => {
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: state(), task: "ref2va" },
    });
    expect(wrapper.text()).toContain("Reference-guided semantic resynthesis");
    expect(wrapper.text()).toContain("not pixel-aligned");
    expect(wrapper.text()).toContain("subject.png");
    expect(wrapper.text()).toContain("motion.mp4");
    expect(wrapper.text()).toContain("soundtrack attached");
    expect(wrapper.text()).toContain("Reattach original media");
    expect(wrapper.get('[data-test="h3-reference-budget"]').text()).toContain(
      "3/12",
    );
  });

  it("offers an explicit same-slot reattach control for redacted provenance", () => {
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: state(), task: "ref2va" },
    });

    const input = wrapper.get('[data-test="h3-reference-reattach-2"]');
    expect(input.attributes("accept")).toBe(
      ".wav,audio/wav,audio/x-wav,audio/wave",
    );
    expect(input.element.closest("label")?.getAttribute("aria-label")).toBe(
      "Reattach reference 3",
    );
  });

  it("offers accessible 44pt reorder alternatives without regrouping kinds", async () => {
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: { modelValue: state(), task: "ref2va", touchFriendly: true },
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
        task: "ref2va",
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
        task: "ref2va",
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

  it("renders all four FL2VA endpoint modes and removes endpoints explicitly", async () => {
    const first = {
      filename: "first.png",
      mimeType: "image/png",
      width: 1280,
      height: 720,
      data: "FIRST",
    };
    const wrapper = mount(MinimaxH3AuthoringPanel, {
      props: {
        modelValue: {
          firstFrame: first,
          lastFrame: { ...first, filename: "last.png" },
          references: [],
        },
        task: "fl2va",
      },
    });
    expect(wrapper.get('[data-test="h3-mode"]').text()).toBe(
      "first-and-last-frame-to-audio-video",
    );
    await wrapper.get('[data-test="h3-first-remove"]').trigger("click");
    const emitted = wrapper.emitted(
      "update:modelValue",
    )?.[0]?.[0] as MinimaxH3AuthoringState;
    expect(emitted.firstFrame).toBeNull();
    expect(emitted.lastFrame?.filename).toBe("last.png");
  });
});
