import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import ScriptComposer from "./ScriptComposer.vue";
import * as api from "../api";

// Stub the network at the module boundary. ScriptComposer renders
// ImagePickerModal, which calls listGallery() on mount, so a spy on
// fetchChainLimits alone still let a real request escape and surface as an
// unhandled ECONNREFUSED that failed the whole run despite every assertion
// passing.
vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  fetchChainLimits: vi.fn(),
  listGallery: vi.fn(async () => []),
}));

/** Minimum props ScriptComposer needs. We always mount with an LTX-2 model
 * by default so the audio toggle is in scope; non-AV cases override `model`
 * AND the mocked chain-limits response below. */
function props() {
  return {
    model: "ltx-2.3-22b-distilled:fp8",
    width: 1216,
    height: 704,
    fps: 24,
    family: "ltx2",
  };
}

function ltx2Limits(): api.ChainLimits {
  return {
    model: "ltx-2.3-22b-distilled:fp8",
    frames_per_clip_cap: 97,
    frames_per_clip_recommended: 97,
    max_stages: 16,
    max_total_frames: 97 * 16,
    fade_frames_max: 32,
    transition_modes: ["smooth", "cut", "fade"],
    quantization_family: "fp8",
    supports_audio: true,
    supports_sequence: true,
  };
}

function ltxVideoLimits(): api.ChainLimits {
  // LTX-Video is chain-capable but has no audio path — toggle must hide.
  return {
    model: "ltx-video-0.9.7-distilled:fp8",
    frames_per_clip_cap: 97,
    frames_per_clip_recommended: 97,
    max_stages: 16,
    max_total_frames: 97 * 16,
    fade_frames_max: 32,
    transition_modes: ["smooth", "cut", "fade"],
    quantization_family: "fp8",
    supports_audio: false,
    supports_sequence: true,
  };
}

describe("ScriptComposer — audio toggle visibility & default", () => {
  beforeEach(() => {
    localStorage.clear();
    // Standing stub so a re-fetch (model change, remount) never falls through
    // to the real implementation and hits the network; the per-test
    // mockResolvedValueOnce calls below still take precedence.
    vi.spyOn(api, "fetchChainLimits").mockResolvedValue(ltx2Limits());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the audio toggle for AV-capable (LTX-2) models", async () => {
    vi.spyOn(api, "fetchChainLimits").mockResolvedValueOnce(ltx2Limits());
    const w = mount(ScriptComposer, { props: props() });
    await flushPromises();

    const toggle = w.find('[data-test="script-composer-enable-audio"]');
    expect(toggle.exists()).toBe(true);
    // Default-on for AV models so the user gets audio without an extra
    // click — matches the same default in single-mode useGenerateForm.
    expect((toggle.element as HTMLInputElement).checked).toBe(true);
  });

  it("hides the audio toggle for chain-capable but video-only families (LTX-Video)", async () => {
    vi.spyOn(api, "fetchChainLimits").mockResolvedValueOnce(ltxVideoLimits());
    const w = mount(ScriptComposer, {
      props: {
        ...props(),
        model: "ltx-video-0.9.7-distilled:fp8",
        family: "ltx-video",
      },
    });
    await flushPromises();

    const toggle = w.find('[data-test="script-composer-enable-audio"]');
    expect(toggle.exists()).toBe(false);
    expect(w.text()).toContain("Join clips");
    expect(
      w
        .findAll(".sc__frames option")
        .map((option) => Number(option.attributes("value"))),
    ).toContain(9);
  });

  it("hides the audio toggle when chain-limits fails (model not chain-capable)", async () => {
    vi.spyOn(api, "fetchChainLimits").mockRejectedValueOnce(
      new Error("not chain-capable"),
    );
    const w = mount(ScriptComposer, {
      props: { ...props(), model: "flux-dev:q4", family: "flux" },
    });
    await flushPromises();

    expect(w.find('[data-test="script-composer-enable-audio"]').exists()).toBe(
      false,
    );
  });

  it("shows the pipeline reason and disables Generate when the model cannot sequence", async () => {
    vi.spyOn(api, "fetchChainLimits").mockResolvedValueOnce({
      ...ltx2Limits(),
      model: "ltx-2.3-22b-dev:fp8",
      supports_sequence: false,
      sequence_unsupported_reason:
        "This checkpoint selects the two-stage LTX-2 pipeline.",
    });
    const w = mount(ScriptComposer, {
      props: { ...props(), model: "ltx-2.3-22b-dev:fp8" },
    });
    await flushPromises();

    expect(w.get('[data-test="chain-pipeline-unsupported"]').text()).toContain(
      "two-stage LTX-2",
    );
    expect(
      (w.get('[data-test="script-generate"]').element as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  it("starts a new sequence with two clips and plain-language transitions", async () => {
    vi.spyOn(api, "fetchChainLimits").mockResolvedValue(ltx2Limits());
    const w = mount(ScriptComposer, { props: props() });
    await flushPromises();

    expect(w.findAll('[data-test="stage-card"]')).toHaveLength(2);
    expect(w.get('[data-test="script-summary"]').text()).toContain("2 clips");
    expect(w.text()).toContain("Smooth");
    expect(w.get('[data-test="script-generate"]').text()).toBe(
      "Generate sequence",
    );
  });

  it("requires every clip prompt before generation", async () => {
    vi.spyOn(api, "fetchChainLimits").mockResolvedValue(ltx2Limits());
    const w = mount(ScriptComposer, { props: props() });
    await flushPromises();

    expect(w.get('[data-test="sequence-validation"]').text()).toContain(
      "Describe clip 1",
    );
    expect(
      (w.get('[data-test="script-generate"]').element as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  it("clears stale enable_audio from the persisted draft when the current model has no audio path", async () => {
    // Reproduces the reported bug: the user generated with LTX-2.3 (which
    // sets `enable_audio: true` in the persisted chain draft), then
    // reloaded the page with a video-only chain-capable model. Without
    // this normalisation the draft's `enable_audio: true` would survive
    // mount and the chain endpoint would 400 on submit.
    localStorage.setItem(
      "mold.chain.draft.v2",
      JSON.stringify({
        schema: "mold.chain.v1",
        chain: {
          model: "ltx-video-0.9.7-distilled:fp8",
          width: 1216,
          height: 704,
          fps: 24,
          steps: 8,
          guidance: 3.0,
          strength: 1.0,
          motion_tail_frames: 17,
          output_format: "mp4",
          enable_audio: true, // ← stale from a prior LTX-2.3 session
        },
        stage: [{ prompt: "x", frames: 97 }],
      }),
    );
    vi.spyOn(api, "fetchChainLimits").mockResolvedValueOnce(ltxVideoLimits());
    const w = mount(ScriptComposer, {
      props: {
        ...props(),
        model: "ltx-video-0.9.7-distilled:fp8",
        family: "ltx-video",
      },
    });
    await flushPromises();

    // Toggle is hidden (no audio path).
    expect(w.find('[data-test="script-composer-enable-audio"]').exists()).toBe(
      false,
    );
    // And the persisted draft has been re-written with enable_audio gone
    // so the next submit doesn't ship it to the server.
    const persisted = JSON.parse(
      localStorage.getItem("mold.chain.draft.v2") ?? "{}",
    );
    expect(persisted.chain.enable_audio).toBeUndefined();
    expect(persisted.chain.motion_tail_frames).toBe(0);
  });
});
