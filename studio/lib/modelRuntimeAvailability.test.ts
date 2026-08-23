import { describe, expect, it } from "vitest";
import {
  isModelRuntimeUnavailable,
  modelRuntimeNotice,
  modelRuntimeNoticeForId,
  RUNTIME_UNAVAILABLE_FALLBACK,
} from "./modelRuntimeAvailability";

describe("model runtime availability", () => {
  it("treats an absent field as runnable, which is what an older server sends", () => {
    expect(isModelRuntimeUnavailable(undefined)).toBe(false);
    expect(isModelRuntimeUnavailable(null)).toBe(false);
    expect(isModelRuntimeUnavailable({})).toBe(false);
    expect(isModelRuntimeUnavailable({ runtime_available: null })).toBe(false);
    expect(isModelRuntimeUnavailable({ runtime_available: true })).toBe(false);
    expect(isModelRuntimeUnavailable({ runtime_available: false })).toBe(true);
  });

  it("has no note for a runnable row", () => {
    expect(modelRuntimeNotice({ runtime_available: true })).toBeNull();
    expect(
      modelRuntimeNotice({
        runtime_available: true,
        // A stale reason beside a runnable row is never rendered.
        runtime_unavailable_reason: "stale",
      }),
    ).toBeNull();
  });

  it("renders the server's own sentence, whichever of the three it is", () => {
    for (const reason of [
      "MiniMax H3 has no runtime for this model's weight layout in this build.",
      "MiniMax H3 reference-to-audio-video (Ref2VA) execution is not available in any released build.",
      "This mold build was compiled without the MiniMax H3 engine.",
    ]) {
      expect(
        modelRuntimeNotice({
          runtime_available: false,
          runtime_unavailable_reason: reason,
        }),
      ).toEqual({ message: reason, fromServer: true });
    }
  });

  it("falls back to a cause-free sentence for a server that predates the reason", () => {
    expect(modelRuntimeNotice({ runtime_available: false })).toEqual({
      message: RUNTIME_UNAVAILABLE_FALLBACK,
      fromServer: false,
    });
    for (const reason of [null, "", "   "]) {
      expect(
        modelRuntimeNotice({
          runtime_available: false,
          runtime_unavailable_reason: reason,
        }),
      ).toEqual({ message: RUNTIME_UNAVAILABLE_FALLBACK, fromServer: false });
    }
  });

  it("answers for a Discover row before anything is downloaded", () => {
    // The whole point of #1276: the manifest row is already in /api/models
    // with `downloaded: false`, so its runtime answer is knowable before the
    // 21-42 GB pull rather than at submit time.
    const rows = [
      { name: "flux-dev:q8" },
      {
        name: "minimax-h3-ref2va:comfy-pruned-int8",
        runtime_available: false,
        runtime_unavailable_reason: "Ref2VA execution is not available.",
      },
      {
        name: "minimax-h3-fl2va:comfy-pruned-int8",
        runtime_available: true,
      },
    ];
    expect(
      modelRuntimeNoticeForId("minimax-h3-ref2va:comfy-pruned-int8", rows),
    ).toEqual({
      message: "Ref2VA execution is not available.",
      fromServer: true,
    });
    expect(
      modelRuntimeNoticeForId("minimax-h3-fl2va:comfy-pruned-int8", rows),
    ).toBeNull();
    expect(modelRuntimeNoticeForId("flux-dev:q8", rows)).toBeNull();
  });

  it("never guesses for an id or a listing it does not have", () => {
    expect(
      modelRuntimeNoticeForId("cv:12345", [{ name: "flux-dev:q8" }]),
    ).toBeNull();
    expect(modelRuntimeNoticeForId("flux-dev:q8", null)).toBeNull();
    expect(modelRuntimeNoticeForId("", [{ name: "" }])).toBeNull();
    expect(modelRuntimeNoticeForId(null, [])).toBeNull();
  });
});
