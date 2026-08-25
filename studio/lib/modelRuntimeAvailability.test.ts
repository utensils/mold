import { describe, expect, it } from "vitest";
import {
  isModelRuntimeUnavailable,
  modelRuntimeNotice,
  modelRuntimeNoticeAcrossHosts,
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
      "MiniMax H3 has no runtime for this model's task partition in this build.",
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
    // 21-42 GB pull rather than at submit time. Both compact task partitions
    // execute since #825, so the download-only row here is a pinned layout
    // the build has no loader for.
    const rows = [
      { name: "flux-dev:q8" },
      {
        name: "minimax-h3-ref2va:comfy-pruned-nvfp4",
        runtime_available: false,
        runtime_unavailable_reason:
          "MiniMax H3 has no runtime for this model's weight layout in this build.",
      },
      {
        name: "minimax-h3-ref2va:comfy-pruned-int8",
        runtime_available: true,
      },
      {
        name: "minimax-h3-fl2va:comfy-pruned-int8",
        runtime_available: true,
      },
    ];
    expect(
      modelRuntimeNoticeForId("minimax-h3-ref2va:comfy-pruned-nvfp4", rows),
    ).toEqual({
      message:
        "MiniMax H3 has no runtime for this model's weight layout in this build.",
      fromServer: true,
    });
    expect(
      modelRuntimeNoticeForId("minimax-h3-ref2va:comfy-pruned-int8", rows),
    ).toBeNull();
    expect(
      modelRuntimeNoticeForId("minimax-h3-fl2va:comfy-pruned-int8", rows),
    ).toBeNull();
    expect(modelRuntimeNoticeForId("flux-dev:q8", rows)).toBeNull();
  });

  describe("across a fleet", () => {
    // Pull can target any connected machine, so the local answer alone is
    // materially wrong wording on a mixed fleet.
    const cannot = [
      {
        name: "minimax-h3-fl2va:comfy-pruned-int8",
        runtime_available: false,
        runtime_unavailable_reason: "This build has no H3 engine.",
      },
    ];
    const can = [
      {
        name: "minimax-h3-fl2va:comfy-pruned-int8",
        runtime_available: true,
      },
    ];

    it("stays silent when any listing machine can run the model", () => {
      expect(
        modelRuntimeNoticeAcrossHosts("minimax-h3-fl2va:comfy-pruned-int8", [
          cannot,
          can,
        ]),
      ).toBeNull();
      expect(
        modelRuntimeNoticeAcrossHosts("minimax-h3-fl2va:comfy-pruned-int8", [
          can,
          cannot,
        ]),
      ).toBeNull();
    });

    it("warns only when every listing machine refuses, naming the obstacle", () => {
      expect(
        modelRuntimeNoticeAcrossHosts("minimax-h3-fl2va:comfy-pruned-int8", [
          cannot,
          cannot,
        ]),
      ).toEqual({ message: "This build has no H3 engine.", fromServer: true });
    });

    it("treats an unread inventory as no evidence at all", () => {
      expect(
        modelRuntimeNoticeAcrossHosts("minimax-h3-fl2va:comfy-pruned-int8", [
          null,
          undefined,
          [],
        ]),
      ).toBeNull();
      // An unread machine beside a refusing one keeps the refusal: only a
      // positive "I can run this" withdraws the warning.
      expect(
        modelRuntimeNoticeAcrossHosts("minimax-h3-fl2va:comfy-pruned-int8", [
          cannot,
          null,
        ]),
      ).not.toBeNull();
    });
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
