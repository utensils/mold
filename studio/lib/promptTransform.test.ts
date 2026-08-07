import { describe, expect, it } from "vitest";
import {
  conditioningFingerprint,
  defaultRemixDimensions,
  promptSource,
  validateRemixVariants,
} from "./promptTransform";

describe("prompt remix contract", () => {
  it("prefers the durable original and falls back to a direct prompt", () => {
    expect(promptSource("expanded", "original")).toEqual({
      prompt: "original",
      rootPrompt: "original",
      kind: "original",
    });
    expect(promptSource("hand written", null)).toEqual({
      prompt: "hand written",
      kind: "direct",
    });
  });

  it.each([
    [
      "text-to-image",
      [
        "composition",
        "camera",
        "lighting",
        "setting",
        "mood",
        "movement",
        "style",
      ],
    ],
    [
      "text-to-video",
      [
        "composition",
        "camera",
        "lighting",
        "setting",
        "mood",
        "movement",
        "style",
      ],
    ],
    ["image-to-video", ["movement"]],
    ["video-to-video", ["movement"]],
    ["retake", ["movement"]],
    ["keyframe-interpolation", ["movement"]],
    ["audio-driven-video", ["movement"]],
    ["text-to-audio", ["mood", "movement"]],
  ] as const)(
    "matches the backend dimension policy for %s",
    (task, expected) => {
      expect(defaultRemixDimensions(task)).toEqual(expected);
    },
  );

  it("removes style whenever a style constraint is locked", () => {
    expect(defaultRemixDimensions("text-to-image", true)).not.toContain(
      "style",
    );
  });

  it("requires exactly three non-empty variants", () => {
    const variants = ["one", "two", "three"].map((prompt) => ({
      prompt,
      dimensions: ["camera" as const],
    }));
    expect(validateRemixVariants(variants)).toHaveLength(3);
    expect(() => validateRemixVariants(variants.slice(0, 2))).toThrow(
      "exactly 3",
    );
  });

  it("changes the client-only fingerprint when conditioned media changes", () => {
    expect(conditioningFingerprint({ source_image: "a" })).not.toBe(
      conditioningFingerprint({ source_image: "b" }),
    );
  });

  it("treats ordered H3 reference reordering as stale conditioning", () => {
    const first = { kind: "image", provenance: { sha256: "a" } };
    const second = { kind: "audio", provenance: { sha256: "b" } };
    expect(conditioningFingerprint({ references: [first, second] })).not.toBe(
      conditioningFingerprint({ references: [second, first] }),
    );
  });

  it("fingerprints H3 reference digests without synchronously hashing media bytes", () => {
    const reference = (data: string, sha256: string) => ({
      kind: "video",
      media: { authority: "inline", data },
      provenance: { name: "motion.mp4", sha256 },
      duration_ms: 4_000,
    });
    expect(
      conditioningFingerprint({ references: [reference("A", "same")] }),
    ).toBe(conditioningFingerprint({ references: [reference("B", "same")] }));
    expect(
      conditioningFingerprint({ references: [reference("A", "first")] }),
    ).not.toBe(
      conditioningFingerprint({ references: [reference("A", "second")] }),
    );
  });
});
