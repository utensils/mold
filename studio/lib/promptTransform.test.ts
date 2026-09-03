import { describe, expect, it } from "vitest";
import {
  conditioningFingerprint,
  defaultRemixDimensions,
  PROMPT_IGNORED_TRANSFORM_REASON,
  promptSource,
  promptTransformBlockedReason,
  transformCountAccepted,
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

  it("accepts the single advisory answer when the recipe ignores the prompt", () => {
    // A prompt-ignoring family (no text encoder) is answered with ONE result,
    // the guide's image-preparation advice, whatever count was requested.
    const advice = [
      { prompt: "Use a clean cutout on a plain background.", dimensions: [] },
    ];
    expect(
      validateRemixVariants(advice, 3, { promptIgnored: true }),
    ).toHaveLength(1);
    expect(() => validateRemixVariants(advice, 3)).toThrow("exactly 3");
    // Any other short answer is still a malformed batch.
    const two = advice.concat({ prompt: "Crop tightly.", dimensions: [] });
    expect(() =>
      validateRemixVariants(two, 3, { promptIgnored: true }),
    ).toThrow("exactly 3");
    expect(transformCountAccepted(3, 3)).toBe(true);
    expect(transformCountAccepted(1, 3)).toBe(false);
    expect(transformCountAccepted(1, 3, { promptIgnored: true })).toBe(true);
    expect(transformCountAccepted(2, 3, { promptIgnored: true })).toBe(false);
  });

  it("blocks Expand and Remix only for a recipe that ignores the prompt", () => {
    expect(promptTransformBlockedReason("ignored")).toBe(
      PROMPT_IGNORED_TRANSFORM_REASON,
    );
    expect(promptTransformBlockedReason("optional")).toBeNull();
    expect(promptTransformBlockedReason("required")).toBeNull();
    expect(promptTransformBlockedReason(null)).toBeNull();
    expect(promptTransformBlockedReason(undefined)).toBeNull();
  });

  it("changes the client-only fingerprint when conditioned media changes", () => {
    expect(conditioningFingerprint({ source_image: "a" })).not.toBe(
      conditioningFingerprint({ source_image: "b" }),
    );
  });

  it("treats an identity photo as conditioning media", () => {
    // A face reference conditions the render exactly like a source image
    // does, so swapping or removing one has to stale reviewed work through
    // the SAME rule rather than a second, identity-only staleness check.
    expect(conditioningFingerprint({ id_image: "a" })).not.toBe(
      conditioningFingerprint({ id_image: "b" }),
    );
    expect(conditioningFingerprint({ id_image: "a" })).not.toBe(
      conditioningFingerprint({}),
    );
  });

  it("treats an edit/reference swap as stale conditioning", () => {
    // Reference images ARE the conditioning for an edit recipe (Qwen edit,
    // FLUX.2 [dev]) and one half of it for an exclusive one (Klein), so
    // swapping, reordering, adding or removing one has to stale reviewed
    // prompt work exactly as a source-image swap does.
    expect(conditioningFingerprint({ edit_images: ["a"] })).not.toBe(
      conditioningFingerprint({ edit_images: ["b"] }),
    );
    expect(conditioningFingerprint({ edit_images: ["a", "b"] })).not.toBe(
      conditioningFingerprint({ edit_images: ["b", "a"] }),
    );
    expect(conditioningFingerprint({ edit_images: ["a"] })).not.toBe(
      conditioningFingerprint({ edit_images: ["a", "b"] }),
    );
    expect(conditioningFingerprint({ edit_images: ["a"] })).not.toBe(
      conditioningFingerprint({}),
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
