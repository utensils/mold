import type { BaseGenerationCapabilities } from "./generationCapabilities";

/**
 * What image-attachment UI the PRIMARY Create form owes the selected model.
 *
 * The model dictates this exactly the way it dictates resolutions: surfaces
 * derive their capabilities as they already do, project them through
 * `sourceMediaPlan`, and render the returned shape — never a second policy.
 * `none` means the model takes no image input and no well may render at all.
 */
export type SourceMediaPlan =
  /** The model takes no image conditioning — render nothing. */
  | { kind: "none" }
  /** One source image, with an optional closing keyframe well. */
  | {
      kind: "single";
      required: boolean;
      /** Renders the optional End frame well (wan first/last, #779). */
      endFrame: boolean;
      /** Video families condition their opening frame on the source. */
      video: boolean;
    }
  /** Ordered picture strip. Qwen edit exposes its first item through the same
   * Target well as other primary sources; FLUX.2 has references only. */
  | {
      kind: "attachments";
      max: number | null;
      required: boolean;
      primary: "target" | null;
    }
  /** MiniMax H3 FL2VA first/last boundaries — the same two wells as
   * `single`+`endFrame`, backed by dedicated H3 authoring state. */
  | { kind: "h3-boundaries"; requiredEndpoint: "first" | null }
  /** MiniMax H3 Ref2VA mixed-media references keep their ordered panel. */
  | { kind: "h3-references" };

/** FLUX.2 Dev's reference strip cap; Qwen edit is unbounded server-side. */
export const FLUX2_DEV_MAX_ATTACHMENTS = 4;

export function sourceMediaPlan(
  caps: Pick<
    BaseGenerationCapabilities,
    | "supportsSourceImage"
    | "requiresSourceImage"
    | "supportsEndFrame"
    | "sourceImageMode"
    | "supportsVideo"
  >,
): SourceMediaPlan {
  switch (caps.sourceImageMode) {
    case "h3-boundaries":
      return {
        kind: "h3-boundaries",
        requiredEndpoint: caps.requiresSourceImage ? "first" : null,
      };
    case "ordered-references":
      return { kind: "h3-references" };
    case "references":
    case "qwen-edit":
      if (!caps.supportsSourceImage) return { kind: "none" };
      return {
        kind: "attachments",
        max:
          caps.sourceImageMode === "references"
            ? FLUX2_DEV_MAX_ATTACHMENTS
            : null,
        required:
          caps.sourceImageMode === "qwen-edit" || caps.requiresSourceImage,
        primary: caps.sourceImageMode === "qwen-edit" ? "target" : null,
      };
    case "single":
      if (!caps.supportsSourceImage) return { kind: "none" };
      return {
        kind: "single",
        required: caps.requiresSourceImage,
        endFrame: caps.supportsEndFrame,
        video: caps.supportsVideo,
      };
  }
}
