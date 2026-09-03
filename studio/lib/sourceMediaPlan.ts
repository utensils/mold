import type {
  BaseGenerationCapabilities,
  SourceImageMode,
} from "./generationCapabilities";

/**
 * What image-attachment UI the PRIMARY Create form owes the selected model.
 *
 * The model dictates this exactly the way it dictates resolutions: surfaces
 * derive their capabilities as they already do, project them through
 * `sourceMediaPlan`, and render the returned shape — never a second policy.
 * `none` means the model takes no image input and no well may render at all.
 *
 * Every reference number here (the strip ceiling, requiredness, whether image
 * 0 is the edit target, the reference pixel budgets) comes from the advertised
 * `capabilities.reference_images` block through `caps.referenceImages`. No
 * surface and no plan carries a model-name constant.
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
   * Target well as other primary sources; FLUX.2 [dev] has references only. */
  | {
      kind: "attachments";
      max: number | null;
      required: boolean;
      primary: "target" | null;
    }
  /**
   * BOTH wells, mutually exclusive (FLUX.2 [klein]): the checkpoint renders
   * from a source image (with strength and a repaint mask) OR from ordered
   * references, never both in one pass. `resolveExclusiveWells` decides which
   * one is active; the other parks with an inline note and keeps its media.
   */
  | {
      kind: "single-or-references";
      single: { required: boolean; endFrame: boolean; video: boolean };
      references: {
        max: number | null;
        maxPixelsSingle: number | null;
        maxPixelsMulti: number | null;
      };
    }
  /** MiniMax H3 FL2VA first/last boundaries — the same two wells as
   * `single`+`endFrame`, backed by dedicated H3 authoring state. */
  | { kind: "h3-boundaries"; requiredEndpoint: "first" | null }
  /** MiniMax H3 Ref2VA mixed-media references keep their ordered panel. */
  | { kind: "h3-references" };

export function sourceMediaPlan(
  caps: Pick<
    BaseGenerationCapabilities,
    | "supportsSourceImage"
    | "requiresSourceImage"
    | "supportsEndFrame"
    | "sourceImageMode"
    | "supportsVideo"
    | "referenceImages"
  >,
): SourceMediaPlan {
  const references = caps.referenceImages;
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
        max: references?.max ?? null,
        required: (references?.required ?? false) || caps.requiresSourceImage,
        primary: references?.primaryIsTarget ? "target" : null,
      };
    case "single-or-references":
      return {
        kind: "single-or-references",
        single: {
          required: caps.requiresSourceImage,
          endFrame: caps.supportsEndFrame,
          video: caps.supportsVideo,
        },
        references: {
          max: references?.max ?? null,
          maxPixelsSingle: references?.maxPixelsSingle ?? null,
          maxPixelsMulti: references?.maxPixelsMulti ?? null,
        },
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

/** The two wells of a `single-or-references` plan. */
export type ExclusiveWell = "source" | "references";

export interface ExclusiveWellsState {
  hasSource: boolean;
  referenceCount: number;
  /**
   * Which well the user wrote most recently. Surfaces persist it beside the
   * form; a snapshot restored from before the field existed carries `null`,
   * which reads as the source well (today's behaviour for a restored print
   * that has both).
   */
  lastWrite?: ExclusiveWell | null;
}

export interface ExclusiveWells {
  /** The well whose media ships; `null` while neither holds any. */
  active: ExclusiveWell | null;
  /** The well that is parked, or `null` while nothing is active. */
  parked: ExclusiveWell | null;
  /** The inline note the parked well renders; `null` when nothing parks. */
  note: string | null;
}

/**
 * The exclusive-wells parking rule: LAST WRITE WINS, and the parked media is
 * kept rather than discarded.
 *
 * Attaching to either well parks the other — it does not refuse the drop and
 * it does not clear the earlier image, so removing the active media restores
 * the parked well exactly as it was. Generate stays ENABLED throughout: only
 * the active well's media reaches the wire, which is what makes parking safe
 * (the identity-photo precedent — staged media parks, it never blocks).
 */
export const EXCLUSIVE_WELLS_NOTE =
  "This model renders from a source image OR reference images, not both — remove one to use the other.";

export function resolveExclusiveWells(
  state: ExclusiveWellsState,
): ExclusiveWells {
  const hasReferences = state.referenceCount > 0;
  if (!state.hasSource && !hasReferences) {
    return { active: null, parked: null, note: null };
  }
  const active: ExclusiveWell =
    state.hasSource && hasReferences
      ? // Both hold media: the last write decides, and an unmarked restore
        // reads as the source well.
        state.lastWrite === "references"
        ? "references"
        : "source"
      : state.hasSource
        ? "source"
        : "references";
  return {
    active,
    parked: active === "source" ? "references" : "source",
    note: EXCLUSIVE_WELLS_NOTE,
  };
}

/**
 * WHICH conditioning a request built in this mode carries — the one decision
 * behind every request builder, every request pruner, and the strength/mask
 * controls that only apply to a source image.
 *
 * It exists because `single-or-references` broke the old shorthand
 * (`mode === "single" ? source : edit_images`): Klein's request is one or the
 * other depending on what the user attached, and a builder that emitted both
 * is refused at admission. H3 modes answer `none` — their boundaries and
 * ordered references have their own serializer.
 */
export function conditioningForRequest(
  mode: SourceImageMode,
  state: ExclusiveWellsState,
): ExclusiveWell | "none" {
  switch (mode) {
    case "single":
      return state.hasSource ? "source" : "none";
    case "references":
    case "qwen-edit":
      return state.referenceCount > 0 ? "references" : "none";
    case "single-or-references":
      return resolveExclusiveWells(state).active ?? "none";
    case "h3-boundaries":
    case "ordered-references":
      return "none";
  }
}
