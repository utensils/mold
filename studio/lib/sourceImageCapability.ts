/**
 * Per-model source-image conditioning contract (#772) and the wan
 * first/last-frame layout that rides on it (#779).
 *
 * Wan is the first family whose checkpoints split three ways — T2V-only,
 * I2V-required, I2V-optional — which no family-level fact can express. The
 * server derives the split from checkpoint tensor shapes and advertises it as
 * the additive `/api/models[].source_image` field; this module is the one
 * place every surface turns that field into a decision.
 *
 * Two rules are load-bearing:
 *
 *   - The field is additive, so it is absent on older servers AND on entries
 *     the current server could not classify. Absence must mean "unknown" and
 *     fall back to the caller's family heuristic — an older server enforces
 *     nothing at admission, so today's behaviour is the compatible answer.
 *   - Requiredness is never inferred. Only an explicit `required` gates
 *     submit; a family heuristic can say a checkpoint *reads* a source image,
 *     never that it cannot render without one.
 */

export type SourceImageCapability = "unsupported" | "optional" | "required";

const CAPABILITIES: readonly SourceImageCapability[] = [
  "unsupported",
  "optional",
  "required",
];

/**
 * Read the advertised contract, or `null` when the server said nothing this
 * client understands. An unrecognized value is treated exactly like an absent
 * one: a future mode must not be guessed into one of today's three.
 */
export function parseSourceImageCapability(
  value: unknown,
): SourceImageCapability | null {
  return typeof value === "string" &&
    (CAPABILITIES as readonly string[]).includes(value)
    ? (value as SourceImageCapability)
    : null;
}

/**
 * The effective contract: the advertised field first, the caller's family
 * heuristic when the server advertised nothing.
 *
 * `familyFallback` is supplied by `generationCapabilities`, which owns family
 * policy — keeping it a parameter is what stops this module from becoming a
 * second family table.
 */
export function resolveSourceImageCapability(
  advertised: unknown,
  familyFallback: SourceImageCapability,
): SourceImageCapability {
  return parseSourceImageCapability(advertised) ?? familyFallback;
}

/** Families whose engines read a first frame AND a last frame (#779). */
const FIRST_LAST_FRAME_FAMILIES: ReadonlySet<string> = new Set(["wan"]);

/**
 * Whether the optional End frame well should be offered.
 *
 * The advertised field is mandatory here, unlike the source well itself: a
 * server old enough to omit it also rejects wan keyframes outright (its
 * keyframe validator is LTX-2-only), so an optimistic fallback could only
 * produce a rejected request. Absence reads as "no", exactly like
 * `canOfferExtend`.
 */
export function supportsFirstLastFrames(
  family: string | null | undefined,
  advertised: unknown,
): boolean {
  if (!FIRST_LAST_FRAME_FAMILIES.has((family ?? "").trim().toLowerCase())) {
    return false;
  }
  const capability = parseSourceImageCapability(advertised);
  return capability === "required" || capability === "optional";
}

/** One entry of the `keyframes` wire array. */
export interface KeyframeConditionWire {
  frame: number;
  image: string;
  name?: string;
}

/** A still attached to one end of a first/last-frame render. */
export interface FirstLastFrameImage {
  base64: string;
  filename?: string | null;
}

/**
 * The two-entry `keyframes` layout for a first/last-frame render, or `null`
 * when the render is not one.
 *
 * The closing index is computed from the frame count held *at submit time* —
 * a user who changes the clip length after attaching the end frame must not
 * ship a stale index the server rejects. A lone first frame deliberately
 * produces nothing: that request is an ordinary `source_image` render and
 * carrying a single keyframe would change its meaning.
 */
export function firstLastFrameKeyframes(
  firstFrame: FirstLastFrameImage | null | undefined,
  endFrame: FirstLastFrameImage | null | undefined,
  frames: number | null | undefined,
): KeyframeConditionWire[] | null {
  if (!firstFrame?.base64 || !endFrame?.base64) return null;
  if (!Number.isInteger(frames) || (frames as number) < 2) return null;
  const closing = (frames as number) - 1;
  return [entry(0, firstFrame), entry(closing, endFrame)];
}

function entry(
  frame: number,
  image: FirstLastFrameImage,
): KeyframeConditionWire {
  const name = image.filename?.trim();
  return { frame, image: image.base64, ...(name ? { name } : {}) };
}

/** What gallery metadata records about a keyframe: provenance, never bytes. */
export interface KeyframeProvenance {
  name?: string | null;
}

/**
 * The reuse affordance for a first/last-frame print, or `null` when the print
 * is not one.
 *
 * Saved metadata carries each keyframe's name and digest and no payload, so
 * the closing still cannot be rebuilt from it — the same position source
 * videos are already in. Every other knob restores; this says the one thing
 * that did not, rather than letting Generate look ready.
 */
export function firstLastFrameRestoreNotice(
  supportsEndFrame: boolean,
  keyframes: readonly KeyframeProvenance[] | null | undefined,
): string | null {
  if (!supportsEndFrame || (keyframes?.length ?? 0) < 2) return null;
  const name = keyframes?.[keyframes.length - 1]?.name?.trim();
  return `${name ? `The end frame (${name})` : "The end frame"} can't be restored — saved metadata records its name and digest, not the image. Attach it again to reproduce this first/last-frame render.`;
}

export interface SourceImageValidationInput {
  capability: SourceImageCapability;
  /** Whether the source-image well holds an image. */
  hasSourceImage: boolean;
  /** Whether the End frame well holds an image. */
  hasEndFrame?: boolean;
  /** The clip length the request will carry. */
  frames?: number | null;
}

/**
 * Explain why the attached conditioning would be refused, or `null` when it
 * is valid. One message at a time, in the order the server checks them, so
 * the two never disagree about which problem to name first.
 */
export function sourceImageValidationError(
  input: SourceImageValidationInput,
): string | null {
  if (input.capability === "unsupported" && input.hasSourceImage) {
    return "This checkpoint is text-to-video only and does not accept a source image. Remove the image, or pick an image-to-video checkpoint.";
  }
  if (input.capability === "required" && !input.hasSourceImage) {
    return "This checkpoint is image-to-video only. Attach a source image to use as the first frame.";
  }
  if (!input.hasEndFrame) return null;
  if (!input.hasSourceImage) {
    return "An end frame needs a first frame. Attach a source image, or remove the end frame.";
  }
  return Number.isInteger(input.frames) && (input.frames as number) >= 2
    ? null
    : "A first/last-frame render needs at least two frames.";
}
