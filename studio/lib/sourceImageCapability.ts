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
 * videos are already in. A first/last render also carries its OPENING frame
 * only as `keyframes[0]` (the request holds no `source_image`), so unless the
 * caller has an independent restore handle for it, both endpoints are gone
 * and the notice must say so — telling users to attach only the end frame
 * would leave a required-source checkpoint still blocked.
 */
export function firstLastFrameRestoreNotice(
  supportsEndFrame: boolean,
  keyframes: readonly KeyframeProvenance[] | null | undefined,
  openingRestorable = false,
): string | null {
  if (!supportsEndFrame || (keyframes?.length ?? 0) < 2) return null;
  const endName = keyframes?.[keyframes.length - 1]?.name?.trim();
  const end = endName ? `The end frame (${endName})` : "The end frame";
  if (openingRestorable) {
    return `${end} can't be restored — saved metadata records its name and digest, not the image. Attach it again to reproduce this first/last-frame render.`;
  }
  const firstName = keyframes?.[0]?.name?.trim();
  const first = firstName
    ? `the first frame (${firstName})`
    : "the first frame";
  return `${end} and ${first} can't be restored — saved metadata records names and digests, not the images. Attach both again to reproduce this first/last-frame render.`;
}

export interface SourceImageValidationInput {
  capability: SourceImageCapability;
  /** Whether the source-image well holds an image. */
  hasSourceImage: boolean;
  /**
   * Whether the request will go out as a continuation (#783).
   *
   * An extend carries its own first frames in the tail of the clip it
   * continues, which is exactly why the server counts it as carrying source:
   * `mold_core::validation::request_carries_source_frames` ORs `is_extend()`
   * in beside `source_image` and `keyframes`, and admission's
   * `enforce_source_image_capability` feeds that whole predicate to
   * `source_image_contract_violation`. Counting only the image left every Wan
   * I2V continuation refused for missing the very contract that makes the
   * checkpoint extend-capable.
   *
   * Absent reads as "not a continuation" — the pre-#783 meaning — so an
   * ordinary render is unaffected. Callers must pass the same predicate their
   * request builder applies, family gate included: a staged clip that the
   * builder drops carries nothing.
   */
  isExtend?: boolean;
  /** Whether the End frame well holds an image. */
  hasEndFrame?: boolean;
  /** The clip length the request will carry. */
  frames?: number | null;
  /**
   * The raw model id the request will carry. TI2V pins endpoints in latent
   * space, so its floor is stricter than the generic two-frame rule; the
   * server rejects the difference at admission, and this check mirrors it so
   * submit never offers a guaranteed 422. Opaque `cv:`/`hf:` ids get no
   * floor, exactly like the server's manifest-name resolution.
   */
  model?: string | null;
}

/**
 * TI2V's 2.2 VAE compresses time 4x, so a 5-frame pixel clip is two latent
 * frames — both pinned by a first/last render, nothing left to denoise. Nine
 * pixel frames (three latent frames) is the smallest 4k+1 clip with an
 * interior. Mirrors `mold-core`'s admission rule for `wan22-ti2v-5b`.
 */
export const WAN_TI2V_FLF_MIN_FRAMES = 9;

function isWanTi2vModel(model: string | null | undefined): boolean {
  return (model ?? "").trim().toLowerCase().startsWith("wan22-ti2v-5b");
}

/**
 * Explain why the attached conditioning would be refused, or `null` when it
 * is valid. One message at a time, in the order the server checks them, so
 * the two never disagree about which problem to name first.
 */
export function sourceImageValidationError(
  input: SourceImageValidationInput,
): string | null {
  // The client mirror of admission's `has_source` (#783): the clip a
  // continuation carries counts, so a Required checkpoint is satisfied by it
  // and an Unsupported one is refused for it. The two arms keep separate
  // wording because "remove the image" is nonsense advice for a continuation
  // that has none.
  if (input.capability === "unsupported") {
    if (input.hasSourceImage) {
      return "This checkpoint is text-to-video only and does not accept a source image. Remove the image, or pick an image-to-video checkpoint.";
    }
    if (input.isExtend) {
      return "This checkpoint is text-to-video only and cannot continue an existing clip — a continuation is seeded with the source clip's final frame. Pick an image-to-video checkpoint.";
    }
  }
  if (
    input.capability === "required" &&
    !input.hasSourceImage &&
    !input.isExtend
  ) {
    return "This checkpoint is image-to-video only. Attach a source image to use as the first frame.";
  }
  if (!input.hasEndFrame) return null;
  if (!input.hasSourceImage) {
    return "An end frame needs a first frame. Attach a source image, or remove the end frame.";
  }
  if (!Number.isInteger(input.frames) || (input.frames as number) < 2) {
    return "A first/last-frame render needs at least two frames.";
  }
  if (
    isWanTi2vModel(input.model) &&
    (input.frames as number) < WAN_TI2V_FLF_MIN_FRAMES
  ) {
    return `This checkpoint pins both endpoints in latent space, so a first/last-frame render needs at least ${WAN_TI2V_FLF_MIN_FRAMES} frames.`;
  }
  return null;
}
