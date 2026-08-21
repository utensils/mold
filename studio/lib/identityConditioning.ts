/**
 * Face-identity conditioning (PuLID) — the ONE browser-side policy for the
 * `id_image` / `id_image_name` / `id_weight` / `id_start_step` request
 * partition, mirroring `mold_core::identity`.
 *
 * Every Studio surface (web, desktop, and later iPhone) reads its labels,
 * defaults, ranges, capability gate, validation, request fields, and Library
 * provenance from here. Surfaces never restate a rule: the server refuses an
 * invalid identity partition outright, so a second local policy that drifted
 * would either block work the server accepts or queue work it refuses.
 *
 * Nothing here is a NEW rule. The admission contract is:
 *   - identity is accepted only on an identity-qualified checkpoint, which
 *     the server advertises per model (never inferred from a name here);
 *   - it may not combine with a LoRA or an img2img source image;
 *   - a weight/start-step/name without an image is invalid;
 *   - the image must be PNG or JPEG, at most 16 MiB encoded, 8192 px per
 *     axis, and 32 MP decoded — all checked from the header alone.
 */
import {
  persistGenerationSourceMedia,
  restoreGenerationSourceMedia,
  type GenerationSourceMedia,
} from "./generationSourceMedia";
import { imageDimensionsFromBase64 } from "./imageDimensions";
import type { GenerationRecipeProfile } from "./generationProfile";

/** Section and control wording. One spelling, every surface. */
export const IDENTITY_SECTION_LABEL = "Identity";
export const IDENTITY_PHOTO_LABEL = "Identity photo";
export const IDENTITY_WEIGHT_LABEL = "Identity strength";
export const IDENTITY_START_STEP_LABEL = "Identity start step";

export const IDENTITY_PHOTO_PLACEHOLDER = "Drop a face photo or click to pick";
export const IDENTITY_PHOTO_HINT =
  "One clear, front-facing face. The render keeps this person's likeness; the photo itself is never composited in.";
export const IDENTITY_WEIGHT_HINT =
  "How strongly the face is held. Higher preserves the likeness; lower lets the prompt reshape it.";
export const IDENTITY_START_STEP_HINT =
  "Delay identity until the composition has settled. 0 pins the face from the first step.";

/** `mold_core::identity::ID_WEIGHT_DEFAULT`. */
export const ID_WEIGHT_DEFAULT = 1.0;
export const ID_WEIGHT_MIN = 0.0;
/** `mold_core::identity::ID_WEIGHT_MAX` — the range is `0.0..=3.0`. */
export const ID_WEIGHT_MAX = 3.0;
/** Slider granularity; presentation only, the wire takes any finite value. */
export const ID_WEIGHT_STEP = 0.05;
/** `mold_core::identity::ID_START_STEP_DEFAULT`. */
export const ID_START_STEP_DEFAULT = 0;

/** The engine takes PNG or JPEG only, exactly like source conditioning. */
export const ID_IMAGE_ACCEPT = "image/png,image/jpeg";

/** `mold_core::identity::ID_IMAGE_LIMITS`, in the browser's own spelling. */
export const ID_IMAGE_LIMITS = {
  maxEncodedBytes: 16 * 1024 * 1024,
  maxAxisPixels: 8192,
  maxDecodedPixels: 32_000_000,
} as const;

/** The `/api/models` row's additive identity flag; absent on older servers. */
export interface IdentityModelEntry {
  supports_identity?: boolean | null;
}

/** Just enough of a resolved recipe to answer the capability question. */
export type IdentityRecipe = Pick<GenerationRecipeProfile, "capabilities"> & {
  legacy_adapter?: true;
};

/**
 * Whether the selected checkpoint accepts an identity photo on this host.
 *
 * The server-authored generation profile is the authority: it is `true` only
 * for a qualified checkpoint on a binary that actually links the adapter. The
 * `/api/models` row's additive `supports_identity` is the fallback for the
 * picker and for hosts whose profile has not been read; an absent field on an
 * older server reads as "no", which is what keeps the control hidden rather
 * than queueing work the host refuses.
 *
 * The client-side Release-N legacy adapter synthesizes `false` locally — that
 * is this client talking, not the server, so it never answers here.
 */
export function supportsIdentity(
  recipe: IdentityRecipe | null | undefined,
  model?: IdentityModelEntry | null,
): boolean {
  if (recipe && !recipe.legacy_adapter) {
    return recipe.capabilities.supports_identity === true;
  }
  return model?.supports_identity === true;
}

/** An attached identity photo: raw base64 plus its provenance label. */
export interface IdentityImage {
  /** Raw base64, no data-URI prefix. */
  base64: string;
  filename?: string | null;
}

export interface IdentityFormInput {
  /** `supportsIdentity` for the CURRENTLY selected checkpoint. */
  supported: boolean;
  image: IdentityImage | null;
  /** `null` = untouched; the server default applies. */
  weight: number | null;
  /** `null` = untouched; the server default applies. */
  startStep: number | null;
  /** Denoise steps this print will render. */
  steps: number;
  hasLora: boolean;
  hasSourceImage: boolean;
  /** Named in the not-qualified message; display only. */
  model?: string | null;
}

/** Decoded byte length of a base64 payload, without decoding it. */
function decodedBase64Bytes(base64: string): number {
  const payload = base64.replace(/\s+/g, "");
  if (!payload) return 0;
  const padding = payload.endsWith("==") ? 2 : payload.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor((payload.length * 3) / 4) - padding);
}

/**
 * The cheap pre-checks the server runs from the encoded header alone, in the
 * server's own order: encoded length, then format, then declared dimensions.
 * Nothing here decodes the image.
 */
export function identityImageError(base64: string): string | null {
  const bytes = decodedBase64Bytes(base64);
  if (bytes === 0) return `${IDENTITY_PHOTO_LABEL} must not be empty.`;
  if (bytes > ID_IMAGE_LIMITS.maxEncodedBytes) {
    return `${IDENTITY_PHOTO_LABEL} must be 16 MiB or smaller.`;
  }
  const dimensions = imageDimensionsFromBase64(base64);
  if (!dimensions) {
    return `${IDENTITY_PHOTO_LABEL} must be a PNG or JPEG image.`;
  }
  const { width, height } = dimensions;
  if (
    width > ID_IMAGE_LIMITS.maxAxisPixels ||
    height > ID_IMAGE_LIMITS.maxAxisPixels
  ) {
    return `${IDENTITY_PHOTO_LABEL} is ${width} × ${height}; each side must be at most ${ID_IMAGE_LIMITS.maxAxisPixels} pixels.`;
  }
  if (width * height > ID_IMAGE_LIMITS.maxDecodedPixels) {
    return `${IDENTITY_PHOTO_LABEL} is ${width} × ${height}; the limit is 32 megapixels.`;
  }
  return null;
}

/**
 * Why the identity partition of this form would be refused, in the server's
 * own order — or `null` when it is valid (including "not used at all").
 *
 * Surfaces render this INLINE beside the control, never as a toast: it is a
 * property of the composed print, not an event.
 */
export function identityValidationError(
  input: IdentityFormInput,
): string | null {
  const mentions =
    input.image != null || input.weight != null || input.startStep != null;
  if (!mentions) return null;

  if (!input.image) {
    return `Attach an ${IDENTITY_PHOTO_LABEL.toLowerCase()}, or clear ${IDENTITY_WEIGHT_LABEL} and ${IDENTITY_START_STEP_LABEL}.`;
  }
  if (!input.supported) {
    const named = input.model?.trim();
    return `${named || "This model"} does not support identity photos. Choose an identity-qualified model, or remove the photo.`;
  }
  if (input.hasLora) {
    return "An identity photo cannot be combined with a LoRA yet. Remove the LoRA or the photo.";
  }
  if (input.hasSourceImage) {
    return "An identity photo cannot be combined with a source image yet. Remove the source image or the photo.";
  }
  if (input.weight != null) {
    if (
      !Number.isFinite(input.weight) ||
      input.weight < ID_WEIGHT_MIN ||
      input.weight > ID_WEIGHT_MAX
    ) {
      return `${IDENTITY_WEIGHT_LABEL} must be from ${ID_WEIGHT_MIN} to ${ID_WEIGHT_MAX}.`;
    }
  }
  if (input.startStep != null) {
    const steps = Number.isFinite(input.steps) ? input.steps : 0;
    if (
      !Number.isInteger(input.startStep) ||
      input.startStep < 0 ||
      input.startStep >= steps
    ) {
      return `${IDENTITY_START_STEP_LABEL} must be a whole number from 0 to ${Math.max(0, steps - 1)} — fewer than the ${steps} steps this print renders.`;
    }
  }
  return identityImageError(input.image.base64);
}

/** The four additive request fields, exactly as they ride the wire. */
export interface IdentityRequestFields {
  id_image?: string;
  id_image_name?: string;
  id_weight?: number;
  id_start_step?: number;
}

export interface IdentityRequestInput {
  supported: boolean;
  image: IdentityImage | null;
  weight: number | null;
  startStep: number | null;
}

/**
 * Project the form's identity state onto the wire.
 *
 * An identity photo is NEVER fitted, cropped, or resized against the canvas —
 * it is a face reference, not a composition input, so the picked bytes travel
 * untouched. Untouched knobs stay absent so the server's own defaults remain
 * authoritative, and everything is dropped when the selected checkpoint is
 * not qualified (staged media survives a model switch on every surface; only
 * the wire is gated).
 */
export function identityRequestFields(
  input: IdentityRequestInput,
): IdentityRequestFields {
  if (!input.supported || !input.image?.base64) return {};
  const name = input.image.filename?.trim();
  return {
    id_image: input.image.base64,
    ...(name ? { id_image_name: name } : {}),
    ...(input.weight != null && Number.isFinite(input.weight)
      ? { id_weight: input.weight }
      : {}),
    ...(input.startStep != null && Number.isInteger(input.startStep)
      ? { id_start_step: input.startStep }
      : {}),
  };
}

/**
 * Contribution to the Advanced "N on" badge. Only the two knobs count: the
 * photo well lives in the primary form beside the source wells, exactly like
 * source images, which stopped counting when they moved there.
 */
export function identityActiveCount(input: {
  weight: number | null;
  startStep: number | null;
}): number {
  return (input.weight != null ? 1 : 0) + (input.startStep != null ? 1 : 0);
}

/** Saved identity provenance — names and digests only, never a face payload. */
export interface IdentityProvenance {
  name: string | null;
  /** Lowercase hex digest of the exact bytes that rendered, when recorded. */
  sha256: string | null;
  /** First 12 hex characters, for a compact Library caption. */
  shortSha: string | null;
  /** Effective weight (the server records what actually applied). */
  weight: number;
  /** Effective first identity-conditioned step. */
  startStep: number;
}

/** The `OutputMetadata` slice identity provenance is read from. */
export interface IdentityMetadata {
  id_image_name?: string | null;
  id_image_sha256?: string | null;
  id_weight?: number | null;
  id_start_step?: number | null;
}

function normalizedSha256(value: string | null | undefined): string | null {
  const trimmed = value?.trim().toLowerCase() ?? "";
  return /^[a-f\d]{64}$/.test(trimmed) ? trimmed : null;
}

/**
 * Identity provenance for the Library aside, or `null` for a print that
 * carried no identity photo. A recorded name alone still counts: the digest
 * is additive and an older host may omit it.
 */
export function identityProvenance(
  metadata: IdentityMetadata | null | undefined,
): IdentityProvenance | null {
  if (!metadata) return null;
  const name = metadata.id_image_name?.trim() || null;
  const sha256 = normalizedSha256(metadata.id_image_sha256);
  const weight =
    typeof metadata.id_weight === "number" &&
    Number.isFinite(metadata.id_weight)
      ? metadata.id_weight
      : null;
  const startStep = Number.isInteger(metadata.id_start_step)
    ? (metadata.id_start_step as number)
    : null;
  const recorded =
    name != null ||
    sha256 != null ||
    weight != null ||
    startStep != null ||
    // A malformed digest is still evidence the print carried a photo.
    Boolean(metadata.id_image_sha256);
  if (!recorded) return null;
  return {
    name,
    sha256,
    shortSha: sha256 ? sha256.slice(0, 12) : null,
    weight: weight ?? ID_WEIGHT_DEFAULT,
    startStep: startStep ?? ID_START_STEP_DEFAULT,
  };
}

/** What Reuse settings restores from a print's identity provenance. */
export interface IdentityReuse {
  name: string | null;
  /** Lookup key for the local identity-photo stash; null = nothing to find. */
  sha256: string | null;
  weight: number;
  startStep: number;
}

/**
 * Reuse settings: the recorded knobs restore exactly, and `sha256` is the key
 * the surface looks the photo itself back up with (the same mechanism
 * `source_image_sha256` uses). Metadata never carries the face bytes, so a
 * miss is disclosed inline rather than silently rendering someone else.
 */
export function identityReuse(
  metadata: IdentityMetadata | null | undefined,
): IdentityReuse | null {
  const provenance = identityProvenance(metadata);
  if (!provenance) return null;
  return {
    name: provenance.name,
    sha256: provenance.sha256,
    weight: provenance.weight,
    startStep: provenance.startStep,
  };
}

/** One-line summary for a compact Library row. */
export function identityProvenanceSummary(
  provenance: IdentityProvenance,
): string {
  const parts = [provenance.name ?? "Identity photo"];
  if (provenance.shortSha) parts.push(provenance.shortSha);
  parts.push(`strength ${provenance.weight}`);
  parts.push(`from step ${provenance.startStep}`);
  return parts.join(" · ");
}

/**
 * Keep the identity photo where Reuse settings can find it again.
 *
 * `OutputMetadata` records only `id_image_sha256` — never the face bytes — so
 * restoring the photo works exactly like restoring a source image: the browser
 * saves it in the same content-addressed local media stash under the digest of
 * the bytes that shipped, and a later reuse looks it up by that digest. The
 * record's `sourceFit` stays absent because an identity photo is never fitted.
 *
 * Best-effort by construction: a failed write returns `null` and the print
 * still renders. It is the digest, not the payload, that travels.
 */
export function persistIdentityPhoto(
  base64: string,
  provenance: {
    filename: string;
    width?: number | null;
    height?: number | null;
    mime?: string | null;
  },
): Promise<string | null> {
  return persistGenerationSourceMedia(base64, {
    base64,
    kind: "upload",
    ...provenance,
  });
}

/** Look a previously-shipped identity photo back up by its recorded digest. */
export function restoreIdentityPhoto(
  sha256: string | null | undefined,
): Promise<GenerationSourceMedia | null> {
  return restoreGenerationSourceMedia(sha256);
}

/** Disclosure when the photo behind a reused print is no longer on this device. */
export const IDENTITY_PHOTO_UNAVAILABLE =
  "The identity photo for this print is not on this device — reattach it to render the same likeness.";
