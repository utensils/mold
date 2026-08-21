/**
 * iPhone-shaped helpers around the shared face-identity (PuLID) policy.
 *
 * Every RULE lives in `@studio/lib/identityConditioning` — ranges, wording,
 * the capability gate, the header pre-checks, provenance, and reuse. What is
 * genuinely phone-shaped lives here: the 45 MiB request-media budget the
 * WebView bridge imposes, the acquire path a native photo pick takes, and the
 * Info-sheet rows the Library viewer renders. Keeping them in this module is
 * what lets `MobileApp.vue` stay an orchestrator.
 */
import {
  IDENTITY_PHOTO_LABEL,
  IDENTITY_PHOTO_UNAVAILABLE,
  IDENTITY_WEIGHT_LABEL,
  identityActiveCount,
  identityImageError,
  identityProvenance,
  type IdentityMetadata,
} from "@studio/lib/identityConditioning";
import type { GenerateForm, PickedImage } from "../lib/generateForm";
import {
  MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES,
  MOBILE_MEDIA_BUDGET_ERROR,
  decodedBase64Bytes,
  inlineGenerationMediaBytes,
} from "../lib/generateValidation";
import { isStillImageFile } from "../lib/image";

/** Label used when a pick (or a reused print) carries no filename of its own. */
export const MOBILE_IDENTITY_FALLBACK_FILENAME = "identity photo";

/**
 * Whether the identity well renders at all.
 *
 * Positive capability only, exactly as the desktop inspector reads it: an
 * unread or absent `supports_identity` is not evidence of support, and a
 * checkpoint that lost the capability PARKS the staged photo — the well
 * disappears, `buildRequest` keeps the partition off the wire, and Develop
 * stays enabled. Selecting a qualified model again brings both back.
 * Sequence clips have no identity slot on any surface.
 */
export function showMobileIdentityWell(form: GenerateForm, isSequence: boolean): boolean {
  return !isSequence && form.identitySupported === true;
}

/**
 * The identity contribution to the Advanced sheet's "N on" badge: the two
 * knobs only. The photo well itself is primary-form media beside the source
 * wells, which stopped counting when they moved there.
 */
export function mobileIdentityAdvancedCount(form: GenerateForm): number {
  if (form.identitySupported !== true) return 0;
  return identityActiveCount({
    weight: form.identityWeight,
    startStep: form.identityStartStep,
  });
}

/** Highest first identity-conditioned step this print can carry. */
export function mobileIdentityStartStepMax(steps: number): number {
  return Number.isFinite(steps) ? Math.max(0, Math.floor(steps) - 1) : 0;
}

/**
 * Request-media budget left for an identity photo.
 *
 * The photo rides the same JSON body as every other inline input even though
 * it is never fitted, so it spends the same 45 MiB. A photo already staged is
 * excluded: replacing one must not be charged twice.
 */
export function mobileIdentityBudgetBytes(form: GenerateForm): number {
  return Math.max(
    0,
    MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES - inlineGenerationMediaBytes(form, "identityImage"),
  );
}

/**
 * Why a picked file cannot be an identity photo, judged before it is read.
 *
 * The engine takes PNG and JPEG only. A native pick sometimes arrives with no
 * MIME type at all, so the filename is the fallback — the same reading the
 * source wells take.
 */
export function mobileIdentityFileRefusal(file: { type: string; name: string }): string | null {
  if (file.type === "image/png" || file.type === "image/jpeg") return null;
  if (!file.type && isStillImageFile(file.name)) return null;
  return `${IDENTITY_PHOTO_LABEL} must be a PNG or JPEG image.`;
}

export type MobileIdentityIngest = { ok: true; image: PickedImage } | { ok: false; error: string };

/**
 * Stage a picked photo, or say why it cannot be staged.
 *
 * The bytes travel VERBATIM: an identity photo is a face reference, not a
 * composition input, so it is never fitted, cropped, or resized against the
 * canvas the way a source image is. The checks are the phone's budget first
 * (it bounds what may enter the WebView at all) and then the server's own
 * header-only pre-checks, so a photo that could not be admitted never becomes
 * part of the draft.
 */
export function ingestMobileIdentityPhoto(
  picked: { filename: string; base64: string },
  budgetBytes: number,
): MobileIdentityIngest {
  if (decodedBase64Bytes(picked.base64) > budgetBytes) {
    return { ok: false, error: MOBILE_MEDIA_BUDGET_ERROR };
  }
  const refused = identityImageError(picked.base64);
  if (refused) return { ok: false, error: refused };
  return {
    ok: true,
    image: {
      filename: picked.filename.trim() || MOBILE_IDENTITY_FALLBACK_FILENAME,
      base64: picked.base64,
    },
  };
}

/**
 * Preview type derived from the provenance label — a JPEG previewed as
 * `data:image/png` relies on sniffing that is not guaranteed in a WebView.
 */
export function mobileIdentityMimeType(filename: string | null | undefined): string {
  const name = filename?.trim().toLowerCase() ?? "";
  return name.endsWith(".jpg") || name.endsWith(".jpeg") ? "image/jpeg" : "image/png";
}

/** Provenance with no bytes: the print's photo is not on this device. */
export function mobileIdentityNeedsReattach(image: PickedImage | null | undefined): boolean {
  return Boolean(image) && !image?.base64;
}

export interface MobileIdentityRow {
  label: string;
  value: string;
  /** Full digest, for the row's `title` — the caption shows 12 characters. */
  title?: string | undefined;
}

/**
 * Identity provenance rows for the Library viewer's Info sheet, or `null` for
 * a print that carried no identity photo. Saved metadata records names and
 * digests only — never the face bytes — which is exactly why the digest is
 * worth showing.
 */
export function mobileIdentityProvenanceRows(
  metadata: IdentityMetadata | null | undefined,
): MobileIdentityRow[] | null {
  const provenance = identityProvenance(metadata);
  if (!provenance) return null;
  const name = provenance.name ?? IDENTITY_PHOTO_LABEL;
  return [
    {
      label: IDENTITY_PHOTO_LABEL,
      value: provenance.shortSha ? `${name} · ${provenance.shortSha}` : name,
      title: provenance.sha256 ?? undefined,
    },
    {
      label: IDENTITY_WEIGHT_LABEL,
      value: `${provenance.weight} · from step ${provenance.startStep}`,
    },
  ];
}

export type MobileIdentityRestore =
  { kind: "skip" } | { kind: "attached"; image: PickedImage } | { kind: "missing"; note: string };

/**
 * What Use as prompt should do with the bytes-less identity descriptor
 * `applyMetadataToForm` leaves behind.
 *
 * The photo lives only in the local content-addressed stash, keyed by the
 * digest of exactly what shipped — the same mechanism a source image uses. A
 * miss is DISCLOSED (the phone's persistent inline status line, never a
 * toast): rendering a different face would be worse than saying the original
 * is gone.
 */
export function resolveMobileIdentityRestore(
  descriptor: PickedImage | null | undefined,
  restored: { base64: string; filename?: string | null } | null,
): MobileIdentityRestore {
  if (!mobileIdentityNeedsReattach(descriptor)) return { kind: "skip" };
  if (!restored?.base64) return { kind: "missing", note: IDENTITY_PHOTO_UNAVAILABLE };
  return {
    kind: "attached",
    image: {
      filename:
        restored.filename?.trim() ||
        descriptor?.filename?.trim() ||
        MOBILE_IDENTITY_FALLBACK_FILENAME,
      base64: restored.base64,
    },
  };
}
