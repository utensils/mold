/**
 * The advertised reference-image contract: whether a checkpoint takes ordered
 * reference images (`GenerateRequest.edit_images`), how many, whether the
 * first one is the edit TARGET, and how references relate to `source_image`.
 *
 * `capabilities.reference_images` on the generation profile is the single
 * authority — `mold_core::generation_profile::reference_images_for_recipe`
 * answers it once for the server, admission, the CLI and every GUI.
 * Absence of the block means an OLDER SERVER, never a refusal (the
 * `supports_strength` lesson): a client falls back to the pre-profile family
 * sniff in `legacyRecipeRules.ts`.
 *
 * The cross-surface expectations are pinned in
 * `tests/fixtures/flux2/reference-parity-v1.json`, read by both a mold-core
 * test and `flux2ReferenceParity.test.ts`.
 *
 * The two WIRE types are the GENERATED ones (`ts-rs` from
 * `crates/mold-core/src/generation_profile.rs`); this module re-exports them
 * so every surface has one import site for the block and its projection.
 */

export type {
  ImageInputFormat,
  ReferenceCanvasRule,
  ReferenceImagesProfile,
  ReferenceSourceRelation,
} from "./generated/generationProfileV1";

import type {
  FloatControl,
  ImageInputFormat,
  ReferenceCanvasRule,
  ReferenceImagesProfile,
  ReferenceSourceRelation,
} from "./generated/generationProfileV1";

/**
 * The containers an EMPTY `formats` list means: every recipe that predates
 * the field, and every older server (`ImageInputFormat::LEGACY`).
 */
export const LEGACY_REFERENCE_IMAGE_FORMATS: readonly ImageInputFormat[] = [
  "png",
  "jpeg",
];

const IMAGE_INPUT_MIME: Record<ImageInputFormat, string> = {
  png: "image/png",
  jpeg: "image/jpeg",
  webp: "image/webp",
};

/** The MIME types a picker for these reference containers accepts. */
export function referenceImageMimeTypes(
  formats: readonly ImageInputFormat[],
): string[] {
  return formats.map((format) => IMAGE_INPUT_MIME[format]);
}

/**
 * Sniff a still's container from its leading bytes — the browser mirror of
 * `mold_core::validation::sniff_image_input_format`. `null` for anything
 * that is not PNG, JPEG or WebP.
 */
export function sniffImageInputFormat(
  bytes: Uint8Array,
): ImageInputFormat | null {
  if (
    bytes.length >= 4 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47
  ) {
    return "png";
  }
  if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    return "jpeg";
  }
  const ascii = (from: number, to: number) =>
    String.fromCharCode(...bytes.subarray(from, to));
  if (bytes.length >= 12 && ascii(0, 4) === "RIFF" && ascii(8, 12) === "WEBP") {
    return "webp";
  }
  return null;
}

/** The client-side projection every surface reads. `null` where the recipe
 * (or the legacy rule standing in for an older host) offers no references. */
export interface ReferenceImagesCapabilities {
  /** Generate stays gated until at least one reference is attached. */
  required: boolean;
  /** Strip ceiling; `null` is unbounded (Qwen edit). */
  max: number | null;
  /** Index 0 is the edit target, rendered through the shared Target well. */
  primaryIsTarget: boolean;
  sourceRelation: ReferenceSourceRelation;
  maxPixelsSingle: number | null;
  maxPixelsMulti: number | null;
  /** The server's own sentence for a hidden block, for refusal copy. */
  reason: string | null;
  /**
   * The adapter's injection-strength control (`GenerateRequest.reference_weight`),
   * or `null` where this reference protocol has no strength at all.
   *
   * The RANGE travels with the capability on purpose, so a surface renders the
   * slider from the server's own bounds instead of hard-coding them the way
   * `id_weight` forces every client to hard-code `0..3`. `null` is BOTH "an
   * older host that never sent the field" and "a recipe with no adapter", and
   * the two want the same thing here: render no slider. That is safe because a
   * recipe with no adapter has no strength to set.
   */
  weight: FloatControl | null;
  /**
   * How the default canvas follows the references (`last-reference`: the
   * last one's aspect at the recipe default area — `referenceCanvas.ts`), or
   * `null` for no rule / an older server.
   */
  canvas: ReferenceCanvasRule | null;
  /**
   * The still containers accepted as references. Never empty: an empty
   * advertised list means the legacy PNG/JPEG pair. References are NEVER
   * flattened or re-encoded on the way out — a transparent PNG or WebP
   * reference travels with its alpha.
   */
  formats: ImageInputFormat[];
}

/**
 * Project the advertised block onto the client shape. A `hidden` block is the
 * server SAYING NO — it answers `null` exactly like an absent one, and the
 * caller must not then fall back to a family sniff (only absence does that).
 */
export function referenceImagesFromProfile(
  profile: ReferenceImagesProfile,
): ReferenceImagesCapabilities | null {
  if (profile.mode === "hidden") return null;
  return {
    required: profile.required,
    max: profile.max_count ?? null,
    primaryIsTarget: profile.primary_is_target,
    sourceRelation: profile.source_relation,
    maxPixelsSingle: profile.max_pixels_single ?? null,
    maxPixelsMulti: profile.max_pixels_multi ?? null,
    reason: profile.reason ?? null,
    weight: profile.weight ?? null,
    canvas: profile.canvas ?? null,
    formats: profile.formats?.length
      ? profile.formats.slice()
      : LEGACY_REFERENCE_IMAGE_FORMATS.slice(),
  };
}

/**
 * Read `capabilities.reference_images` off an advertised recipe's capability
 * block. `null` is an OLDER SERVER, which is why this is a separate question
 * from a `hidden` block above.
 */
export function advertisedReferenceImages(
  capabilities:
    { reference_images?: ReferenceImagesProfile | null } | null | undefined,
): ReferenceImagesProfile | null {
  return capabilities?.reference_images ?? null;
}

const IMAGE_INPUT_LABEL: Record<ImageInputFormat, string> = {
  png: "PNG",
  jpeg: "JPEG",
  webp: "WebP",
};

const IMAGE_INPUT_EXTENSION: Record<ImageInputFormat, RegExp> = {
  png: /\.png$/i,
  jpeg: /\.jpe?g$/i,
  webp: /\.webp$/i,
};

/** "PNG or JPEG", "PNG, JPEG, or WebP" — the picker's refusal wording. */
export function imageInputFormatsSentence(
  formats: readonly ImageInputFormat[],
): string {
  const labels = formats.map((format) => IMAGE_INPUT_LABEL[format]);
  if (labels.length <= 1) return labels[0] ?? "";
  if (labels.length === 2) return `${labels[0]} or ${labels[1]}`;
  return `${labels.slice(0, -1).join(", ")}, or ${labels[labels.length - 1]}`;
}

/** The container a filename names, by extension; `null` for anything else. */
export function imageInputFormatForName(
  filename: string,
): ImageInputFormat | null {
  const name = filename.trim();
  for (const format of ["png", "jpeg", "webp"] as const) {
    if (IMAGE_INPUT_EXTENSION[format].test(name)) return format;
  }
  return null;
}

/**
 * Whether a picked file is one of these containers — by its MIME type, or by
 * its extension when the platform reports none (drag-and-drop from some file
 * managers).
 */
export function fileMatchesImageInputFormats(
  file: { type: string; name: string },
  formats: readonly ImageInputFormat[],
): boolean {
  if (file.type) {
    return referenceImageMimeTypes(formats).includes(file.type);
  }
  const format = imageInputFormatForName(file.name);
  return format !== null && formats.includes(format);
}

/**
 * The container of a base64 payload (raw or a data URL), from its first
 * bytes — `null` when it is none of PNG/JPEG/WebP or not base64 at all.
 */
export function imageInputFormatOfBase64(
  base64: string,
): ImageInputFormat | null {
  const comma = base64.indexOf(",");
  const payload = (comma >= 0 ? base64.slice(comma + 1) : base64)
    .replace(/\s+/g, "")
    .slice(0, 16);
  try {
    const binary = globalThis.atob(payload);
    return sniffImageInputFormat(
      Uint8Array.from(binary, (character) => character.charCodeAt(0)),
    );
  } catch {
    return null;
  }
}
