/**
 * The transparent-background contract (`GenerateRequest.transparent_background`,
 * advertised as `capabilities.transparency`).
 *
 * `mold_core::generation_profile::transparency_for_recipe` answers it once for
 * the server, admission, the CLI and every GUI; this module is the one browser
 * reading. ABSENCE of the block means an OLDER SERVER, and there is no older
 * behaviour to fall back to — the toggle is simply not offered. There is no
 * family sniff here on purpose: a toggle an older host would silently DROP
 * would render an opaque picture the user asked to be transparent.
 *
 * The prompt recipe ("This is an RGBA image with transparency. …") is applied
 * by the ENGINE, never by a client: the stored prompt, the composer, Reuse and
 * the expander all keep the user's own words.
 */

import type {
  ControlMode,
  TransparencyCapabilitiesProfile,
} from "./generated/generationProfileV1";

export type { TransparencyCapabilitiesProfile } from "./generated/generationProfileV1";

/** The client-side projection of `capabilities.transparency`. */
export interface TransparencyCapabilities {
  mode: ControlMode;
  /** The toggle's default position (always `false` today). */
  default: boolean;
  /** Output containers that carry alpha, narrowed to what the binary encodes. */
  formats: string[];
  /**
   * An alpha-carrying REFERENCE keeps alpha in the output even with the
   * toggle off (Qwen Image 2.1's four-channel VAE).
   */
  nativeAlpha: boolean;
  /** The server's own sentence for a hidden block. */
  reason: string | null;
}

/** What a surface needs to render (and serialize) the toggle. */
export interface TransparencyControl {
  default: boolean;
  formats: readonly string[];
  nativeAlpha: boolean;
}

/** The toggle's label on every surface. */
export const TRANSPARENCY_LABEL = "Transparent background";

/** The one-line explanation under the toggle. */
export const TRANSPARENCY_NOTE =
  "Cut the subject out onto a transparent background (PNG or WebP).";

/**
 * Why JPEG is disabled while the toggle is on — JPEG has no alpha channel, so
 * admission refuses the pair (`validate_transparency_against`) rather than
 * silently flattening the cut-out.
 */
export const TRANSPARENCY_UNAVAILABLE_FORMAT_REASON =
  "JPEG has no transparency, so a transparent background saves as PNG or WebP.";

/** Project the advertised block; `null` is an older server. */
export function transparencyFromProfile(
  profile: TransparencyCapabilitiesProfile | null | undefined,
): TransparencyCapabilities | null {
  if (!profile) return null;
  return {
    mode: profile.mode,
    default: profile.default,
    formats: profile.formats.slice(),
    nativeAlpha: profile.native_alpha,
    reason: profile.reason ?? null,
  };
}

/**
 * The toggle to render, or `null` where none may be offered: an older server,
 * a `hidden` recipe, or an adjustable one whose alpha formats the binary
 * cannot encode at all (nothing the user could pick would carry the alpha).
 */
export function transparencyControl(
  capabilities:
    { transparency: TransparencyCapabilities | null } | null | undefined,
): TransparencyControl | null {
  const block = capabilities?.transparency ?? null;
  if (!block || block.mode !== "adjustable" || block.formats.length === 0) {
    return null;
  }
  return {
    default: block.default,
    formats: block.formats.slice(),
    nativeAlpha: block.nativeAlpha,
  };
}

/** Whether a request built now carries the toggle. */
export function transparencyActive(
  enabled: boolean | null | undefined,
  control: TransparencyControl | null,
): boolean {
  return enabled === true && control !== null;
}

/**
 * Resolve the output format while the toggle is on: a format with no alpha
 * channel moves to the first advertised alpha format, and the returned note
 * says so. Off (or unavailable) leaves the format alone.
 */
export function coerceFormatForTransparency<F extends string>(
  format: F,
  control: TransparencyControl | null,
  enabled: boolean | null | undefined,
): { format: F; note: string | null } {
  if (!transparencyActive(enabled, control) || !control) {
    return { format, note: null };
  }
  if (control.formats.includes(format)) return { format, note: null };
  return {
    format: control.formats[0] as F,
    note: TRANSPARENCY_UNAVAILABLE_FORMAT_REASON,
  };
}

/**
 * The request fields the toggle contributes. Only `true` ever travels: off is
 * the ABSENCE of the field, so an ordinary render is byte-identical on the
 * wire to one from a client that predates it. A toggle left on for a model
 * that does not advertise it parks — kept in the form, absent from the wire.
 */
export function transparencyRequestFields(
  enabled: boolean | null | undefined,
  control: TransparencyControl | null,
): { transparent_background?: true } {
  return transparencyActive(enabled, control)
    ? { transparent_background: true }
    : {};
}
