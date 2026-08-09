/**
 * Cross-surface contract for the server-advertised default negative prompt
 * (`/api/models[].default_negative_prompt`, additive — wan today, whose
 * checkpoints were trained against a specific long Chinese negative that the
 * engine substitutes whenever a request carries no `negative_prompt`).
 *
 * Web, desktop, and iPhone derive their Negative-control behavior from these
 * helpers; `crates/mold-tui/src/ui/create_form.rs` mirrors them exactly for
 * the terminal surface. The shared tri-state (#787):
 *
 * - **untouched** — the control shows the advertised default and the wire
 *   field stays ABSENT, so the server applies the same default and older
 *   servers keep today's behavior byte-for-byte;
 * - **cleared** — the wire carries an explicit `""`, the empty uncond the
 *   engine honors as an opt-out and never re-substitutes;
 * - **typed** — the text replaces the default verbatim.
 *
 * The engine trims, so comparisons are on trimmed text throughout.
 */

/** The `/api/models` row fields the negative default depends on. */
export type NegativeDefaultModel = {
  default_negative_prompt?: string | null;
};

/** The model's advertised default negative, normalized ("" when none). */
export function advertisedNegativeDefault(
  model: NegativeDefaultModel | null | undefined,
): string {
  return model?.default_negative_prompt?.trim() ?? "";
}

/**
 * What the Negative control should show after the advertised default changes
 * (model switch, or a fresher catalog for the same model). A control still
 * showing the previous default follows the new model — that is also how the
 * default first appears, since "empty with no previous default" is the
 * untouched state. Typed text is user authority and survives; so does an
 * explicit clear made while a default was advertised, which keeps the
 * opt-out across a wan→wan switch.
 */
export function negativePromptOnDefaultChange(
  current: string,
  previousDefault: string,
  nextDefault: string,
): string {
  return current.trim() === previousDefault.trim()
    ? nextDefault.trim()
    : current;
}

/**
 * The wire value for `negative_prompt` given the control's text and the
 * advertised default. `undefined` means "leave the field absent"; `""` is
 * the explicit empty-uncond opt-out and MUST be serialized, not dropped.
 * Callers still gate on the family's `supportsNegativePrompt` capability.
 */
export function negativePromptWireValue(
  text: string,
  advertisedDefault: string,
): string | undefined {
  const trimmed = text.trim();
  const normalizedDefault = advertisedDefault.trim();
  if (normalizedDefault === "") {
    return trimmed === "" ? undefined : trimmed;
  }
  return trimmed === normalizedDefault ? undefined : trimmed;
}

/**
 * The Negative control's text when restoring saved settings (gallery
 * metadata or a queued request). Absence predates truthful recording — for a
 * model with an advertised default it means the default conditioned the
 * render, so restoring an empty control would silently flip the reuse into
 * an explicit opt-out. A recorded `""` was a real empty uncond and stays.
 */
export function restoredNegativePrompt(
  saved: string | null | undefined,
  advertisedDefault: string,
): string {
  return saved ?? advertisedDefault.trim();
}
