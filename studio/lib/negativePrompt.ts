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

import { isWanFamily } from "./generationCapabilities";

/** The `/api/models` row fields the negative default depends on. */
export type NegativeDefaultModel = {
  default_negative_prompt?: string | null;
};

/**
 * Wan's tuned default negative — the engine's absence fallback
 * (`mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT`; upstream
 * `Wan2.2/wan/configs/shared_config.py`). This is the browser-side authority
 * for the family constant; the TUI parity test in
 * `crates/mold-tui/src/ui/create_form.rs` pins it byte-for-byte against the
 * Rust constant so the two can never drift.
 */
export const WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT =
  "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

/**
 * The model's *effective* default negative: the advertised additive field
 * when present, else the family constant for a family whose engine
 * substitutes one anyway (wan). A known default must survive additive-field
 * absence — reconciling the same wan model against an older server that
 * omits `default_negative_prompt` would otherwise collapse the stored
 * default to `""`, at which point an explicit `""` opt-out serializes as
 * absence and silently re-enables the engine fallback. Mirrors
 * `create_form::effective_negative_default` on the TUI.
 */
export function effectiveNegativeDefault(
  model: NegativeDefaultModel | null | undefined,
  family: string | null | undefined,
): string {
  const advertised = advertisedNegativeDefault(model);
  if (advertised !== "") return advertised;
  return isWanFamily(family ?? "") ? WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT : "";
}

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
  explicitClear = false,
): string {
  if (explicitClear && current.trim() === "") return current;
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
  explicitClear = false,
): string | undefined {
  const trimmed = text.trim();
  const normalizedDefault = advertisedDefault.trim();
  if (explicitClear && trimmed === "") return "";
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

/**
 * Whether restored metadata carried the explicit `""` opt-out (#787 round 3).
 * A restore can land before the model rows do, when the advertised default is
 * still unknown — at that moment an explicit clear and "untouched" both show
 * an empty control and are indistinguishable afterwards. This marker carries
 * the restore-time authority forward: `negativePromptOnDefaultChange` keeps a
 * marked empty control cleared when the default finally resolves, and
 * `negativePromptWireValue` serializes it as `""` instead of letting absence
 * silently re-enable the conditioning the print explicitly disabled. The
 * marker resets whenever the user selects a model (`applyModelDefaults` /
 * `reconcileModelCapabilities` on a model switch), and typed text bypasses it.
 */
export function restoredNegativeExplicitClear(
  saved: string | null | undefined,
): boolean {
  return saved != null && saved.trim() === "";
}
