/*
 * Wan sampler-recipe controls — the flow shift (#782) and the per-expert
 * Lightning distill strengths (#795) — turned into their additive request
 * fields and back. The sample solver rides the ordinary `scheduler` field, so
 * it is not part of this state; `schedulerOptions` in `generationCapabilities`
 * narrows it to the three solvers wan accepts.
 *
 * The same two rules that govern LTX-2's `guidance_overrides` govern these,
 * and for the same reason:
 *
 * 1. An untouched control must not appear on the wire. The engine keeps the
 *    tier's own shift and distill strengths only while the field is absent, so
 *    a null serialized as `0` would silently change every render.
 * 2. A touched control must reach the server or say why it cannot. Both bands
 *    mirror `mold-core`'s validation, which rejects — never ignores — an
 *    out-of-band or off-family value.
 */
import type { WanRecipeCapabilities } from "./generationCapabilities";

/** Additive wan recipe fields on `GenerateRequest` / `OutputMetadata`. Every
 * field is optional: absent keeps the resolved tier's own value. */
export interface WanRecipeWire {
  sample_shift?: number | null;
  distill_strength_high?: number | null;
  distill_strength_low?: number | null;
}

/** Form-side mirror of the wire fields. */
export interface WanRecipeState {
  sampleShift: number | null;
  distillStrengthHigh: number | null;
  distillStrengthLow: number | null;
}

/** Upper bound of the accepted distill band. Mirrors the `(0, 4]` check in
 * `mold-core`'s validation and `resolve_distill_strength`. */
export const MAX_WAN_DISTILL_STRENGTH = 4;

export function emptyWanRecipe(): WanRecipeState {
  return {
    sampleShift: null,
    distillStrengthHigh: null,
    distillStrengthLow: null,
  };
}

/** How many recipe controls are set, for the Advanced section's count badge.
 * Capability gating is the caller's job — the same split `guidanceOverrideCount`
 * uses, so a stale value on a family that hides the controls never counts. */
export function wanRecipeCount(
  state: WanRecipeState | null | undefined,
): number {
  if (!state) return 0;
  let count = 0;
  if (state.sampleShift !== null) count += 1;
  if (state.distillStrengthHigh !== null) count += 1;
  if (state.distillStrengthLow !== null) count += 1;
  return count;
}

/** Validate the flow shift. The server accepts any finite positive value —
 * there is no upper bound, because the useful range moves with resolution and
 * frame count. */
export function sampleShiftError(value: number | null): string | null {
  if (value === null) return null;
  if (!Number.isFinite(value) || value <= 0) {
    return "Flow shift must be a positive number.";
  }
  return null;
}

/** Validate one distill strength against the accepted `(0, 4]` band. */
export function distillStrengthError(
  value: number | null,
  label: string,
): string | null {
  if (value === null) return null;
  if (!Number.isFinite(value) || value <= 0) {
    return `${label} distill strength must be greater than 0.`;
  }
  if (value > MAX_WAN_DISTILL_STRENGTH) {
    return `${label} distill strength must be at most ${MAX_WAN_DISTILL_STRENGTH}.`;
  }
  return null;
}

/** One message covering every recipe control, for a surface's submit gate.
 * The bands mirror `mold-core`'s validation so a request that would come back
 * 422 is caught before the round trip; the server stays the authority. */
export function wanRecipeError(
  state: WanRecipeState | null | undefined,
): string | null {
  if (!state) return null;
  return (
    sampleShiftError(state.sampleShift) ??
    distillStrengthError(state.distillStrengthHigh, "High-noise") ??
    distillStrengthError(state.distillStrengthLow, "Low-noise")
  );
}

/**
 * Build the additive wire fields for the resolved model's capabilities.
 * Returns `{}` when nothing is set or the model takes none of these controls —
 * a value the server would reject off-family or off-tier contributes nothing,
 * and the caller surfaces `wanRecipeError` before submitting so this can never
 * be the user's only feedback.
 */
export function wanRecipeToWire(
  state: WanRecipeState | null | undefined,
  capabilities: WanRecipeCapabilities,
): WanRecipeWire {
  if (!state || !capabilities.supported) return {};
  const wire: WanRecipeWire = {};
  if (
    sampleShiftError(state.sampleShift) === null &&
    state.sampleShift !== null
  ) {
    wire.sample_shift = state.sampleShift;
  }
  if (!capabilities.supportsDistillStrength) return wire;
  if (
    distillStrengthError(state.distillStrengthHigh, "High-noise") === null &&
    state.distillStrengthHigh !== null
  ) {
    wire.distill_strength_high = state.distillStrengthHigh;
  }
  if (
    distillStrengthError(state.distillStrengthLow, "Low-noise") === null &&
    state.distillStrengthLow !== null
  ) {
    wire.distill_strength_low = state.distillStrengthLow;
  }
  return wire;
}

/** Restore form state from saved metadata or a reused request. */
export function wanRecipeFromWire(
  wire: WanRecipeWire | null | undefined,
): WanRecipeState {
  const state = emptyWanRecipe();
  if (!wire) return state;
  state.sampleShift = wire.sample_shift ?? null;
  state.distillStrengthHigh = wire.distill_strength_high ?? null;
  state.distillStrengthLow = wire.distill_strength_low ?? null;
  return state;
}
