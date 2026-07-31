/*
 * LTX-2 advanced guidance overrides — the one place a surface turns the
 * Advanced controls into the additive `guidance_overrides` wire object and
 * back.
 *
 * Two rules make the contract safe, and both live here rather than in any
 * component:
 *
 * 1. An untouched control must not appear on the wire. The engine keeps a
 *    pipeline constant only while its field is absent, so a null that
 *    serializes as `0` would silently change every render.
 * 2. A control the user did touch must reach the server or say why it
 *    cannot. `stgBlocks` is free text and `skipStep` is a `u32` the wire
 *    cannot carry fractionally, so a value neither can express is reported
 *    as an error rather than quietly dropped.
 */

/** Additive `guidance_overrides` request/metadata object. Every field is
 * optional: an absent field keeps the engine's pipeline constant. */
export interface Ltx2GuidanceOverrides {
  stg_scale?: number | null;
  stg_blocks?: number[] | null;
  rescale_scale?: number | null;
  modality_scale?: number | null;
  skip_step?: number | null;
}

/** Form-side mirror of the wire object. `stgBlocks` is free text because the
 * control accepts a comma-separated block list. */
export interface Ltx2GuidanceOverridesState {
  stgScale: number | null;
  stgBlocks: string;
  rescaleScale: number | null;
  modalityScale: number | null;
  skipStep: number | null;
}

/** Deepest transformer block any shipped LTX-2 checkpoint has. Mirrors
 * `MAX_STG_BLOCK_INDEX` in `mold-core`'s validation; the server re-checks
 * against the resolved checkpoint's real depth. */
export const MAX_STG_BLOCK_INDEX = 64;
/** Maximum simultaneously perturbed STG blocks. Mirrors `MAX_STG_BLOCKS`. */
export const MAX_STG_BLOCKS = 8;
/** Ceiling shared by the STG and modality scales. Mirrors `MAX_SCALE`. */
export const MAX_GUIDANCE_SCALE = 10;
/** Ceiling for the guidance skip stride. Mirrors `MAX_SKIP_STEP`. */
export const MAX_GUIDANCE_SKIP_STEP = 8;

export function emptyGuidanceOverrides(): Ltx2GuidanceOverridesState {
  return {
    stgScale: null,
    stgBlocks: "",
    rescaleScale: null,
    modalityScale: null,
    skipStep: null,
  };
}

/** True when nothing is set — the request should carry no override object. */
export function guidanceOverridesAreEmpty(
  state: Ltx2GuidanceOverridesState | null | undefined,
): boolean {
  return guidanceOverrideCount(state) === 0;
}

/** How many overrides are active, for the Advanced section's count badge. */
export function guidanceOverrideCount(
  state: Ltx2GuidanceOverridesState | null | undefined,
): number {
  if (!state) return 0;
  let count = 0;
  if (state.stgScale !== null) count += 1;
  if (state.stgBlocks.trim() !== "") count += 1;
  if (state.rescaleScale !== null) count += 1;
  if (state.modalityScale !== null) count += 1;
  if (state.skipStep !== null) count += 1;
  return count;
}

/** Parse the comma-separated block list. Returns the message the control
 * shows inline, or null when the text is a usable (or empty) list. */
export function stgBlocksError(text: string): string | null {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  const entries = trimmed
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry !== "");
  if (entries.length === 0) return "List at least one block index.";
  if (entries.length > MAX_STG_BLOCKS) {
    return `At most ${MAX_STG_BLOCKS} blocks.`;
  }
  const seen = new Set<number>();
  for (const entry of entries) {
    if (!/^\d+$/.test(entry)) {
      return `"${entry}" is not a block index.`;
    }
    const block = Number(entry);
    if (block >= MAX_STG_BLOCK_INDEX) {
      return `Block ${block} is deeper than any LTX-2 checkpoint.`;
    }
    if (seen.has(block)) return `Block ${block} is listed twice.`;
    seen.add(block);
  }
  return null;
}

/** Validate the guidance skip stride. It is a `u32` on the wire, so a
 * fractional value would not even deserialize — the request would come back
 * as an opaque body-parse failure instead of a field-named 422. */
export function skipStepError(value: number | null): string | null {
  if (value === null) return null;
  if (!Number.isFinite(value)) return "Enter a whole number of steps.";
  if (!Number.isInteger(value)) {
    return "The skip stride is a whole number of steps.";
  }
  if (value < 0 || value > MAX_GUIDANCE_SKIP_STEP) {
    return `Enter a stride between 0 and ${MAX_GUIDANCE_SKIP_STEP}.`;
  }
  return null;
}

function scaleError(
  value: number | null,
  label: string,
  max: number,
): string | null {
  if (value === null) return null;
  if (!Number.isFinite(value)) return `${label} must be a number.`;
  if (value < 0 || value > max) {
    return `${label} must be between 0 and ${max}.`;
  }
  return null;
}

/** One message covering every override control, for a surface's submit gate.
 * The bounds mirror `mold-core`'s validation so a request that would come
 * back 422 is caught before the round trip; the server stays the authority. */
export function guidanceOverridesError(
  state: Ltx2GuidanceOverridesState | null | undefined,
): string | null {
  if (!state) return null;
  const blocks = stgBlocksError(state.stgBlocks);
  if (blocks) return `STG blocks: ${blocks}`;
  const skip = skipStepError(state.skipStep);
  if (skip) return `Guidance skip stride: ${skip}`;
  return (
    scaleError(state.stgScale, "STG scale", MAX_GUIDANCE_SCALE) ??
    scaleError(state.rescaleScale, "CFG rescale", 1) ??
    scaleError(state.modalityScale, "Modality scale", MAX_GUIDANCE_SCALE)
  );
}

function parseStgBlocks(text: string): number[] | null {
  if (stgBlocksError(text) !== null) return null;
  const blocks = text
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry !== "")
    .map(Number);
  return blocks.length > 0 ? blocks : null;
}

/** Build the wire object, or `undefined` when nothing is set. A value the
 * wire cannot carry — an unparsable block list, a fractional skip stride —
 * contributes nothing; the caller surfaces `guidanceOverridesError` before
 * submitting so this can never be the user's only feedback. */
export function guidanceOverridesToWire(
  state: Ltx2GuidanceOverridesState | null | undefined,
): Ltx2GuidanceOverrides | undefined {
  if (!state) return undefined;
  const wire: Ltx2GuidanceOverrides = {};
  if (state.stgScale !== null) wire.stg_scale = state.stgScale;
  const blocks = parseStgBlocks(state.stgBlocks);
  if (blocks) wire.stg_blocks = blocks;
  if (state.rescaleScale !== null) wire.rescale_scale = state.rescaleScale;
  if (state.modalityScale !== null) wire.modality_scale = state.modalityScale;
  // `skip_step` is a `u32`: a fractional value fails JSON deserialization
  // outright, so it is omitted here the same way an unparsable block list is,
  // and the surface's submit gate reports it via `guidanceOverridesError`.
  if (skipStepError(state.skipStep) === null && state.skipStep !== null) {
    wire.skip_step = state.skipStep;
  }
  return Object.keys(wire).length > 0 ? wire : undefined;
}

/** Restore form state from saved metadata or a reused request. */
export function guidanceOverridesFromWire(
  wire: Ltx2GuidanceOverrides | null | undefined,
): Ltx2GuidanceOverridesState {
  const state = emptyGuidanceOverrides();
  if (!wire) return state;
  state.stgScale = wire.stg_scale ?? null;
  state.stgBlocks = wire.stg_blocks?.length ? wire.stg_blocks.join(", ") : "";
  state.rescaleScale = wire.rescale_scale ?? null;
  state.modalityScale = wire.modality_scale ?? null;
  state.skipStep = wire.skip_step ?? null;
  return state;
}
