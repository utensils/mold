/**
 * Device-placement types and pure helpers, mirrored from mold-core's
 * `DevicePlacement` (crates/mold-core/src/types.rs) and the web SPA's
 * `usePlacement` composable / `PlacementPanel`. Kept dependency-free so the
 * mapping, tier-2 gating, and reducer logic are unit-testable in isolation.
 */

/** Where a component should run. Serde externally-tagged, snake_case. */
export type DeviceRef = { kind: "auto" } | { kind: "cpu" } | { kind: "gpu"; ordinal: number };

/**
 * Per-component overrides, available only for tier-2 families. `transformer`
 * and `vae` always carry a `DeviceRef` (default auto). The per-encoder fields
 * are nullable: `null` means "follow the tier-1 text-encoders group knob".
 */
export interface AdvancedPlacement {
  transformer: DeviceRef;
  vae: DeviceRef;
  clip_l?: DeviceRef | null;
  clip_g?: DeviceRef | null;
  t5?: DeviceRef | null;
  qwen?: DeviceRef | null;
}

/** Top-level placement persisted under `[models."name:tag".placement]`. */
export interface DevicePlacement {
  text_encoders: DeviceRef;
  advanced?: AdvancedPlacement | null;
}

/**
 * Families that expose the Advanced (tier-2) per-component disclosure. Mirrors
 * `TIER2_FAMILIES` in web/src/composables/usePlacement.ts — keep in lock-step.
 * SD3.5 is intentionally absent: the engine only honors tier 1 for it.
 */
export const TIER2_FAMILIES: readonly string[] = [
  "flux",
  "flux2",
  "flux.2",
  "flux2-klein",
  "z-image",
  "qwen-image",
  "qwen_image",
];

export function supportsAdvanced(family: string): boolean {
  return TIER2_FAMILIES.includes(family);
}

/** DeviceRef → `<select>` option value (`auto` / `cpu` / `gpu:<n>`). */
export function refToOption(r: DeviceRef): string {
  if (r.kind === "auto") return "auto";
  if (r.kind === "cpu") return "cpu";
  return `gpu:${r.ordinal}`;
}

/** `<select>` option value → DeviceRef; anything unrecognized becomes auto. */
export function optionToRef(opt: string): DeviceRef {
  if (opt === "auto") return { kind: "auto" };
  if (opt === "cpu") return { kind: "cpu" };
  const m = /^gpu:(\d+)$/.exec(opt);
  return m ? { kind: "gpu", ordinal: Number(m[1]) } : { kind: "auto" };
}

export function defaultAdvanced(): AdvancedPlacement {
  return {
    transformer: { kind: "auto" },
    vae: { kind: "auto" },
    clip_l: null,
    clip_g: null,
    t5: null,
    qwen: null,
  };
}

export function defaultPlacement(): DevicePlacement {
  return { text_encoders: { kind: "auto" }, advanced: null };
}

/** Set the tier-1 group knob, preserving any existing advanced block. */
export function setTextEncoders(current: DevicePlacement | null, ref: DeviceRef): DevicePlacement {
  return { text_encoders: ref, advanced: current?.advanced ?? null };
}

/**
 * Pure reducer for a single advanced field. `transformer` and `vae` never go
 * null (a null coerces to auto); the per-encoder fields accept null to mean
 * "follow the group knob". Mirrors usePlacement.ts:70-85.
 */
export function setAdvancedField(
  current: DevicePlacement | null,
  field: keyof AdvancedPlacement,
  ref: DeviceRef | null,
): DevicePlacement {
  const base: DevicePlacement = current ?? { text_encoders: { kind: "auto" }, advanced: null };
  const adv: AdvancedPlacement = { ...(base.advanced ?? defaultAdvanced()) };
  if (field === "transformer" || field === "vae") {
    adv[field] = ref ?? { kind: "auto" };
  } else {
    adv[field] = ref;
  }
  return { ...base, advanced: adv };
}
