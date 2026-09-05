import type { IntegerControl } from "@studio/lib/generated/generationProfileV1";

/**
 * Draft / Good / Best, built from the recipe's recommended ladder. The host
 * publishes the rungs it stands behind; Draft is the lowest, Good the recipe's
 * default, Best the highest. The control's raw floor and ceiling are admission
 * bounds, not advice: at FLUX's 100-step ceiling a print costs four times a
 * 38-step one and looks the same, and its 1-step floor is not a draft of that
 * picture but a different, broken one. So the rows never reach the bounds.
 *
 * A host older than the ladder advertises one rung, or none at all (every
 * 0.27.x server does). An absent additive field means an OLDER SERVER, never
 * "no choice offered" — the supports_strength lesson — so the client stands in
 * with the profile's own formula (`generation_profile::steps_ladder`: half,
 * default, one and a half times, clamped into the control's bounds, deduped)
 * rather than hiding the group on every remote-only style. A recipe that pins
 * its steps (a guidance-distilled tier's schedule) has no choice to offer, and
 * the profile's own note under the Detail slider stays the whole explanation.
 */
export interface QualityPreset {
  key: "draft" | "good" | "best";
  label: string;
  steps: number;
}

const ROWS = [
  { key: "draft", label: "Draft", pick: (_c: IntegerControl, ladder: number[]) => ladder[0]! },
  { key: "good", label: "Good", pick: (c: IntegerControl) => c.default },
  {
    key: "best",
    label: "Best",
    pick: (_c: IntegerControl, ladder: number[]) => ladder[ladder.length - 1]!,
  },
] as const;

/** The host's formula, for a host that predates it. Mirrors `steps_ladder`. */
function standInLadder(c: IntegerControl): number[] {
  const rungs = [
    Math.max(c.min, Math.ceil(c.default / 2)),
    c.default,
    Math.min(c.max, Math.ceil((c.default * 3) / 2)),
  ];
  return [...new Set(rungs)].sort((a, b) => a - b);
}

export function qualityPresets(steps: IntegerControl | null | undefined): QualityPreset[] {
  if (!steps || steps.mode !== "adjustable") return [];
  // Sorted defensively: the rows are lowest / default / highest whatever order
  // the host listed its rungs in.
  const advertised = [...(steps.recommended ?? [])].sort((a, b) => a - b);
  const ladder = advertised.length >= 2 ? advertised : standInLadder(steps);
  if (ladder.length < 2) return [];
  const presets: QualityPreset[] = [];
  for (const row of ROWS) {
    const value = row.pick(steps, ladder);
    if (presets.some((preset) => preset.steps === value)) continue;
    presets.push({ key: row.key, label: row.label, steps: value });
  }
  return presets.length > 1 ? presets : [];
}

/** The row the current step count reads as, or null when it matches none. */
export function activeQualityPreset(
  presets: readonly QualityPreset[],
  steps: number,
): QualityPreset["key"] | null {
  return presets.find((preset) => preset.steps === steps)?.key ?? null;
}
