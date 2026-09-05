import type { IntegerControl } from "@studio/lib/generated/generationProfileV1";

/**
 * Draft / Good / Best, built from the recipe's OWN recommended ladder — never
 * a client ladder. The host publishes the rungs it stands behind; Draft is the
 * lowest, Good the recipe's default, Best the highest. The control's raw floor
 * and ceiling are admission bounds, not advice: at FLUX's 50-step ceiling a
 * print costs twice a 38-step one and looks the same, and its 1-step floor is
 * not a draft of that picture but a different, broken one. So the rows never
 * reach past the ladder.
 *
 * A recipe that pins its steps (a guidance-distilled tier's schedule) has no
 * choice to offer, and a host older than the ladder advertises none: the rows
 * collapse to nothing and the profile's own note under the Detail slider stays
 * the whole explanation.
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

export function qualityPresets(steps: IntegerControl | null | undefined): QualityPreset[] {
  if (!steps || steps.mode !== "adjustable") return [];
  // Sorted defensively: the rows are lowest / default / highest whatever order
  // the host listed its rungs in.
  const ladder = [...(steps.recommended ?? [])].sort((a, b) => a - b);
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
