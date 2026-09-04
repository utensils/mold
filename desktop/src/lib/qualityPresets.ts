import type { IntegerControl } from "@studio/lib/generated/generationProfileV1";

/**
 * Draft / Good / Best, built from the recipe's OWN steps control — never a
 * client ladder. Draft is the profile's floor, Good its default, Best its
 * ceiling, so a host that widens the range widens the rows with no client
 * release and a pick can never land outside admission.
 *
 * A recipe that pins its steps (a guidance-distilled tier's schedule) has no
 * choice to offer: the rows collapse to nothing and the profile's own note
 * under the Detail slider stays the whole explanation.
 */
export interface QualityPreset {
  key: "draft" | "good" | "best";
  label: string;
  steps: number;
}

const ROWS = [
  { key: "draft", label: "Draft", pick: (c: IntegerControl) => c.min },
  { key: "good", label: "Good", pick: (c: IntegerControl) => c.default },
  { key: "best", label: "Best", pick: (c: IntegerControl) => c.max },
] as const;

export function qualityPresets(steps: IntegerControl | null | undefined): QualityPreset[] {
  if (!steps || steps.mode !== "adjustable") return [];
  const presets: QualityPreset[] = [];
  for (const row of ROWS) {
    const value = row.pick(steps);
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
