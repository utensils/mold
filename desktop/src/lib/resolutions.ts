export * from "@studio/lib/resolutions";

import { ASPECTS } from "@ui/lib/resolution";
import {
  closestResolutionPreset,
  presetsForFamily,
  presetsNearRatio,
} from "@studio/lib/resolutions";

export function aspectIdFor(width: number, height: number, tolerance = 0.075): string | null {
  if (!(width > 0) || !(height > 0)) return null;
  const ratio = width / height;
  let best: { id: string; diff: number } | null = null;
  for (const aspect of ASPECTS) {
    const diff = Math.abs(ratio - aspect.ratio) / aspect.ratio;
    if (!best || diff < best.diff) best = { id: aspect.id, diff };
  }
  return best && best.diff <= tolerance ? best.id : null;
}

export function applyMp(
  form: { width: number; height: number; family?: string },
  mp: number,
): void {
  const candidates = presetsNearRatio(
    presetsForFamily(form.family ?? "flux"),
    form.width / form.height,
  );
  const preset = [...candidates].sort(
    (a, b) =>
      Math.abs((a.width * a.height) / 1_000_000 - mp) -
      Math.abs((b.width * b.height) / 1_000_000 - mp),
  )[0];
  if (preset) {
    form.width = preset.width;
    form.height = preset.height;
  }
}

export function applyAspectId(
  form: { width: number; height: number; family?: string },
  id: string,
): void {
  const aspect = ASPECTS.find((option) => option.id === id);
  if (!aspect) return;
  const preset = closestResolutionPreset(
    presetsNearRatio(presetsForFamily(form.family ?? "flux"), aspect.ratio),
    form.width,
    form.height,
  );
  if (preset) {
    form.width = preset.width;
    form.height = preset.height;
  }
}
