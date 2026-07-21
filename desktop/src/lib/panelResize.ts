/**
 * Resizable panel widths (px). Pure math — no DOM. The persisted value lives
 * in settings.json (`navRailWidth` / `generateParamsWidth`, null = default);
 * these helpers own the clamping so every surface agrees on the limits.
 */
export const PANEL_LIMITS = {
  // 210 default per the Mold Studio prototype sidebar.
  navRail: { min: 160, def: 210, max: 320 },
  // 340 keeps all five 52px aspect controls on one row after 18px side padding.
  generateParams: { min: 280, def: 340, max: 480 },
} as const;

export type PanelId = keyof typeof PANEL_LIMITS;

/** Clamp a width into the panel's limits, rounded to whole pixels. */
export function clampPanelWidth(panel: PanelId, px: number): number {
  const { min, max } = PANEL_LIMITS[panel];
  return Math.min(max, Math.max(min, Math.round(px)));
}

/**
 * Width for a drag in progress. A handle on the panel's `right` edge grows
 * the panel with +dx; a handle on its `left` edge (a panel docked to the
 * window's right side, like the Generate inspector) grows with −dx.
 */
export function dragWidth(
  panel: PanelId,
  startWidth: number,
  dx: number,
  edge: "right" | "left",
): number {
  return clampPanelWidth(panel, startWidth + (edge === "right" ? dx : -dx));
}

/**
 * Normalize a persisted settings value at load time: anything that is not a
 * finite number (null, undefined, strings, NaN, ±Infinity) falls back to the
 * panel's default; finite numbers are clamped.
 */
export function normalizePanelWidth(panel: PanelId, raw: unknown): number {
  if (typeof raw !== "number" || !Number.isFinite(raw)) return PANEL_LIMITS[panel].def;
  return clampPanelWidth(panel, raw);
}
