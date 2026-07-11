export const UI_SCALE_LEVELS = [80, 90, 100, 110, 120, 130] as const;
export type UiScaleDirection = "reset" | "in" | "out";

export function normalizeUiScale(value: number): number {
  const bounded = Math.min(UI_SCALE_LEVELS.at(-1)!, Math.max(UI_SCALE_LEVELS[0], value));
  return UI_SCALE_LEVELS.reduce((closest, level) =>
    Math.abs(level - bounded) < Math.abs(closest - bounded) ? level : closest,
  );
}

export function nextUiScale(current: number, direction: UiScaleDirection): number {
  if (direction === "reset") return 100;
  const normalized = normalizeUiScale(current);
  const index = UI_SCALE_LEVELS.indexOf(normalized as (typeof UI_SCALE_LEVELS)[number]);
  const delta = direction === "in" ? 1 : -1;
  const next = Math.min(UI_SCALE_LEVELS.length - 1, Math.max(0, index + delta));
  return UI_SCALE_LEVELS[next]!;
}

/** Scale the entire webview, including fixed overlays and teleported menus. */
export async function applyUiScale(value: number): Promise<number> {
  const percent = normalizeUiScale(value);
  document.documentElement.dataset.uiScale = String(percent);
  if ("__TAURI_INTERNALS__" in window) {
    const { getCurrentWebview } = await import("@tauri-apps/api/webview");
    await getCurrentWebview().setZoom(percent / 100);
  } else {
    // Browser preview fallback. Tauri uses native webview zoom above.
    document.documentElement.style.zoom = String(percent / 100);
  }
  return percent;
}
