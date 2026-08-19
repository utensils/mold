/**
 * iPhone Library thumbnail sizing. Web and desktop expose the shared
 * Lightroom-style slider from `@studio/lib/galleryThumbnailSize`, which
 * persists a pixel target; a phone is too narrow for that unit to mean
 * anything (its 220px default would be one-and-a-bit columns), and a slider is
 * not the native gesture. The iPhone parity control is therefore a pinch over
 * the grid that moves a discrete column count, exactly as Photos.app does, and
 * it persists under its own key so a phone's zoom never rewrites a Mac's grid.
 */
export const MOBILE_GALLERY_COLUMNS_STORAGE_KEY = "mold.mobile.galleryColumns.v1";
export const MOBILE_GALLERY_COLUMNS_MIN = 2;
export const MOBILE_GALLERY_COLUMNS_MAX = 5;
/** The three-across grid every build before this one rendered. */
export const MOBILE_GALLERY_COLUMNS_DEFAULT = 3;

/**
 * How far past a step's midpoint the gesture must travel before the grid
 * relayouts. Without it a finger resting on a boundary flickers the whole
 * Library between two column counts.
 */
const PINCH_HYSTERESIS_COLUMNS = 0.15;

export interface MobileGalleryColumnStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function browserStorage(): MobileGalleryColumnStorage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export function normalizeMobileGalleryColumns(value: number): number {
  if (!Number.isFinite(value)) return MOBILE_GALLERY_COLUMNS_DEFAULT;
  return Math.min(
    MOBILE_GALLERY_COLUMNS_MAX,
    Math.max(MOBILE_GALLERY_COLUMNS_MIN, Math.round(value)),
  );
}

export function loadMobileGalleryColumns(
  storage: MobileGalleryColumnStorage | null = browserStorage(),
): number {
  if (!storage) return MOBILE_GALLERY_COLUMNS_DEFAULT;
  try {
    const saved = storage.getItem(MOBILE_GALLERY_COLUMNS_STORAGE_KEY);
    if (saved === null || saved.trim() === "") return MOBILE_GALLERY_COLUMNS_DEFAULT;
    const parsed = Number(saved);
    return Number.isFinite(parsed)
      ? normalizeMobileGalleryColumns(parsed)
      : MOBILE_GALLERY_COLUMNS_DEFAULT;
  } catch {
    return MOBILE_GALLERY_COLUMNS_DEFAULT;
  }
}

export function saveMobileGalleryColumns(
  value: number,
  storage: MobileGalleryColumnStorage | null = browserStorage(),
): void {
  if (!storage) return;
  try {
    storage.setItem(
      MOBILE_GALLERY_COLUMNS_STORAGE_KEY,
      String(normalizeMobileGalleryColumns(value)),
    );
  } catch {
    // WebKit storage can be unavailable; the pinch still works for this visit.
  }
}

export interface PinchPoint {
  pointerId: number;
  clientX: number;
  clientY: number;
}

export interface PinchZoomState {
  /** The at-most-two fingers that own the gesture, in the order they landed. */
  points: Map<number, { x: number; y: number }>;
  /** Separation when the current pair landed, or 0 while fewer than two are down. */
  baselineDistance: number;
  /** Column count the current pair started from. */
  baselineColumns: number;
  /** Column count the grid is rendering right now. */
  columns: number;
}

export function createPinchZoom(columns: number): PinchZoomState {
  const start = normalizeMobileGalleryColumns(columns);
  return { points: new Map(), baselineDistance: 0, baselineColumns: start, columns: start };
}

/** True once two fingers are down, which is when the grid may resize. */
export function isPinching(state: PinchZoomState): boolean {
  return state.points.size === 2 && state.baselineDistance > 0;
}

function separation(state: PinchZoomState): number {
  const [first, second] = [...state.points.values()];
  if (!first || !second) return 0;
  return Math.hypot(second.x - first.x, second.y - first.y);
}

/** Re-measure from wherever the fingers are now, keeping the rendered size. */
function rebaseline(state: PinchZoomState): void {
  state.baselineDistance = state.points.size === 2 ? separation(state) : 0;
  state.baselineColumns = state.columns;
}

export function pinchPointerDown(state: PinchZoomState, point: PinchPoint): void {
  // A third finger never joins: the pair that started the gesture keeps it, so
  // an accidental palm touch cannot jerk the grid mid-pinch.
  if (state.points.size >= 2) return;
  state.points.set(point.pointerId, { x: point.clientX, y: point.clientY });
  rebaseline(state);
}

/**
 * Feed a moved finger in. Returns the new column count when the grid should
 * relayout, or null when nothing changed.
 *
 * Tile width is container width over column count, so a pinch that scales the
 * fingers' separation by `r` should scale tile width by `r` too — which makes
 * the column count inversely proportional to `r`.
 */
export function pinchPointerMove(state: PinchZoomState, point: PinchPoint): number | null {
  const tracked = state.points.get(point.pointerId);
  if (!tracked) return null;
  tracked.x = point.clientX;
  tracked.y = point.clientY;
  if (!isPinching(state)) return null;

  const distance = separation(state);
  if (distance <= 0) return null;

  const target = state.baselineColumns / (distance / state.baselineDistance);
  const next = normalizeMobileGalleryColumns(target);
  if (next === state.columns) return null;
  // Clearing the midpoint alone is not enough; the extra margin is what stops a
  // resting finger from oscillating between two layouts.
  if (Math.abs(target - state.columns) < 0.5 + PINCH_HYSTERESIS_COLUMNS) return null;

  state.columns = next;
  return next;
}

export function pinchPointerUp(state: PinchZoomState, pointerId: number): void {
  if (!state.points.delete(pointerId)) return;
  rebaseline(state);
}

/** Drop the whole gesture, e.g. when the Library unmounts or a mode changes. */
export function resetPinch(state: PinchZoomState, columns: number = state.columns): void {
  state.points.clear();
  state.baselineDistance = 0;
  state.columns = normalizeMobileGalleryColumns(columns);
  state.baselineColumns = state.columns;
}
