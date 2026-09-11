/**
 * Touch geometry for the full-screen still-image viewer.
 *
 * The state is deliberately independent of Vue and the DOM. Both native
 * shells render the same WebView surface, and keeping the gesture math pure
 * lets iOS and Android exercise the exact same focal zoom and pan contract.
 */
export const VIEWER_IMAGE_ZOOM_MIN = 1;
export const VIEWER_IMAGE_ZOOM_MAX = 5;

export interface ViewerImageZoomPoint {
  pointerId: number;
  clientX: number;
  clientY: number;
}

export interface ViewerImageZoomMetrics {
  left: number;
  top: number;
  viewportWidth: number;
  viewportHeight: number;
  /** The contained image's rendered dimensions at 1x. */
  mediaWidth: number;
  mediaHeight: number;
}

export interface ViewerImageZoomState {
  points: Map<number, { x: number; y: number }>;
  scale: number;
  x: number;
  y: number;
  baselineScale: number;
  baselineX: number;
  baselineY: number;
  baselineDistance: number;
  baselineCenterX: number;
  baselineCenterY: number;
  singleStartX: number;
  singleStartY: number;
  /** Once two fingers land, no remaining pointer may become a gallery swipe. */
  ownsSequence: boolean;
}

export interface ViewerImageZoomResult {
  tracked: boolean;
  consumed: boolean;
  changed: boolean;
}

export function createViewerImageZoom(): ViewerImageZoomState {
  return {
    points: new Map(),
    scale: VIEWER_IMAGE_ZOOM_MIN,
    x: 0,
    y: 0,
    baselineScale: VIEWER_IMAGE_ZOOM_MIN,
    baselineX: 0,
    baselineY: 0,
    baselineDistance: 0,
    baselineCenterX: 0,
    baselineCenterY: 0,
    singleStartX: 0,
    singleStartY: 0,
    ownsSequence: false,
  };
}

export function viewerImageIsZoomed(state: ViewerImageZoomState): boolean {
  return state.scale > VIEWER_IMAGE_ZOOM_MIN + 0.001;
}

function pair(state: ViewerImageZoomState) {
  const [first, second] = [...state.points.values()];
  if (!first || !second) return null;
  const centerX = (first.x + second.x) / 2;
  const centerY = (first.y + second.y) / 2;
  return {
    centerX,
    centerY,
    distance: Math.hypot(second.x - first.x, second.y - first.y),
  };
}

function rebaseline(state: ViewerImageZoomState): void {
  state.baselineScale = state.scale;
  state.baselineX = state.x;
  state.baselineY = state.y;
  const currentPair = pair(state);
  state.baselineDistance = currentPair?.distance ?? 0;
  state.baselineCenterX = currentPair?.centerX ?? 0;
  state.baselineCenterY = currentPair?.centerY ?? 0;
  const single = state.points.size === 1 ? [...state.points.values()][0] : null;
  state.singleStartX = single?.x ?? 0;
  state.singleStartY = single?.y ?? 0;
}

function clampTranslation(state: ViewerImageZoomState, metrics: ViewerImageZoomMetrics): void {
  const maxX = Math.max(0, (metrics.mediaWidth * state.scale - metrics.viewportWidth) / 2);
  const maxY = Math.max(0, (metrics.mediaHeight * state.scale - metrics.viewportHeight) / 2);
  state.x = maxX === 0 ? 0 : Math.max(-maxX, Math.min(maxX, state.x));
  state.y = maxY === 0 ? 0 : Math.max(-maxY, Math.min(maxY, state.y));
}

/** Reconcile a retained zoom after rotation or another viewport resize. */
export function constrainViewerImageZoom(
  state: ViewerImageZoomState,
  metrics: ViewerImageZoomMetrics,
): boolean {
  const priorX = state.x;
  const priorY = state.y;
  clampTranslation(state, metrics);
  return priorX !== state.x || priorY !== state.y;
}

export function beginViewerImageZoom(
  state: ViewerImageZoomState,
  point: ViewerImageZoomPoint,
): ViewerImageZoomResult {
  if (state.points.size >= 2 || state.points.has(point.pointerId)) {
    return { tracked: false, consumed: state.ownsSequence, changed: false };
  }
  state.points.set(point.pointerId, { x: point.clientX, y: point.clientY });
  if (state.points.size === 2 || viewerImageIsZoomed(state)) state.ownsSequence = true;
  rebaseline(state);
  return {
    tracked: true,
    consumed: state.ownsSequence,
    changed: false,
  };
}

export function moveViewerImageZoom(
  state: ViewerImageZoomState,
  point: ViewerImageZoomPoint,
  metrics: ViewerImageZoomMetrics,
): ViewerImageZoomResult {
  const tracked = state.points.get(point.pointerId);
  if (!tracked) return { tracked: false, consumed: false, changed: false };
  tracked.x = point.clientX;
  tracked.y = point.clientY;

  if (state.points.size === 2) {
    state.ownsSequence = true;
    const current = pair(state);
    if (!current || current.distance <= 0) {
      rebaseline(state);
      return { tracked: true, consumed: true, changed: false };
    }
    if (state.baselineDistance <= 0) {
      rebaseline(state);
      return { tracked: true, consumed: true, changed: false };
    }

    const nextScale = Math.max(
      VIEWER_IMAGE_ZOOM_MIN,
      Math.min(
        VIEWER_IMAGE_ZOOM_MAX,
        state.baselineScale * (current.distance / state.baselineDistance),
      ),
    );
    const viewportCenterX = metrics.left + metrics.viewportWidth / 2;
    const viewportCenterY = metrics.top + metrics.viewportHeight / 2;
    // The pixel under the original midpoint stays under the moving midpoint.
    const contentX =
      (state.baselineCenterX - viewportCenterX - state.baselineX) / state.baselineScale;
    const contentY =
      (state.baselineCenterY - viewportCenterY - state.baselineY) / state.baselineScale;
    state.scale = nextScale;
    state.x = current.centerX - viewportCenterX - contentX * nextScale;
    state.y = current.centerY - viewportCenterY - contentY * nextScale;
    clampTranslation(state, metrics);
    if (!viewerImageIsZoomed(state)) {
      state.scale = VIEWER_IMAGE_ZOOM_MIN;
      state.x = 0;
      state.y = 0;
    }
    return { tracked: true, consumed: true, changed: true };
  }

  if (viewerImageIsZoomed(state)) {
    state.ownsSequence = true;
    state.x = state.baselineX + point.clientX - state.singleStartX;
    state.y = state.baselineY + point.clientY - state.singleStartY;
    clampTranslation(state, metrics);
    return { tracked: true, consumed: true, changed: true };
  }
  return { tracked: true, consumed: state.ownsSequence, changed: false };
}

export function endViewerImageZoom(
  state: ViewerImageZoomState,
  pointerId: number,
): ViewerImageZoomResult {
  if (!state.points.delete(pointerId)) {
    return { tracked: false, consumed: false, changed: false };
  }
  const consumed = state.ownsSequence || viewerImageIsZoomed(state);
  rebaseline(state);
  if (state.points.size === 0) state.ownsSequence = false;
  return { tracked: true, consumed, changed: false };
}

export function resetViewerImageZoom(state: ViewerImageZoomState): void {
  state.points.clear();
  state.scale = VIEWER_IMAGE_ZOOM_MIN;
  state.x = 0;
  state.y = 0;
  state.ownsSequence = false;
  rebaseline(state);
}
