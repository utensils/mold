/** Fallback shared by older video-model responses that omit an FPS value. */
export const FALLBACK_VIDEO_FPS = 24;

/**
 * Compact clip duration for filmstrip and sequence controls. Frames remain
 * first because 8n+1 is a model constraint; seconds provide editor-friendly
 * context.
 */
export function formatFrameDuration(frames: number, fps: number): string {
  const safeFrames = Math.max(
    0,
    Math.round(Number.isFinite(frames) ? frames : 0),
  );
  const safeFps = Number.isFinite(fps) && fps > 0 ? fps : FALLBACK_VIDEO_FPS;
  return `${safeFrames}f · ${(safeFrames / safeFps).toFixed(1)}s`;
}

/**
 * What a clip in a sequence is called, by position. One answer, so the rail's
 * block title and the seam editor's "join · A → B" sentence cannot disagree.
 */
export function clipLabel(index: number): string {
  return index === 0 ? "Opening clip" : `Clip ${index + 1}`;
}
