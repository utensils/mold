import { beforeEach, describe, expect, it } from "vitest";
import {
  VIEWER_IMAGE_ZOOM_MAX,
  beginViewerImageZoom,
  constrainViewerImageZoom,
  createViewerImageZoom,
  endViewerImageZoom,
  moveViewerImageZoom,
  resetViewerImageZoom,
  viewerImageIsZoomed,
  type ViewerImageZoomMetrics,
} from "./imageViewerZoom";

const square: ViewerImageZoomMetrics = {
  left: 0,
  top: 0,
  viewportWidth: 400,
  viewportHeight: 400,
  mediaWidth: 400,
  mediaHeight: 400,
};

describe("full-screen image pinch zoom", () => {
  let state: ReturnType<typeof createViewerImageZoom>;

  beforeEach(() => {
    state = createViewerImageZoom();
  });

  function down(pointerId: number, x: number, y: number) {
    return beginViewerImageZoom(state, { pointerId, clientX: x, clientY: y });
  }

  function move(pointerId: number, x: number, y: number) {
    return moveViewerImageZoom(state, { pointerId, clientX: x, clientY: y }, square);
  }

  it("leaves one-finger movement at 1x available to gallery navigation", () => {
    expect(down(1, 300, 200).consumed).toBe(false);
    expect(move(1, 180, 205).consumed).toBe(false);
    expect(endViewerImageZoom(state, 1).consumed).toBe(false);
    expect(state.scale).toBe(1);
  });

  it("zooms around the fingers instead of jumping toward the viewport center", () => {
    down(1, 250, 200);
    down(2, 350, 200);

    move(2, 450, 200);

    expect(state.scale).toBeCloseTo(2);
    // The original midpoint was 100px right of center. At 2x that pixel sits
    // 200px right of center, so keeping it under the new 150px offset requires
    // a 50px leftward translation.
    expect(state.x).toBeCloseTo(-50);
    expect(state.y).toBeCloseTo(0);
    expect(viewerImageIsZoomed(state)).toBe(true);
  });

  it("pans a zoomed image and clamps blank space away", () => {
    down(1, 150, 200);
    down(2, 250, 200);
    move(2, 350, 200); // 2x
    endViewerImageZoom(state, 2);

    move(1, 1000, 900);

    expect(state.x).toBe(200);
    expect(state.y).toBe(200);
    expect(endViewerImageZoom(state, 1).consumed).toBe(true);
  });

  it("locks an axis until a contained portrait image grows past that viewport edge", () => {
    const portrait = { ...square, mediaWidth: 200, mediaHeight: 400 };
    down(1, 150, 200);
    down(2, 250, 200);
    moveViewerImageZoom(state, { pointerId: 2, clientX: 350, clientY: 200 }, portrait);
    expect(state.scale).toBeCloseTo(2);
    expect(state.x).toBe(0);

    endViewerImageZoom(state, 2);
    moveViewerImageZoom(state, { pointerId: 1, clientX: 250, clientY: 260 }, portrait);
    expect(state.x).toBe(0);
    expect(state.y).toBe(60);
  });

  it("caps magnification and keeps the whole pinch sequence out of paging", () => {
    down(1, 190, 200);
    down(2, 210, 200);
    expect(move(2, 2000, 200).consumed).toBe(true);
    expect(state.scale).toBe(VIEWER_IMAGE_ZOOM_MAX);

    expect(endViewerImageZoom(state, 2).consumed).toBe(true);
    expect(endViewerImageZoom(state, 1).consumed).toBe(true);
  });

  it("returns exactly to home when the pinch closes back to 1x", () => {
    down(1, 100, 200);
    down(2, 300, 200);
    move(2, 380, 240);
    expect(state.scale).toBeGreaterThan(1);

    move(2, 300, 200);

    expect(state.scale).toBe(1);
    expect(state.x).toBe(0);
    expect(state.y).toBe(0);
    expect(endViewerImageZoom(state, 2).consumed).toBe(true);
    expect(endViewerImageZoom(state, 1).consumed).toBe(true);
  });

  it("re-clamps a retained pan when rotation changes the fitted image bounds", () => {
    down(1, 150, 200);
    down(2, 250, 200);
    move(2, 350, 200);
    endViewerImageZoom(state, 2);
    move(1, 40, 200);
    expect(state.x).not.toBe(0);

    const landscape = { ...square, viewportWidth: 800, mediaWidth: 400 };
    expect(constrainViewerImageZoom(state, landscape)).toBe(true);
    expect(state.x).toBe(0);
    expect(state.scale).toBeCloseTo(2);
  });

  it("resets scale, translation, pointers, and gesture ownership for a new print", () => {
    down(1, 150, 200);
    down(2, 250, 200);
    move(2, 350, 200);

    resetViewerImageZoom(state);

    expect(state).toMatchObject({ scale: 1, x: 0, y: 0, ownsSequence: false });
    expect(state.points.size).toBe(0);
    expect(down(3, 300, 200).consumed).toBe(false);
  });
});
