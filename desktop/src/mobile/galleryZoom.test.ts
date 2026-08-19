import { beforeEach, describe, expect, it } from "vitest";
import {
  MOBILE_GALLERY_COLUMNS_DEFAULT,
  MOBILE_GALLERY_COLUMNS_MAX,
  MOBILE_GALLERY_COLUMNS_MIN,
  MOBILE_GALLERY_COLUMNS_STORAGE_KEY,
  createPinchZoom,
  isPinching,
  loadMobileGalleryColumns,
  normalizeMobileGalleryColumns,
  pinchPointerDown,
  pinchPointerMove,
  pinchPointerUp,
  resetPinch,
  saveMobileGalleryColumns,
  tracksPointer,
} from "./galleryZoom";

interface FakeStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function memoryStorage(seed: Record<string, string> = {}): FakeStorage {
  const map = new Map(Object.entries(seed));
  return {
    getItem: (key) => map.get(key) ?? null,
    setItem: (key, value) => void map.set(key, value),
  };
}

describe("mobile gallery column persistence", () => {
  it("defaults to today's three-across grid", () => {
    expect(MOBILE_GALLERY_COLUMNS_DEFAULT).toBe(3);
    expect(loadMobileGalleryColumns(memoryStorage())).toBe(3);
    expect(loadMobileGalleryColumns(null)).toBe(3);
  });

  it("clamps and rounds every stored or restored value into the supported range", () => {
    expect(normalizeMobileGalleryColumns(0)).toBe(MOBILE_GALLERY_COLUMNS_MIN);
    expect(normalizeMobileGalleryColumns(99)).toBe(MOBILE_GALLERY_COLUMNS_MAX);
    expect(normalizeMobileGalleryColumns(3.4)).toBe(3);
    expect(normalizeMobileGalleryColumns(Number.NaN)).toBe(MOBILE_GALLERY_COLUMNS_DEFAULT);
  });

  it("restores a saved choice and ignores a corrupt one", () => {
    expect(
      loadMobileGalleryColumns(memoryStorage({ [MOBILE_GALLERY_COLUMNS_STORAGE_KEY]: "5" })),
    ).toBe(5);
    expect(
      loadMobileGalleryColumns(memoryStorage({ [MOBILE_GALLERY_COLUMNS_STORAGE_KEY]: "banana" })),
    ).toBe(MOBILE_GALLERY_COLUMNS_DEFAULT);
    expect(
      loadMobileGalleryColumns(memoryStorage({ [MOBILE_GALLERY_COLUMNS_STORAGE_KEY]: "" })),
    ).toBe(MOBILE_GALLERY_COLUMNS_DEFAULT);
  });

  it("persists the normalized value and survives unavailable storage", () => {
    const storage = memoryStorage();
    saveMobileGalleryColumns(42, storage);
    expect(storage.getItem(MOBILE_GALLERY_COLUMNS_STORAGE_KEY)).toBe(
      String(MOBILE_GALLERY_COLUMNS_MAX),
    );
    expect(() => saveMobileGalleryColumns(3, null)).not.toThrow();
  });
});

describe("mobile gallery pinch gesture", () => {
  let state: ReturnType<typeof createPinchZoom>;

  beforeEach(() => {
    state = createPinchZoom(3);
  });

  function startPinch(distance: number): void {
    pinchPointerDown(state, { pointerId: 1, clientX: 0, clientY: 0 });
    pinchPointerDown(state, { pointerId: 2, clientX: distance, clientY: 0 });
  }

  function pinchTo(distance: number): number | null {
    return pinchPointerMove(state, { pointerId: 2, clientX: distance, clientY: 0 });
  }

  it("stays idle until a second finger lands", () => {
    pinchPointerDown(state, { pointerId: 1, clientX: 0, clientY: 0 });
    expect(isPinching(state)).toBe(false);
    expect(pinchPointerMove(state, { pointerId: 1, clientX: 400, clientY: 0 })).toBeNull();

    pinchPointerDown(state, { pointerId: 2, clientX: 100, clientY: 0 });
    expect(isPinching(state)).toBe(true);
  });

  it("spreading fingers apart grows the thumbnails by dropping a column", () => {
    startPinch(200);
    expect(pinchTo(240)).toBeNull();
    expect(pinchTo(340)).toBe(2);
  });

  it("pinching fingers together shrinks the thumbnails by adding columns", () => {
    startPinch(300);
    expect(pinchTo(280)).toBeNull();
    expect(pinchTo(210)).toBe(4);
    expect(pinchTo(165)).toBe(5);
  });

  it("never passes the supported range no matter how far the pinch travels", () => {
    startPinch(200);
    expect(pinchTo(20)).toBe(MOBILE_GALLERY_COLUMNS_MAX);
    expect(pinchTo(10)).toBeNull();

    state = createPinchZoom(3);
    startPinch(100);
    expect(pinchTo(900)).toBe(MOBILE_GALLERY_COLUMNS_MIN);
    expect(pinchTo(2000)).toBeNull();
  });

  it("reversing the pinch returns to the starting size", () => {
    startPinch(200);
    expect(pinchTo(340)).toBe(2);
    expect(pinchTo(200)).toBe(3);
  });

  it("holds a step against wobble instead of flickering across the boundary", () => {
    startPinch(200);
    // 240px is the 2.5-column midpoint; hysteresis must hold the grid at three.
    expect(pinchTo(240)).toBeNull();
    expect(pinchTo(238)).toBeNull();
    expect(pinchTo(242)).toBeNull();
  });

  it("lifting a finger ends the gesture and re-baselines the next one", () => {
    startPinch(200);
    expect(pinchTo(340)).toBe(2);
    pinchPointerUp(state, 2);
    expect(isPinching(state)).toBe(false);
    expect(pinchTo(600)).toBeNull();

    // A fresh pinch measures from the committed two-column layout: the same
    // 0.7 squeeze that reached four columns from three only reaches three here.
    pinchPointerDown(state, { pointerId: 3, clientX: 200, clientY: 0 });
    expect(isPinching(state)).toBe(true);
    expect(pinchPointerMove(state, { pointerId: 3, clientX: 140, clientY: 0 })).toBe(3);
  });

  it("measures the finger separation in both axes", () => {
    pinchPointerDown(state, { pointerId: 1, clientX: 0, clientY: 0 });
    pinchPointerDown(state, { pointerId: 2, clientX: 120, clientY: 160 }); // 200px apart
    expect(pinchPointerMove(state, { pointerId: 2, clientX: 204, clientY: 272 })).toBe(2); // 340px
  });

  it("ignores a third finger so the original pair keeps owning the gesture", () => {
    startPinch(200);
    pinchPointerDown(state, { pointerId: 9, clientX: 1000, clientY: 1000 });
    expect(pinchPointerMove(state, { pointerId: 9, clientX: 2000, clientY: 2000 })).toBeNull();
    expect(pinchTo(340)).toBe(2);
  });

  it("cancelling drops every finger so a stale pair cannot resume", () => {
    startPinch(200);
    pinchPointerUp(state, 1);
    pinchPointerUp(state, 2);
    expect(isPinching(state)).toBe(false);
    expect(pinchTo(340)).toBeNull();
  });

  it("resetPinch drops a gesture stranded without its pointerup", () => {
    startPinch(200);
    expect(pinchTo(340)).toBe(2);

    resetPinch(state, 4);

    expect(isPinching(state)).toBe(false);
    expect(tracksPointer(state, 1)).toBe(false);
    expect(state.columns).toBe(4);
    // The stranded finger is gone, so one new finger cannot form a pair.
    pinchPointerDown(state, { pointerId: 5, clientX: 0, clientY: 0 });
    expect(isPinching(state)).toBe(false);
    expect(pinchPointerMove(state, { pointerId: 5, clientX: 900, clientY: 0 })).toBeNull();
  });

  it("resetPinch keeps the rendered size when no replacement is named", () => {
    startPinch(200);
    expect(pinchTo(340)).toBe(2);
    resetPinch(state);
    expect(state.columns).toBe(2);
  });

  it("reports which fingers it is tracking", () => {
    expect(tracksPointer(state, 1)).toBe(false);
    startPinch(200);
    expect(tracksPointer(state, 1)).toBe(true);
    expect(tracksPointer(state, 2)).toBe(true);
    expect(tracksPointer(state, 3)).toBe(false);
    pinchPointerUp(state, 2);
    expect(tracksPointer(state, 2)).toBe(false);
  });

  it("holds a step against wobble in the shrink direction too", () => {
    startPinch(300);
    expect(pinchTo(210)).toBe(4);
    // 3.5 columns is the midpoint below the committed four; hysteresis holds.
    expect(pinchTo(257)).toBeNull();
    expect(pinchTo(255)).toBeNull();
  });

  it("re-measures at a limit so reversing an overshoot responds at once", () => {
    startPinch(100);
    expect(pinchTo(900)).toBe(MOBILE_GALLERY_COLUMNS_MIN);

    // Without re-baselining, undoing the 9x overshoot would move nothing until
    // the fingers travelled most of the way back.
    expect(pinchTo(500)).toBe(4);
  });

  it("adopts a scale when both fingers land on the same point", () => {
    pinchPointerDown(state, { pointerId: 1, clientX: 40, clientY: 40 });
    pinchPointerDown(state, { pointerId: 2, clientX: 40, clientY: 40 });
    expect(isPinching(state)).toBe(false);

    // The first move that separates them becomes the baseline, not a dead end.
    expect(pinchPointerMove(state, { pointerId: 2, clientX: 240, clientY: 40 })).toBeNull();
    expect(isPinching(state)).toBe(true);
    expect(pinchPointerMove(state, { pointerId: 2, clientX: 380, clientY: 40 })).toBe(2);
  });
});
