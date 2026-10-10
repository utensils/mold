import { describe, expect, it } from "vitest";
import {
  aspectOf,
  layoutJustifiedRows,
  justifiedWindow,
} from "./justifiedLayout";

const image = (width: number, height = 100) => ({
  metadata: { width, height },
});
describe("continuous library rows", () => {
  it("fills completed rows, preserves order and real aspect ratios", () => {
    const items = [50, 150, 100, 200, 60, 100, 160, 70].map((w) => image(w));
    const rows = layoutJustifiedRows(items, 600, 180);
    expect(rows.flatMap((r) => r.items.map((t) => t.item))).toEqual(items);
    for (const [i, row] of rows.entries()) {
      const width =
        row.items.reduce((s, t) => s + t.width, 0) + 2 * (row.items.length - 1);
      if (i < rows.length - 1) expect(width).toBeCloseTo(600);
      expect(width).toBeLessThanOrEqual(600.00001);
      for (const tile of row.items)
        expect(tile.width / tile.height).toBeCloseTo(aspectOf(tile.item));
    }
  });
  it("leaves widows at target height and fits panoramas without cropping", () => {
    expect(layoutJustifiedRows([image(50)], 600, 180)[0]!.height).toBe(180);
    const row = layoutJustifiedRows([image(4000)], 600, 180)[0]!;
    expect(row.height).toBe(15);
    expect(row.items[0]!.width).toBe(600);
  });
  it("falls back only for invalid dimensions and refuses invalid geometry", () => {
    for (const w of [0, -1, NaN, Infinity]) expect(aspectOf(image(w))).toBe(1);
    expect(aspectOf(image(1, 10000))).toBe(0.0001);
    expect(layoutJustifiedRows([image(100)], 0)).toEqual([]);
  });
  it("closes a row before gaps consume all space for very thin portraits", () => {
    const items = Array.from({ length: 100 }, () => image(1, 10000));
    for (const width of [2, 100, 600]) {
      const rows = layoutJustifiedRows(items, width, 180);
      expect(rows.flatMap((row) => row.items).length).toBe(items.length);
      for (const row of rows) {
        expect(row.height).toBeGreaterThan(0);
        expect(row.height).toBeLessThanOrEqual(270);
        for (const tile of row.items)
          expect(tile.width / tile.height).toBeCloseTo(0.0001, 8);
      }
    }
  });
  it("windows a 30,000-item library in both directions with correct offsets", () => {
    const rows = layoutJustifiedRows(
      Array.from({ length: 30000 }, (_, i) => image([50, 100, 180][i % 3]!)),
      1000,
      180,
    );
    for (const top of [0, 100000, 400, 900000]) {
      const band = justifiedWindow(rows, top, 800, 2);
      expect(band.end - band.start).toBeLessThan(20);
      expect(band.start).toBeGreaterThanOrEqual(0);
      expect(band.end).toBeLessThanOrEqual(rows.length);
    }
  });
});
