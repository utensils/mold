import { describe, expect, it } from "vitest";
import { layoutJustifiedRows } from "./layout";
import type { GalleryImage } from "../api/types";

const img = (w: number, h: number, name: string): GalleryImage => ({
  filename: name,
  timestamp: 0,
  metadata: {
    prompt: "",
    model: "m",
    seed: 1,
    steps: 1,
    guidance: 0,
    width: w,
    height: h,
  },
});

describe("layoutJustifiedRows", () => {
  it("fills full rows edge to edge", () => {
    const images = Array.from({ length: 6 }, (_, i) => img(1024, 1024, `${i}.png`));
    const rows = layoutJustifiedRows(images, 800, 180, 8);
    expect(rows.length).toBeGreaterThan(1);
    const first = rows[0]!;
    const rowWidth =
      first.items.reduce((sum, it) => sum + it.width, 0) + 8 * (first.items.length - 1);
    expect(Math.abs(rowWidth - 800)).toBeLessThan(1);
  });

  it("keeps the last partial row at target height", () => {
    const rows = layoutJustifiedRows([img(1024, 1024, "a.png")], 800, 180, 8);
    expect(rows).toHaveLength(1);
    expect(rows[0]!.height).toBe(180);
  });

  it("preserves aspect ratios", () => {
    const rows = layoutJustifiedRows([img(1920, 1080, "wide.png")], 4000, 180, 8);
    const it0 = rows[0]!.items[0]!;
    expect(it0.width / it0.height).toBeCloseTo(1920 / 1080, 3);
  });

  it("caps runaway row heights from a single ultra-narrow row", () => {
    const rows = layoutJustifiedRows(
      [img(500, 2000, "tall.png"), img(500, 2000, "tall2.png")],
      2000,
      180,
      8,
    );
    for (const row of rows) expect(row.height).toBeLessThanOrEqual(270);
  });

  it("handles zero-width containers and metadata without dimensions", () => {
    expect(layoutJustifiedRows([img(1024, 1024, "a.png")], 0)).toEqual([]);
    const rows = layoutJustifiedRows([img(0, 0, "no-dims.png")], 800);
    expect(rows[0]!.items[0]!.width).toBe(180);
  });
});
