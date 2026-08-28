import { describe, expect, it } from "vitest";
import { chunkForProbe, planPrewarm, type PrewarmCandidate } from "./thumbnailPrewarm";

function candidate(row: number, host = "plato", index = row): PrewarmCandidate {
  return { sourceKey: host, filename: `${host}-${index}.png`, mediaVersion: "1:1", rowIndex: row };
}

describe("planPrewarm", () => {
  const viewport = { startRow: 10, endRow: 13, rowsPerViewport: 4 };

  it("skips on-screen rows and orders near before background, below before above", () => {
    const candidates = [
      candidate(0),
      candidate(5),
      candidate(9),
      candidate(11), // visible
      candidate(14),
      candidate(20),
      candidate(40),
    ];
    const plan = planPrewarm(candidates, viewport);
    // Distance first (14 and 9 are both one row away; below wins the tie),
    // then 5 (5 rows), 20 (7), 0 (10), 40 (27).
    expect(plan.map((p) => p.candidate.rowIndex)).toEqual([14, 9, 5, 20, 0, 40]);
    // near = within 2 viewports (8 rows) of the visible span.
    expect(plan.map((p) => p.priority)).toEqual([
      "near",
      "near",
      "near",
      "near",
      "background",
      "background",
    ]);
    expect(plan.some((p) => p.candidate.rowIndex === 11)).toBe(false);
  });

  it("caps the work per host, keeping the nearest tiles", () => {
    const candidates: PrewarmCandidate[] = [];
    for (let i = 0; i < 50; i++) candidates.push(candidate(20 + i, "plato", i));
    for (let i = 0; i < 5; i++) candidates.push(candidate(20 + i, "hal", i));
    const plan = planPrewarm(candidates, viewport, { maxPerHost: 3 });
    const plato = plan.filter((p) => p.candidate.sourceKey === "plato");
    const hal = plan.filter((p) => p.candidate.sourceKey === "hal");
    expect(plato.map((p) => p.candidate.rowIndex)).toEqual([20, 21, 22]);
    expect(hal).toHaveLength(3);
  });

  it("chunks probes in order", () => {
    expect(chunkForProbe([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
    expect(chunkForProbe([], 2)).toEqual([]);
  });
});
