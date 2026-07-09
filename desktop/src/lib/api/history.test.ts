import { describe, expect, it } from "vitest";
import { groupByDay, type HistoryEntry } from "./history";

const at = (iso: string): number => new Date(iso).getTime() / 1000;
const entry = (prompt: string, iso: string): HistoryEntry => ({
  prompt,
  model: "flux2-klein:q4",
  used_at: at(iso),
});

describe("groupByDay", () => {
  const now = new Date("2026-07-08T20:00:00");

  it("labels today and yesterday, then dates", () => {
    const groups = groupByDay(
      [
        entry("a", "2026-07-08T14:00:00"),
        entry("b", "2026-07-07T22:00:00"),
        entry("c", "2026-07-01T10:00:00"),
      ],
      now,
    );
    expect(groups.map((g) => g.label)).toEqual(["Today", "Yesterday", "July 1"]);
  });

  it("keeps consecutive same-day entries in one group", () => {
    const groups = groupByDay(
      [entry("a", "2026-07-08T14:00:00"), entry("b", "2026-07-08T09:00:00")],
      now,
    );
    expect(groups).toHaveLength(1);
    expect(groups[0]!.entries.map((e) => e.prompt)).toEqual(["a", "b"]);
  });

  it("handles empty input", () => {
    expect(groupByDay([], now)).toEqual([]);
  });
});
