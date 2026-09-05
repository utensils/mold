import { describe, expect, it } from "vitest";
import { rulerTickInterval, rulerTicks } from "./rulerTicks";

const clock = (seconds: number) =>
  `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, "0")}`;

describe("ruler ticks", () => {
  it("marks a round clip out in a handful of steps", () => {
    expect(rulerTickInterval(20)).toBe(5);
    expect(rulerTicks(20, clock).map((tick) => tick.at)).toEqual([0, 5, 10, 15, 20]);
  });

  it("pins the closing mark to the right edge instead of left:100%", () => {
    const marks = rulerTicks(20, clock);
    const last = marks.at(-1)!;
    expect(last.atEnd).toBe(true);
    // left:100% starts the label AT the edge and paints it past — the bench's
    // overflow-hidden then cuts the closing caption off entirely.
    expect(last.style).toEqual({ right: "0" });
    expect(last.style.left).toBeUndefined();
  });

  it("leaves every other mark on the left, in percent", () => {
    const marks = rulerTicks(20, clock);
    expect(marks.slice(0, -1).every((tick) => tick.atEnd === false)).toBe(true);
    expect(marks[1]!.style).toEqual({ left: "25%" });
  });

  it("does not pin a mark that stops well short of the end", () => {
    // 23s at a 5s step ends on 20s — 87% of the way, a genuinely interior mark.
    const marks = rulerTicks(23, clock);
    expect(marks.at(-1)!.at).toBe(20);
    expect(marks.at(-1)!.atEnd).toBe(false);
  });

  it("returns nothing for an empty clip", () => {
    expect(rulerTicks(0, clock)).toEqual([]);
    expect(rulerTicks(-4, clock)).toEqual([]);
  });
});
