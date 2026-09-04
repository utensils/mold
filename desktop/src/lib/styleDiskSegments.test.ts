import { describe, expect, it } from "vitest";
import { styleDiskSegments } from "./styleDiskSegments";

/**
 * The mock's disk meter draws three segments. The shelf drew one per family
 * with a five-tone cycle and no cap, so a ten-family fleet became ten slivers
 * inside an 8px bar with tones repeating every fifth one.
 */
describe("styleDiskSegments", () => {
  const section = (heading: string, bytes: number): [string, number] => [heading, bytes];

  it("keeps the four biggest families and folds the rest into one 'Other'", () => {
    const segments = styleDiskSegments([
      section("flux", 10),
      section("sdxl", 9),
      section("wan", 8),
      section("ltx", 7),
      section("qwen", 6),
      section("sd35", 5),
    ]);
    expect(segments.map((s) => s.heading)).toEqual(["flux", "sdxl", "wan", "ltx", "Other"]);
    expect(segments.at(-1)!.bytes).toBe(11);
    // The fold is a distinct, neutral tone — never a repeat of a family's.
    expect(segments.at(-1)!.tone).toBe("bg-surface-3");
    expect(new Set(segments.slice(0, 4).map((s) => s.tone)).size).toBe(4);
  });

  it("sorts by bytes so a wide family never hides behind a sliver", () => {
    const segments = styleDiskSegments([section("a", 1), section("b", 30), section("c", 2)]);
    expect(segments.map((s) => s.heading)).toEqual(["b", "c", "a"]);
  });

  it("drops empty families and never emits an empty 'Other'", () => {
    const segments = styleDiskSegments([
      section("flux", 10),
      section("sdxl", 0),
      section("wan", 3),
    ]);
    expect(segments.map((s) => s.heading)).toEqual(["flux", "wan"]);
  });

  it("names the folded families so the meter's tooltip can say what is in there", () => {
    const segments = styleDiskSegments([
      section("a", 5),
      section("b", 4),
      section("c", 3),
      section("d", 2),
      section("e", 1),
      section("f", 1),
    ]);
    expect(segments.at(-1)!.headings).toEqual(["e", "f"]);
    expect(segments[0]!.headings).toEqual(["a"]);
  });

  it("leaves four or fewer families alone", () => {
    const segments = styleDiskSegments([section("a", 4), section("b", 3)]);
    expect(segments.map((s) => s.heading)).toEqual(["a", "b"]);
    expect(segments.every((s) => s.tone !== "bg-surface-3")).toBe(true);
  });
});
