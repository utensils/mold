/**
 * The Styles shelf's "Disk used by styles" meter, as segments.
 *
 * A segment per family with a five-tone cycle degenerates into stripes the
 * moment a fleet carries more than a handful of families: ten segments inside
 * an 8px bar, several of them the same colour and a few pixels wide. The mock
 * draws three. So: sort by bytes, keep the four biggest as their own tones,
 * and fold everything left into ONE neutral "Other" that still names what it
 * swallowed for the tooltip.
 */

/** Distinct tones for the named families. Never cycled — the cap is the point. */
const SEGMENT_TONES = ["bg-accent", "bg-sapphire", "bg-mauve", "bg-teal"] as const;
/** The fold is deliberately not one of the family tones. */
const OTHER_TONE = "bg-surface-3";
const NAMED_SEGMENTS = SEGMENT_TONES.length;

export interface StyleDiskSegment {
  /** What the segment is called — a family heading, or "Other" for the fold. */
  heading: string;
  /** Every family inside it: one for a named segment, the rest for the fold. */
  headings: string[];
  bytes: number;
  tone: string;
}

/** `sections` is `[heading, bytes]` in whatever order the shelf grouped them. */
export function styleDiskSegments(
  sections: readonly (readonly [string, number])[],
): StyleDiskSegment[] {
  const sized = sections
    .filter(([, bytes]) => bytes > 0)
    .slice()
    .sort((a, b) => b[1] - a[1]);

  const named = sized.slice(0, NAMED_SEGMENTS).map(([heading, bytes], index) => ({
    heading,
    headings: [heading],
    bytes,
    tone: SEGMENT_TONES[index]!,
  }));

  const rest = sized.slice(NAMED_SEGMENTS);
  if (rest.length === 0) return named;
  return [
    ...named,
    {
      heading: "Other",
      headings: rest.map(([heading]) => heading),
      bytes: rest.reduce((sum, [, bytes]) => sum + bytes, 0),
      tone: OTHER_TONE,
    },
  ];
}
