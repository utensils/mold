/*
 * The clip timeline's ruler. Ticks are absolutely positioned inside a
 * relatively positioned strip, so a tick at 100% starts its label AT the
 * right edge and paints the whole label past it — where the bench's
 * `overflow-hidden` cuts it off. Whether that happens depends on the clip's
 * length (a round 20s at a 5s step lands a tick exactly on 100%; 23s does
 * not), which is why the clipping looked intermittent.
 *
 * The mock authors the closing mark specially — `right: 0` with the label
 * right-aligned — and so does this: the last mark is pinned to the right edge
 * and grows inward.
 */

/** Coarsest round interval that still marks the clip out in a few steps. */
const TICK_INTERVALS = [1, 2, 5, 10, 15, 30, 60] as const;
const FALLBACK_INTERVAL = 120;

/**
 * How close to the end a mark must sit before it is pinned right. A mark one
 * part in 200 short of the end is visually AT the end, and its label overflows
 * just as badly.
 */
const END_TOLERANCE_PERCENT = 99.5;

export interface RulerTick {
  /** Seconds from the start of the clip. */
  at: number;
  /** Mono clock caption, e.g. `0:15`. */
  label: string;
  /** True where the mark is pinned to the strip's right edge. */
  atEnd: boolean;
  /** Inline style for the mark: `left` normally, `right: 0` at the end. */
  style: Record<string, string>;
}

export function rulerTickInterval(totalSeconds: number): number {
  return TICK_INTERVALS.find((candidate) => totalSeconds / candidate <= 7) ?? FALLBACK_INTERVAL;
}

/**
 * Every mark on the ruler for a clip of `totalSeconds`, with the style each
 * one needs. `label` formats the clock caption.
 */
export function rulerTicks(totalSeconds: number, label: (seconds: number) => string): RulerTick[] {
  if (totalSeconds <= 0) return [];
  const step = rulerTickInterval(totalSeconds);
  const span = Math.max(totalSeconds, 0.001);
  const marks: RulerTick[] = [];
  for (let at = 0; at <= totalSeconds + 0.001; at += step) {
    const percent = (at / span) * 100;
    const atEnd = percent >= END_TOLERANCE_PERCENT;
    marks.push({
      at,
      label: label(at),
      atEnd,
      style: atEnd ? { right: "0" } : { left: `${percent}%` },
    });
  }
  return marks;
}
