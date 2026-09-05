/*
 * Severity marks for every notification surface — the toast shelves and the
 * notifications bell, on web and desktop alike.
 *
 * This lives in the design-system layer because @ui may not import the studio
 * layer (studio consumes @ui, not the reverse), and there must be exactly ONE
 * table: three parallel copies had already drifted, showing the same info event
 * as "i" in one place and "•" in another.
 *
 * Color is the fast signal but never the only one. Each severity carries a
 * distinct glyph so a viewer with a color-vision deficiency can still tell them
 * apart, plus a name for assistive tech.
 */

export type NotificationSeverity = "info" | "success" | "warning" | "error";

export interface SeverityMark {
  /** Distinct visible mark — never repeated across severities. */
  glyph: string;
  /** Severity name, rendered as assistive text next to the glyph. */
  label: string;
  /** Token reference for the glyph color, resolved by the active theme. */
  color: string;
}

export const SEVERITY_MARKS: Record<NotificationSeverity, SeverityMark> = {
  // Green covers everything that is not a warning or an error: an ordinary
  // notice and a success both mean "nothing is wrong". The glyph, not the hue,
  // separates them.
  info: { glyph: "•", label: "Info", color: "var(--mold-success)" },
  success: { glyph: "✓", label: "Success", color: "var(--mold-success)" },
  warning: { glyph: "!", label: "Warning", color: "var(--mold-warning)" },
  error: { glyph: "✕", label: "Error", color: "var(--mold-error)" },
};

export function severityMark(kind: NotificationSeverity): SeverityMark {
  return SEVERITY_MARKS[kind] ?? SEVERITY_MARKS.info;
}

/**
 * Severities that belong in an assertive live region. A warning here is not a
 * "by the way" — the one that exists is the sticky "the machine you are
 * generating on is gone", which is as time-sensitive as an error and must not
 * wait for a screen-reader user to go idle.
 */
export function severityIsUrgent(kind: NotificationSeverity): boolean {
  return kind === "error" || kind === "warning";
}

/**
 * Presentation derived from the one color, so no surface restates a hue.
 * `tint` is the translucent wash a chip or border uses; `solid` is the filled
 * treatment an error chip takes, whose text reads with `--mold-on-accent`.
 */
export function severityTint(
  kind: NotificationSeverity,
  percent: number,
): string {
  return `color-mix(in srgb, ${severityMark(kind).color} ${percent}%, transparent)`;
}
