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
  info: { glyph: "•", label: "Info", color: "var(--ink-3)" },
  success: { glyph: "✓", label: "Success", color: "var(--success)" },
  warning: { glyph: "!", label: "Warning", color: "var(--warning)" },
  error: { glyph: "✕", label: "Error", color: "var(--stop)" },
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
