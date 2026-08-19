/*
 * Notification tones for the bell: the shared severity marks (`@ui/lib/
 * notificationSeverity`, the one table every surface reads) plus the badge
 * fill only a counted badge needs. Green success, yellow warning, red error,
 * neutral info — always as design tokens, never hexes, so every theme family
 * and light/dark pair stays consistent.
 */
import {
  SEVERITY_MARKS,
  severityMark,
  type SeverityMark,
} from "@ui/lib/notificationSeverity";
import type { NotificationKind } from "../stores/notifications";

/** A severity mark (glyph, label, color) plus the badge fill the bell needs. */
export interface NotificationTone extends SeverityMark {
  /**
   * Solid fill for a counted badge. Never a translucent hint ink such as
   * `--ink-3`: printing a count on one has no predictable contrast. Text on
   * any of these reads with `--on-status`.
   */
  badge: string;
}

/** Badge fills, keyed to the shared marks. Green for anything that is not a
 *  warning or an error, so a bell carrying only notices reads as "all fine". */
const BADGE_FILLS: Record<NotificationKind, string> = {
  info: "var(--success)",
  success: "var(--success)",
  warning: "var(--warning)",
  error: "var(--stop)",
};

export const NOTIFICATION_TONES: Record<NotificationKind, NotificationTone> = {
  info: { ...SEVERITY_MARKS.info, badge: BADGE_FILLS.info },
  success: { ...SEVERITY_MARKS.success, badge: BADGE_FILLS.success },
  warning: { ...SEVERITY_MARKS.warning, badge: BADGE_FILLS.warning },
  error: { ...SEVERITY_MARKS.error, badge: BADGE_FILLS.error },
};

/** Ink that stays legible on any `badge` fill (defined per theme). */
export const NOTIFICATION_BADGE_INK = "var(--on-status)";

export function notificationTone(kind: NotificationKind): NotificationTone {
  return {
    ...severityMark(kind),
    badge: BADGE_FILLS[kind] ?? BADGE_FILLS.info,
  };
}

/** Highest-severity-wins order for a summary affordance (the bell badge). */
const SEVERITY_RANK: Record<NotificationKind, number> = {
  info: 0,
  success: 1,
  warning: 2,
  error: 3,
};

/**
 * The kind a single badge should take for a set of entries — a red badge over
 * a lone "saved to Library" is a false alarm, and a green one over an error
 * hides it.
 */
export function mostSevereKind(
  kinds: readonly NotificationKind[],
): NotificationKind {
  let worst: NotificationKind = "info";
  for (const kind of kinds) {
    if (SEVERITY_RANK[kind] > SEVERITY_RANK[worst]) worst = kind;
  }
  return worst;
}
