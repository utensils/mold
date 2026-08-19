/*
 * One severity palette for every notification surface (bell, toasts): green
 * for success, yellow for a warning, red for an error, neutral for plain
 * information. The colors are design tokens (`ui/tokens.css`), never hexes, so
 * every theme family and light/dark pair stays consistent.
 *
 * Each tone also carries a text label. Severity is never communicated by color
 * alone — the label ships as assistive text next to the dot.
 */
import type { NotificationKind } from "../stores/notifications";

export interface NotificationTone {
  /** CSS color for the dot/glyph — a token reference, resolved by the theme. */
  color: string;
  /** Assistive-text severity name. */
  label: string;
}

export const NOTIFICATION_TONES: Record<NotificationKind, NotificationTone> = {
  info: { color: "var(--ink-3)", label: "Info" },
  success: { color: "var(--success)", label: "Success" },
  warning: { color: "var(--warning)", label: "Warning" },
  error: { color: "var(--stop)", label: "Error" },
};

export function notificationTone(kind: NotificationKind): NotificationTone {
  return NOTIFICATION_TONES[kind] ?? NOTIFICATION_TONES.info;
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
