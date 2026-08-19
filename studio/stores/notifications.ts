import { defineStore } from "pinia";
import { computed, ref } from "vue";

/*
 * Durable record of toast-level messages (spec §08 G11 adjunct). Toasts stay
 * transient; this store is the history behind the notifications bell so a
 * long server error the user never caught in time is still readable in full.
 * Session-scoped by design — it answers "what just happened", not an audit
 * log. Every surface records through here; none renders a second policy.
 */

/** Severity, rendered green (success) / yellow (warning) / red (error);
 *  plain info stays neutral. See `lib/notificationTone.ts`. */
export type NotificationKind = "info" | "success" | "warning" | "error";

export interface NotificationEntry {
  id: number;
  kind: NotificationKind;
  /** The toast's primary line, untruncated. */
  text: string;
  /** Optional supporting copy (full server error bodies land here too). */
  description: string | null;
  /** Origin machine, when the caller knows it. */
  hostLabel: string | null;
  atMs: number;
  read: boolean;
  /** Consecutive identical messages collapse into one row with a count. */
  repeat: number;
}

export interface NotificationInput {
  kind: NotificationKind;
  text: string;
  description?: string | null;
  hostLabel?: string | null;
  /** Wall-clock stamp; callers pass Date.now() (kept injectable for tests). */
  atMs?: number;
}

/** Newest entries kept; older history falls off. */
export const NOTIFICATION_CAP = 100;

let nextId = 1;

export const useNotificationsStore = defineStore("studio-notifications", () => {
  /** Newest first. */
  const entries = ref<NotificationEntry[]>([]);

  const unreadCount = computed(
    () => entries.value.filter((entry) => !entry.read).length,
  );

  function record(input: NotificationInput): void {
    const atMs = input.atMs ?? Date.now();
    const newest = entries.value[0];
    if (
      newest &&
      newest.kind === input.kind &&
      newest.text === input.text &&
      (newest.description ?? null) === (input.description ?? null) &&
      (newest.hostLabel ?? null) === (input.hostLabel ?? null)
    ) {
      newest.repeat += 1;
      newest.atMs = atMs;
      newest.read = false;
      return;
    }
    entries.value.unshift({
      id: nextId++,
      kind: input.kind,
      text: input.text,
      description: input.description ?? null,
      hostLabel: input.hostLabel ?? null,
      atMs,
      read: false,
      repeat: 1,
    });
    if (entries.value.length > NOTIFICATION_CAP) {
      entries.value.length = NOTIFICATION_CAP;
    }
  }

  function markAllRead(): void {
    for (const entry of entries.value) entry.read = true;
  }

  function clear(): void {
    entries.value = [];
  }

  return { entries, unreadCount, record, markAllRead, clear };
});
