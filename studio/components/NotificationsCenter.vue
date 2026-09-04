<script setup lang="ts">
import { computed, nextTick, onUnmounted, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import Popover from "@ui/components/Popover.vue";
import {
  useNotificationsStore,
  type NotificationAction,
  type NotificationEntry,
} from "../stores/notifications";
import {
  NOTIFICATION_BADGE_INK,
  mostSevereKind,
  notificationTone,
} from "../lib/notificationTone";
import {
  copyTextToClipboard,
  notificationClipboardText,
} from "../lib/notificationClipboard";

/*
 * The notifications bell — the durable record behind transient toasts. A long
 * server error the user never caught in time is readable here in full. One
 * shared component on web and desktop (spec §06); iPhone gets its own
 * placement pass. Opening the panel marks everything read; Clear empties it.
 */
const store = useNotificationsStore();
const open = ref(false);

const unreadLabel = computed(() =>
  store.unreadCount > 99 ? "99+" : String(store.unreadCount),
);

/* The badge takes the worst unread severity: red for an error, yellow for a
 * warning, green when only good news is waiting. */
const badgeTone = computed(() =>
  notificationTone(
    mostSevereKind(
      store.entries.filter((entry) => !entry.read).map((entry) => entry.kind),
    ),
  ),
);

function toggle() {
  const next = !open.value;
  open.value = next;
  if (next) store.markAllRead();
}

function toneStyle(entry: NotificationEntry) {
  return { color: notificationTone(entry.kind).color };
}

function toneGlyph(entry: NotificationEntry): string {
  return notificationTone(entry.kind).glyph;
}

function toneLabel(entry: NotificationEntry): string {
  return notificationTone(entry.kind).label;
}

/*
 * Copying a notification out. The app shells disable text selection on their
 * chrome, so a long server error body is unreachable without a real control —
 * and the bell exists precisely to keep that text readable after the toast is
 * gone. The row reports the outcome instead of assuming it worked; an insecure
 * origin or a denied permission is a normal case, not an edge.
 */
const copyState = ref<{ id: number; ok: boolean } | null>(null);
type ActionStatus = "pending" | "done" | "failed";
interface EntryActionState {
  action: NotificationAction;
  status: ActionStatus;
}
const actionStates = ref<Record<number, EntryActionState>>({});
let copyResetTimer: ReturnType<typeof setTimeout> | null = null;
/** Rising token: only the newest click may paint the outcome. Clicking one row
 *  then another must not settle "Copied" on whichever promise resolves last. */
let copyEpoch = 0;

async function copyEntry(entry: NotificationEntry) {
  const epoch = ++copyEpoch;
  const ok = await copyTextToClipboard(
    notificationClipboardText(entry, timeLabel(entry)),
  );
  if (epoch !== copyEpoch) return;
  copyState.value = { id: entry.id, ok };
  await announceCopy(
    ok
      ? "Notification copied to the clipboard"
      : "Could not copy the notification",
  );
  if (copyResetTimer) clearTimeout(copyResetTimer);
  copyResetTimer = setTimeout(() => {
    copyState.value = null;
    copyAnnouncement.value = "";
    copyResetTimer = null;
  }, 2000);
}

function copyLabel(entry: NotificationEntry): string {
  if (copyState.value?.id !== entry.id) return "Copy";
  return copyState.value.ok ? "Copied" : "Copy failed";
}

function actionLabel(entry: NotificationEntry): string {
  if (!entry.action) return "";
  const state = actionStates.value[entry.id];
  const status = state?.action === entry.action ? state.status : undefined;
  if (status === "pending") {
    return entry.action.pendingLabel ?? "Working…";
  }
  if (status === "done") {
    return entry.action.doneLabel ?? "Done";
  }
  return status === "failed"
    ? `${entry.action.label} failed`
    : entry.action.label;
}

function actionDisabled(entry: NotificationEntry): boolean {
  const state = actionStates.value[entry.id];
  return (
    state?.action === entry.action &&
    (state.status === "pending" || state.status === "done")
  );
}

async function runEntryAction(entry: NotificationEntry) {
  const action = entry.action;
  if (!action || actionDisabled(entry)) return;
  actionStates.value[entry.id] = { action, status: "pending" };
  try {
    await action.run();
    if (actionStates.value[entry.id]?.action === action) {
      actionStates.value[entry.id] = { action, status: "done" };
    }
  } catch {
    if (actionStates.value[entry.id]?.action === action) {
      actionStates.value[entry.id] = { action, status: "failed" };
    }
  }
}

/* The button's accessible name cannot change under the user's fingers, so the
 * outcome goes to a live region instead — otherwise a failed copy on a plain
 * http origin is announced as nothing at all and the text was never taken. */
const copyAnnouncement = ref("");

/**
 * A live region only speaks when its text actually changes, so re-setting the
 * same sentence is silence — and pressing Copy twice is exactly what someone
 * unsure whether the first one worked does. Clear it, let that render, then
 * write the outcome so every press is announced.
 */
async function announceCopy(message: string) {
  copyAnnouncement.value = "";
  await nextTick();
  copyAnnouncement.value = message;
}

onUnmounted(() => {
  if (copyResetTimer) clearTimeout(copyResetTimer);
});

function timeLabel(entry: NotificationEntry): string {
  return new Date(entry.atMs).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}
</script>

<template>
  <Popover v-model:open="open" placement="bottom-end" label="Notifications">
    <template #trigger>
      <button
        type="button"
        class="notifications-bell"
        data-test="notifications-bell"
        aria-label="Notifications"
        :aria-expanded="open"
        @click="toggle"
      >
        <Icon name="bell" :size="16" />
        <span
          v-if="store.unreadCount > 0"
          class="notifications-bell__badge"
          data-test="notifications-unread"
          :style="{
            background: badgeTone.badge,
            color: NOTIFICATION_BADGE_INK,
          }"
        >
          {{ unreadLabel }}
        </span>
      </button>
    </template>

    <div class="notifications-panel" data-test="notifications-panel">
      <p
        class="notifications-panel__sr"
        role="status"
        aria-live="polite"
        data-test="notifications-copy-status"
      >
        {{ copyAnnouncement }}
      </p>
      <header>
        <strong>Notifications</strong>
        <button
          v-if="store.entries.length"
          type="button"
          data-test="notifications-clear"
          @click="store.clear()"
        >
          Clear
        </button>
      </header>
      <p v-if="!store.entries.length" class="notifications-panel__empty">
        No notifications yet — errors and status messages collect here.
      </p>
      <ul v-else>
        <li
          v-for="entry in store.entries"
          :key="entry.id"
          :data-kind="entry.kind"
        >
          <!-- Glyph, not a bare dot: severity must survive a color-vision
               deficiency, and the mark differs per kind. -->
          <span
            class="notifications-panel__dot"
            data-test="notifications-dot"
            :style="toneStyle(entry)"
            aria-hidden="true"
            >{{ toneGlyph(entry) }}</span
          >
          <span class="notifications-panel__sr">{{ toneLabel(entry) }}</span>
          <div class="notifications-panel__copy">
            <p class="notifications-panel__text">
              {{ entry.text }}
              <span v-if="entry.action" aria-hidden="true"> — </span>
              <button
                v-if="entry.action"
                type="button"
                class="notifications-panel__inline-action"
                data-test="notifications-action"
                :disabled="actionDisabled(entry)"
                @click="runEntryAction(entry)"
              >
                {{ actionLabel(entry) }}
              </button>
              <span v-if="entry.repeat > 1" class="notifications-panel__repeat">
                ×{{ entry.repeat }}
              </span>
            </p>
            <p v-if="entry.description" class="notifications-panel__detail">
              {{ entry.description }}
            </p>
            <p class="notifications-panel__meta">
              <template v-if="entry.hostLabel"
                >{{ entry.hostLabel }} · </template
              >{{ timeLabel(entry) }}
            </p>
          </div>
          <button
            type="button"
            class="notifications-panel__copy-action"
            data-test="notifications-copy"
            :aria-label="`Copy notification: ${entry.text}`"
            @click="copyEntry(entry)"
          >
            {{ copyLabel(entry) }}
          </button>
        </li>
      </ul>
    </div>
  </Popover>
</template>

<style scoped>
.notifications-bell {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 1px solid var(--mold-border-control, var(--mold-border, #d0d5dd));
  border-radius: 8px;
  background: transparent;
  color: var(--mold-text-2, #667085);
  cursor: pointer;
}

.notifications-bell:hover {
  color: var(--mold-text);
}

.notifications-bell__badge {
  position: absolute;
  top: -5px;
  right: -5px;
  min-width: 15px;
  padding: 0 3px;
  border-radius: 999px;
  background: var(--mold-error, #b42318);
  color: var(--mold-on-accent, #fff);
  font-size: 9px;
  font-weight: 700;
  line-height: 15px;
  text-align: center;
}

.notifications-panel {
  width: min(360px, calc(100vw - 32px));
  max-height: min(420px, 70vh);
  overflow-y: auto;
}

.notifications-panel > header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 2px 4px 8px;
  font-size: 13px;
}

.notifications-panel > header button {
  border: none;
  background: none;
  color: var(--mold-text-2, #667085);
  font-size: 12px;
  cursor: pointer;
}

.notifications-panel > header button:hover {
  color: var(--mold-text);
}

.notifications-panel__empty {
  margin: 0;
  padding: 10px 4px 12px;
  color: var(--mold-text-dim, #98a2b3);
  font-size: 12px;
}

.notifications-panel ul {
  margin: 0;
  padding: 0;
  list-style: none;
}

.notifications-panel li {
  display: flex;
  gap: 8px;
  padding: 8px 4px;
  border-top: 1px solid var(--mold-border-control, var(--mold-border, #d0d5dd));
}

.notifications-panel__dot {
  width: 12px;
  flex: none;
  margin-top: 1px;
  font-size: 11px;
  line-height: 1.5;
  text-align: center;
  color: var(--mold-text-dim, #98a2b3);
}

/* Visually-hidden utility: the severity name next to each glyph, and the
 * copy-outcome live region. Both are text only assistive tech needs. */
.notifications-panel__sr {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
  border: 0;
}

.notifications-panel__copy {
  min-width: 0;
  flex: 1;
  /* The desktop shell disables selection app-wide; notification text is
     content, so it stays selectable for a manual drag-copy too. */
  user-select: text;
  -webkit-user-select: text;
}

.notifications-panel__copy-action {
  flex: none;
  align-self: flex-start;
  margin-top: 1px;
  padding: 1px 5px;
  border: 1px solid transparent;
  border-radius: 6px;
  background: none;
  color: var(--mold-text-dim, #98a2b3);
  font-size: 11px;
  cursor: pointer;
}

.notifications-panel li:hover .notifications-panel__copy-action,
.notifications-panel__copy-action:focus-visible {
  border-color: var(--mold-border-control, var(--mold-border, #d0d5dd));
  color: var(--mold-text);
}

.notifications-panel__copy p {
  margin: 0;
}

.notifications-panel__text {
  font-size: 13px;
  overflow-wrap: anywhere;
}

.notifications-panel__repeat {
  color: var(--mold-text-dim, #98a2b3);
  font-size: 11px;
}

.notifications-panel__inline-action {
  margin: 0;
  padding: 0;
  border: 0;
  background: none;
  color: var(--link, var(--mold-text, #2563eb));
  font: inherit;
  font-weight: 600;
  text-decoration: underline;
  text-underline-offset: 2px;
  cursor: pointer;
}

.notifications-panel__inline-action:hover:not(:disabled) {
  text-decoration-thickness: 2px;
}

.notifications-panel__inline-action:focus-visible {
  border-radius: 2px;
  outline: 2px solid var(--mold-blue, currentColor);
  outline-offset: 2px;
}

.notifications-panel__inline-action:disabled {
  color: var(--mold-text-dim, #98a2b3);
  cursor: default;
}

.notifications-panel__detail {
  margin-top: 2px !important;
  color: var(--mold-text-2, #667085);
  font-size: 12px;
  line-height: 1.45;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

.notifications-panel__meta {
  margin-top: 3px !important;
  color: var(--mold-text-dim, #98a2b3);
  font-size: 11px;
}
</style>
