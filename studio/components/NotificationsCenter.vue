<script setup lang="ts">
import { computed, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import Popover from "@ui/components/Popover.vue";
import {
  useNotificationsStore,
  type NotificationEntry,
} from "../stores/notifications";

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

function toggle() {
  const next = !open.value;
  open.value = next;
  if (next) store.markAllRead();
}

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
        >
          {{ unreadLabel }}
        </span>
      </button>
    </template>

    <div class="notifications-panel" data-test="notifications-panel">
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
          <span class="notifications-panel__dot" aria-hidden="true" />
          <div class="notifications-panel__copy">
            <p class="notifications-panel__text">
              {{ entry.text }}
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
  border: 1px solid var(--ce, var(--edge, #d0d5dd));
  border-radius: 8px;
  background: transparent;
  color: var(--ink-2, #667085);
  cursor: pointer;
}

.notifications-bell:hover {
  color: var(--ink, currentColor);
}

.notifications-bell__badge {
  position: absolute;
  top: -5px;
  right: -5px;
  min-width: 15px;
  padding: 0 3px;
  border-radius: 999px;
  background: var(--stop, #b42318);
  color: #fff;
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
  color: var(--ink-2, #667085);
  font-size: 12px;
  cursor: pointer;
}

.notifications-panel > header button:hover {
  color: var(--ink, currentColor);
}

.notifications-panel__empty {
  margin: 0;
  padding: 10px 4px 12px;
  color: var(--ink-3, #98a2b3);
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
  border-top: 1px solid var(--ce, var(--edge, #d0d5dd));
}

.notifications-panel__dot {
  width: 7px;
  height: 7px;
  flex: none;
  margin-top: 5px;
  border-radius: 999px;
  background: var(--ink-3, #98a2b3);
}

.notifications-panel li[data-kind="error"] .notifications-panel__dot {
  background: var(--stop, #b42318);
}

.notifications-panel li[data-kind="success"] .notifications-panel__dot {
  background: var(--safelight, #067647);
}

.notifications-panel__copy {
  min-width: 0;
  flex: 1;
}

.notifications-panel__copy p {
  margin: 0;
}

.notifications-panel__text {
  font-size: 13px;
  overflow-wrap: anywhere;
}

.notifications-panel__repeat {
  color: var(--ink-3, #98a2b3);
  font-size: 11px;
}

.notifications-panel__detail {
  margin-top: 2px !important;
  color: var(--ink-2, #667085);
  font-size: 12px;
  line-height: 1.45;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

.notifications-panel__meta {
  margin-top: 3px !important;
  color: var(--ink-3, #98a2b3);
  font-size: 11px;
}
</style>
