<script setup lang="ts">
import { computed } from "vue";
import { severityIsUrgent, severityMark, severityTint } from "@ui/lib/notificationSeverity";
import { useToastStore, type Toast } from "../../stores/toasts";

const toasts = useToastStore();

/*
 * Glyphs, severity names, and hues all come from the one shared table
 * (@ui/lib/notificationSeverity) — restating a color here is exactly how the
 * surfaces drifted before. Severity reads as green (an ordinary notice or a
 * success) / yellow (warning) / red (error), and never by color alone.
 */
function tone(kind: Toast["kind"]) {
  const mark = severityMark(kind);
  return {
    ...mark,
    // An error is the one filled chip; the rest are washed so the shelf does
    // not read as a wall of alarm.
    chip:
      kind === "error"
        ? { background: mark.color, color: "var(--mold-on-accent)" }
        : { background: severityTint(kind, 20), color: mark.color },
  };
}

// The store appends, so the newest toast is last; the shelf shows it first and
// pushes the older ones down.
const ordered = computed(() => [...toasts.items].reverse());
</script>

<template>
  <div
    class="pointer-events-none fixed top-14 right-4 z-50 flex flex-col items-end gap-2"
    aria-label="Notifications"
  >
    <TransitionGroup
      enter-active-class="transition duration-150 ease-out"
      enter-from-class="-translate-y-2 opacity-0"
      leave-active-class="absolute transition duration-150 ease-in"
      leave-to-class="-translate-y-1 opacity-0"
      move-class="transition-transform duration-150 ease-out"
    >
      <div
        v-for="toast in ordered"
        :key="toast.id"
        class="border-border pointer-events-auto grid w-[min(25rem,calc(100vw-2rem))] grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-x-3 gap-y-2 rounded-control border bg-bg px-4 py-3 text-sm shadow-md"
        :role="severityIsUrgent(toast.kind) ? 'alert' : 'status'"
        :aria-live="severityIsUrgent(toast.kind) ? 'assertive' : 'polite'"
        @click.self="toasts.click(toast.id)"
      >
        <span
          data-test="toast-status-icon"
          aria-hidden="true"
          :data-kind="toast.kind"
          class="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-control font-semibold"
          :style="tone(toast.kind).chip"
        >
          {{ tone(toast.kind).glyph }}
        </span>
        <button
          type="button"
          class="min-w-0 text-left"
          :title="toast.onClick ? undefined : 'Dismiss'"
          @click="toasts.click(toast.id)"
        >
          <span class="sr-only">{{ tone(toast.kind).label }}</span>
          <span data-test="toast-title" class="block font-semibold text-fg">
            {{ toast.message }}
          </span>
          <span
            v-if="toast.description"
            data-test="toast-description"
            class="mt-0.5 block text-micro leading-relaxed text-fg-2"
          >
            {{ toast.description }}
          </span>
        </button>
        <button
          type="button"
          data-test="toast-dismiss"
          class="-mt-1 -mr-1 flex h-7 w-7 items-center justify-center rounded-control text-fg-dim transition-colors duration-100 hover:bg-row-hover hover:text-fg"
          aria-label="Dismiss notification"
          @click="toasts.dismiss(toast.id)"
        >
          <svg viewBox="0 0 16 16" class="h-3.5 w-3.5" aria-hidden="true">
            <path
              d="m3.75 3.75 8.5 8.5m0-8.5-8.5 8.5"
              fill="none"
              stroke="currentColor"
              stroke-linecap="round"
              stroke-width="1.5"
            />
          </svg>
        </button>
        <button
          v-if="toast.action"
          type="button"
          data-test="toast-action"
          class="col-start-2 col-end-4 justify-self-end rounded-control px-2 py-1 text-micro font-semibold text-accent transition-colors duration-100 hover:bg-accent-tint hover:brightness-110"
          @click="toasts.runAction(toast.id)"
        >
          {{ toast.action.label }}
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>
