<script setup lang="ts">
/*
 * Toasts (README §04): 320px cards above the status bar — a glyph column in
 * the severity's colour, a title and one line, one action — bordered in the
 * state colour when the state is urgent. Transient; the bell keeps the record.
 */
import { computed } from "vue";
import { severityIsUrgent, severityMark } from "@ui/lib/notificationSeverity";
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
    glyphStyle: { color: mark.color },
    // Only an urgent card borders itself in its colour; the rest stay quiet.
    cardStyle: severityIsUrgent(kind) ? { borderColor: mark.color } : undefined,
  };
}

// The store appends, so the newest toast is last; the shelf shows it first and
// pushes the older ones down.
const ordered = computed(() => [...toasts.items].reverse());
</script>

<template>
  <!-- Teleported for the same reason ContextMenu and Tooltip are: a `fixed`
       layer resolves against the nearest ancestor with a transform, filter or
       container-type, so any such property added to the app frame later would
       silently relocate every toast. <body> can never grow one. -->
  <Teleport to="body">
    <div
      class="pointer-events-none fixed right-4 bottom-[calc(var(--mold-shell-statusbar-h)+12px)] z-50 flex flex-col items-end gap-2"
      aria-label="Notifications"
    >
      <TransitionGroup
        enter-active-class="transition duration-150 ease-out"
        enter-from-class="translate-y-2 opacity-0"
        leave-active-class="absolute transition duration-150 ease-in"
        leave-to-class="translate-y-1 opacity-0"
        move-class="transition-transform duration-150 ease-out"
      >
        <div
          v-for="toast in ordered"
          :key="toast.id"
          class="pointer-events-auto grid w-80 max-w-[calc(100vw-2rem)] grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-x-2.5 gap-y-1.5 rounded-control border border-border bg-surface p-3 shadow-md"
          :style="tone(toast.kind).cardStyle"
          :role="severityIsUrgent(toast.kind) ? 'alert' : 'status'"
          :aria-live="severityIsUrgent(toast.kind) ? 'assertive' : 'polite'"
          @click.self="toasts.click(toast.id)"
        >
          <span
            data-test="toast-status-icon"
            aria-hidden="true"
            :data-kind="toast.kind"
            class="w-3 text-center font-mono text-xs leading-4"
            :style="tone(toast.kind).glyphStyle"
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
            <span data-test="toast-title" class="block text-xs font-semibold leading-4 text-fg">
              {{ toast.message }}
            </span>
            <span
              v-if="toast.description"
              data-test="toast-description"
              class="mt-0.5 block text-micro leading-body text-fg-dim"
            >
              {{ toast.description }}
            </span>
          </button>
          <!-- A toast with an onClick opens something instead of dismissing, so
             the ✕ is the only way to send that card away. -->
          <button
            type="button"
            data-test="toast-dismiss"
            class="-mt-1 -mr-1 flex h-6 w-6 items-center justify-center rounded-control text-fg-dim transition-colors duration-100 hover:bg-row-hover hover:text-fg"
            aria-label="Dismiss notification"
            @click="toasts.dismiss(toast.id)"
          >
            <svg viewBox="0 0 16 16" class="h-3 w-3" aria-hidden="true">
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
            class="col-start-2 col-end-4 justify-self-end text-micro font-semibold text-accent hover:brightness-110"
            @click="toasts.runAction(toast.id)"
          >
            {{ toast.action.label }}
          </button>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>
