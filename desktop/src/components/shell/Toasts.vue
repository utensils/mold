<script setup lang="ts">
import { computed } from "vue";
import { useToastStore, type Toast } from "../../stores/toasts";

const toasts = useToastStore();

/*
 * Severity reads as green (success) / yellow (warning) / red (error), with
 * plain info left cool and neutral. The glyph is decorative, so each tone also
 * carries a name for assistive tech — severity is never color alone.
 */
const TONES: Record<Toast["kind"], { glyph: string; label: string; chip: string }> = {
  info: {
    glyph: "i",
    label: "Info",
    chip: "bg-[color-mix(in_srgb,var(--halide)_18%,transparent)] text-halide",
  },
  success: {
    glyph: "✓",
    label: "Success",
    chip: "bg-[color-mix(in_srgb,var(--success)_20%,transparent)] text-success",
  },
  warning: {
    glyph: "!",
    label: "Warning",
    chip: "bg-[color-mix(in_srgb,var(--warning)_22%,transparent)] text-warning",
  },
  error: { glyph: "!", label: "Error", chip: "bg-stop text-[var(--desk)]" },
};

function tone(kind: Toast["kind"]) {
  return TONES[kind] ?? TONES.info;
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
        class="border-edge pointer-events-auto grid w-[min(25rem,calc(100vw-2rem))] grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-x-3 gap-y-2 rounded-[var(--radius-card)] border bg-bench px-4 py-3 text-body shadow-[0_10px_28px_rgba(0,0,0,0.28)]"
        :role="toast.kind === 'error' ? 'alert' : 'status'"
        :aria-live="toast.kind === 'error' ? 'assertive' : 'polite'"
        @click.self="toasts.click(toast.id)"
      >
        <span
          data-test="toast-status-icon"
          aria-hidden="true"
          :data-kind="toast.kind"
          class="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full font-semibold"
          :class="tone(toast.kind).chip"
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
          <span data-test="toast-title" class="block font-semibold text-ink">
            {{ toast.message }}
          </span>
          <span
            v-if="toast.description"
            data-test="toast-description"
            class="mt-0.5 block text-caption leading-relaxed text-ink-2"
          >
            {{ toast.description }}
          </span>
        </button>
        <button
          type="button"
          data-test="toast-dismiss"
          class="-mt-1 -mr-1 flex h-7 w-7 items-center justify-center rounded-full text-ink-3 transition-colors duration-100 hover:bg-[color-mix(in_srgb,var(--rebate)_8%,transparent)] hover:text-ink"
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
          class="col-start-2 col-end-4 justify-self-end rounded-control px-2 py-1 text-caption font-semibold text-safelight transition-colors duration-100 hover:bg-[color-mix(in_srgb,var(--safelight)_10%,transparent)] hover:brightness-110"
          @click="toasts.runAction(toast.id)"
        >
          {{ toast.action.label }}
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>
