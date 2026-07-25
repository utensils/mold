<script setup lang="ts">
import { computed } from "vue";
import { useToastStore } from "../../stores/toasts";

const toasts = useToastStore();

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
        class="border-edge pointer-events-auto flex items-center gap-2 rounded-chrome border bg-bench px-3 py-2 text-body shadow-raised"
        :class="toast.kind === 'error' ? 'text-stop' : 'text-ink'"
        :role="toast.kind === 'error' ? 'alert' : 'status'"
        :aria-live="toast.kind === 'error' ? 'assertive' : 'polite'"
        @click.self="toasts.click(toast.id)"
      >
        <button
          type="button"
          class="min-w-0 flex-1 truncate text-left"
          :title="toast.onClick ? undefined : 'Dismiss'"
          @click="toasts.click(toast.id)"
        >
          {{ toast.message }}
        </button>
        <button
          v-if="toast.action"
          type="button"
          data-test="toast-action"
          class="shrink-0 rounded-control px-2 py-0.5 text-caption font-semibold text-safelight hover:brightness-110"
          @click="toasts.runAction(toast.id)"
        >
          {{ toast.action.label }}
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>
