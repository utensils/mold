<script setup lang="ts">
import { useToastStore } from "../../stores/toasts";

const toasts = useToastStore();
</script>

<template>
  <div
    class="pointer-events-none fixed right-4 bottom-10 z-50 flex flex-col items-end gap-2"
    aria-label="Notifications"
  >
    <TransitionGroup
      enter-active-class="transition duration-150 ease-out"
      enter-from-class="opacity-0 translate-y-2"
      leave-active-class="transition duration-150"
      leave-to-class="opacity-0"
    >
      <div
        v-for="toast in toasts.items"
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
