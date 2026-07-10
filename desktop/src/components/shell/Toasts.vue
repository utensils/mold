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
      <button
        v-for="toast in toasts.items"
        :key="toast.id"
        type="button"
        class="border-edge pointer-events-auto rounded-chrome border bg-bench px-3 py-2 text-left text-body shadow-raised"
        :class="toast.kind === 'error' ? 'text-stop' : 'text-ink'"
        :role="toast.kind === 'error' ? 'alert' : 'status'"
        :aria-live="toast.kind === 'error' ? 'assertive' : 'polite'"
        title="Dismiss"
        @click="toasts.dismiss(toast.id)"
      >
        {{ toast.message }}
      </button>
    </TransitionGroup>
  </div>
</template>
