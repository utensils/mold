<script setup lang="ts">
// Empty states are invitations: a display headline, one sentence, one action.
defineProps<{
  headline: string;
  detail?: string;
  action?: string;
}>();
defineEmits<{ action: [] }>();
</script>

<template>
  <div class="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
    <h1 class="font-display text-display-lg font-bold text-ink" style="font-stretch: 90%">
      {{ headline }}
    </h1>
    <p v-if="detail" class="max-w-md text-body text-ink-2">{{ detail }}</p>
    <button
      v-if="action"
      type="button"
      class="mt-2 h-9 rounded-chrome bg-safelight px-4 text-body font-semibold text-on-accent transition-[filter] duration-100 hover:brightness-105 active:translate-y-px"
      @click="$emit('action')"
    >
      {{ action }}
    </button>
  </div>
</template>
