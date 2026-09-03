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
    <h1 class="font-sans font-semibold text-xl font-bold text-fg" style="font-stretch: 90%">
      {{ headline }}
    </h1>
    <p v-if="detail" class="max-w-md text-sm text-fg-2">{{ detail }}</p>
    <button
      v-if="action"
      type="button"
      class="mt-2 h-9 rounded-window bg-accent px-4 text-sm font-semibold text-on-accent transition-[filter] duration-100 hover:brightness-105 active:translate-y-px"
      @click="$emit('action')"
    >
      {{ action }}
    </button>
  </div>
</template>
