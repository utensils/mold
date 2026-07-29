<script setup lang="ts">
import ErrorNotice from "@ui/components/ErrorNotice.vue";

withDefaults(defineProps<{ message: string; copyMessage?: string | null }>(), {
  copyMessage: null,
});
</script>

<template>
  <ErrorNotice :message="message" :copy-message="copyMessage">
    <template v-if="$slots.actions" #actions>
      <slot name="actions" />
    </template>
  </ErrorNotice>
</template>
