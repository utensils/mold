<script setup lang="ts">
import { ref, watch } from "vue";
import ErrorNotice from "@ui/components/ErrorNotice.vue";

const props = withDefaults(
  defineProps<{ message: string; copyMessage?: string | null; dismissible?: boolean }>(),
  {
    copyMessage: null,
    dismissible: true,
  },
);

const dismissed = ref(false);

watch(
  () => props.message,
  () => {
    dismissed.value = false;
  },
);
</script>

<template>
  <ErrorNotice
    v-if="!dismissed"
    compact
    :dismissible="dismissible"
    :message="message"
    :copy-message="copyMessage"
    @dismiss="dismissed = true"
  >
    <template v-if="$slots.actions" #actions>
      <slot name="actions" />
    </template>
  </ErrorNotice>
</template>
