<script setup lang="ts">
import { computed } from "vue";

const props = withDefaults(
  defineProps<{
    title: string;
    subtitle: string;
    status: string;
    detail?: string | null;
    cancelling?: boolean;
    ariaLabel?: string;
  }>(),
  {
    detail: null,
    cancelling: false,
  },
);

const emit = defineEmits<{
  activate: [];
}>();

const detailedStatus = computed(() => props.status.length > 18);
</script>

<template>
  <div
    class="mobile-generation-job"
    :class="{ 'mobile-generation-job--detailed-status': detailedStatus }"
    role="button"
    tabindex="0"
    :aria-label="ariaLabel"
    data-test="mobile-generation-queue-card"
    @click="emit('activate')"
    @keydown.enter.prevent="emit('activate')"
    @keydown.space.prevent="emit('activate')"
  >
    <div class="mobile-generation-job-copy">
      <p>{{ title }}</p>
      <span>{{ subtitle }}</span>
      <p
        v-if="detail"
        class="mobile-generation-held-error"
        data-test="mobile-generation-held-error"
      >
        {{ detail }}
      </p>
    </div>
    <div class="mobile-generation-job-action">
      <span data-test="mobile-generation-status">{{ status }}</span>
      <span v-if="cancelling" data-test="mobile-generation-cancelling"> Cancelling… </span>
    </div>
  </div>
</template>
