<script setup lang="ts">
import Icon from "@ui/components/Icon.vue";
import { MOBILE_TABS, type MobileTab } from "./navigation";

defineProps<{ modelValue: MobileTab; queueCount: number }>();
defineEmits<{ "update:modelValue": [tab: MobileTab] }>();
</script>

<template>
  <nav class="mobile-tabs" aria-label="Primary">
    <button
      v-for="item in MOBILE_TABS"
      :key="item.id"
      class="mobile-tab"
      type="button"
      :aria-current="modelValue === item.id ? 'page' : undefined"
      :data-test="`mobile-tab-${item.id}`"
      @click="$emit('update:modelValue', item.id)"
    >
      <span class="mobile-tab-icon">
        <Icon :name="item.icon" :size="22" />
        <span
          v-if="item.id === 'queue' && queueCount > 0"
          class="mobile-tab-dot"
          aria-hidden="true"
        />
      </span>
      <span>{{ item.label }}</span>
      <span v-if="item.id === 'queue' && queueCount > 0" class="sr-only"
        >{{ queueCount }} items</span
      >
    </button>
  </nav>
</template>
