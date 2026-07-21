<script setup lang="ts">
import { computed } from "vue";
import { ICONS, type IconName } from "../icons";

const props = withDefaults(
  defineProps<{
    name: IconName;
    /** Icon sizes from the spec: 14 / 16 / 17 / 22 (px). */
    size?: number;
    strokeWidth?: number;
  }>(),
  { size: 17, strokeWidth: 1.7 },
);

const markup = computed(() => ICONS[props.name]);
</script>

<template>
  <svg
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    :stroke-width="strokeWidth"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
    v-html="markup"
  />
</template>
