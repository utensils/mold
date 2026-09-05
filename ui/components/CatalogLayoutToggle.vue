<script setup lang="ts">
import Icon from "./Icon.vue";

export type CatalogLayoutChoice = "list" | "table" | "grid";

const props = withDefaults(
  defineProps<{
    modelValue: CatalogLayoutChoice;
    listValue?: "list" | "table";
    listLabel?: string;
    label?: string;
  }>(),
  {
    listValue: "list",
    listLabel: "List",
    label: "Catalog layout",
  },
);

const emit = defineEmits<{
  "update:modelValue": [value: CatalogLayoutChoice];
}>();
</script>

<template>
  <div class="ms-layout" role="radiogroup" :aria-label="label">
    <button
      type="button"
      role="radio"
      class="ms-layout__option"
      :data-on="modelValue === props.listValue ? 'true' : undefined"
      :data-test="`layout-${props.listValue}`"
      :aria-checked="modelValue === props.listValue"
      @click="emit('update:modelValue', props.listValue)"
    >
      <Icon name="list" :size="13" />
      {{ props.listLabel }}
    </button>
    <button
      type="button"
      role="radio"
      class="ms-layout__option"
      :data-on="modelValue === 'grid' ? 'true' : undefined"
      data-test="layout-grid"
      :aria-checked="modelValue === 'grid'"
      @click="emit('update:modelValue', 'grid')"
    >
      <Icon name="grid" :size="13" />
      Grid
    </button>
  </div>
</template>

<style scoped>
.ms-layout {
  display: flex;
  align-items: center;
  gap: 2px;
  padding: 3px;
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
}

.ms-layout__option {
  display: inline-flex;
  min-height: 28px;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 0 9px;
  border: 0;
  border-radius: var(--mold-radius-1);
  background: transparent;
  color: var(--mold-text-dim);
  font-family: var(--mold-font-sans);
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition:
    color var(--mold-dur-quick) var(--mold-ease-out),
    background var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-layout__option:hover:not([data-on="true"]) {
  color: var(--mold-text);
}

.ms-layout__option[data-on="true"] {
  color: var(--mold-blue);
  background: var(--mold-accent-tint);
  box-shadow: inset 0 0 0 1px var(--mold-blue);
}

.ms-layout__option:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}
</style>
