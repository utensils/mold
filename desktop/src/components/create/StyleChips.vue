<script setup lang="ts">
import { ref } from "vue";
import Chip from "@ui/components/Chip.vue";
import Icon from "@ui/components/Icon.vue";
import { STYLE_PRESETS, stylePresetLabel } from "../../lib/stylePresets";

/**
 * Composer style row (Mold Studio Create). A collapsible strip of look
 * presets; the selection is a prompt modifier composed at submit, so this only
 * owns the id. Single-select — clicking the active chip clears it.
 */
const props = defineProps<{ modelValue: string }>();
const emit = defineEmits<{ "update:modelValue": [value: string] }>();

const open = ref(false);

function pick(id: string) {
  emit("update:modelValue", props.modelValue === id ? "" : id);
}
</script>

<template>
  <div class="ms-styles">
    <button
      type="button"
      class="ms-styles__head"
      data-test="style-toggle"
      :aria-expanded="open"
      @click="open = !open"
    >
      <span class="ms-group-label uppercase">Style</span>
      <Chip :active="modelValue !== ''" data-test="style-active">{{
        stylePresetLabel(modelValue)
      }}</Chip>
      <span class="ms-styles__spacer" />
      <Icon :name="open ? 'chevron-up' : 'chevron-down'" :size="15" />
    </button>
    <div v-if="open" class="ms-styles__row">
      <Chip
        v-for="preset in STYLE_PRESETS"
        :key="preset.id"
        :active="modelValue === preset.id"
        :data-test="`style-${preset.id}`"
        @click="pick(preset.id)"
        >{{ preset.label }}</Chip
      >
    </div>
  </div>
</template>

<style scoped>
.ms-styles {
  padding: 6px 0 2px;
}
.ms-styles__head {
  display: flex;
  align-items: center;
  gap: 9px;
  width: 100%;
  border: 0;
  background: transparent;
  color: var(--mold-text-2);
  padding: 4px 0;
  text-align: left;
  cursor: pointer;
}
.ms-styles__spacer {
  flex: 1;
}
.ms-styles__row {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 9px;
}
</style>
