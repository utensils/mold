<script setup lang="ts">
// C1 STUB — replaced at merge by lane C1's MakeChip.vue.
/*
 * Composer count chip — the mock's `Make 4 ▼`. It is the ONE batch control on
 * Create; the rail keeps no second stepper. `locked` is the page's own
 * batch-lock rule (an edit recipe renders one print at a time), never a
 * family name decided here.
 */
import { computed } from "vue";
import Stepper from "@ui/components/Stepper.vue";

const props = withDefaults(
  defineProps<{
    modelValue: number;
    min?: number;
    max?: number;
    disabled?: boolean;
    locked?: boolean;
    lockedReason?: string;
  }>(),
  { min: 1, max: 10_000, disabled: false, locked: false, lockedReason: "" },
);

const emit = defineEmits<{ "update:modelValue": [value: number] }>();

const shown = computed(() => (props.locked ? 1 : props.modelValue));
</script>

<template>
  <div class="make" data-test="composer-make-chip">
    <span class="make__label">Make</span>
    <Stepper
      :model-value="shown"
      :min="min"
      :max="locked ? 1 : max"
      editable
      label="Batch size"
      @update:model-value="emit('update:modelValue', $event)"
    />
    <span
      v-if="locked && lockedReason"
      class="make__note"
      data-test="make-chip-locked"
    >
      {{ lockedReason }}
    </span>
  </div>
</template>

<style scoped>
.make {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: var(--mold-ctl-md, 28px);
}
.make__label {
  font-size: 12px;
  color: var(--ink-2);
}
.make__note {
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
}
</style>
