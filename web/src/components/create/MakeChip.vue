<script setup lang="ts">
/*
 * The composer's Make chip — "Make 4 ▼" over a small stepper popover.
 *
 * Make is a QUEUE decision, not a slider: every print in a batch is its own
 * job on the machine, so the menu says so under the stepper rather than
 * leaving a large number looking like one longer render. A recipe that can
 * only render one at a time (an edit model) states the one it will make and
 * carries the recipe's own reason, instead of offering a count admission
 * would refuse.
 */
import { computed, ref } from "vue";
import Popover from "@ui/components/Popover.vue";
import Stepper from "@ui/components/Stepper.vue";

const props = withDefaults(
  defineProps<{
    modelValue: number;
    min?: number;
    /** Omit for a control with no product-imposed upper bound. */
    max?: number | undefined;
    disabled?: boolean;
    /** The recipe renders one at a time: the chip reads 1 and locks. */
    locked?: boolean;
    lockedReason?: string | null;
  }>(),
  {
    min: 1,
    max: undefined,
    disabled: false,
    locked: false,
    lockedReason: null,
  },
);

const emit = defineEmits<{ "update:modelValue": [value: number] }>();

const open = ref(false);
const inert = computed(() => props.locked || props.disabled);
const count = computed(() => (props.locked ? 1 : props.modelValue));
const title = computed(() =>
  props.locked ? (props.lockedReason ?? undefined) : "How many to make",
);

function toggle() {
  if (inert.value) return;
  open.value = !open.value;
}
</script>

<template>
  <Popover
    v-model:open="open"
    class="make-chip__anchor"
    placement="top-start"
    label="How many to make"
  >
    <template #trigger>
      <button
        type="button"
        class="make-chip"
        data-test="make-chip"
        aria-haspopup="dialog"
        :aria-expanded="open"
        :disabled="inert"
        :title="title"
        @click="toggle"
      >
        Make {{ count }}
        <span class="make-chip__caret" aria-hidden="true">▼</span>
      </button>
    </template>
    <div class="make-chip__menu" data-test="make-menu">
      <Stepper
        :model-value="modelValue"
        :min="min"
        :max="max"
        editable
        label="How many to make"
        @update:model-value="emit('update:modelValue', $event)"
      />
      <p class="make-chip__note">Each one is queued separately</p>
    </div>
  </Popover>
</template>

<style scoped>
.make-chip__anchor {
  flex-shrink: 0;
}

.make-chip {
  /* literal: the mock's 28px chip row, the one metric between --mold-ctl-sm
   * (24px, dense icon buttons) and --mold-ctl-md (26px, toolbars). */
  --composer-chip-h: 28px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: var(--composer-chip-h);
  padding: 0 10px;
  flex-shrink: 0;
  white-space: nowrap;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
  cursor: pointer;
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}

.make-chip:hover:not(:disabled) {
  border-color: var(--mold-border-focus);
  color: var(--mold-text);
}

.make-chip:disabled {
  color: var(--mold-text-dim);
  cursor: default;
}

.make-chip__caret {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}

.make-chip__menu {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
}

.make-chip__note {
  margin: 0;
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim);
}
</style>
