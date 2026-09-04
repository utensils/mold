<script setup lang="ts">
/* The one-field dialog: a name, Cancel, Save. */
import { nextTick, ref, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";

const props = defineProps<{ open: boolean; title: string; initial: string }>();
const emit = defineEmits<{ save: [name: string]; cancel: [] }>();

const value = ref("");
const inputEl = ref<HTMLInputElement | null>(null);

watch(
  () => props.open,
  (open) => {
    if (!open) return;
    value.value = props.initial;
    void nextTick(() => inputEl.value?.select());
  },
  { immediate: true },
);

/** An empty name is a cancel, not a save — hosts always keep some label. */
function save() {
  const trimmed = value.value.trim();
  if (trimmed) emit("save", trimmed);
  else emit("cancel");
}
</script>

<template>
  <ModalPanel
    :open="open"
    :width="480"
    :title="title"
    data-test="rename-dialog"
    @close="emit('cancel')"
  >
    <input
      ref="inputEl"
      v-model="value"
      data-selectable
      type="text"
      class="h-8 w-full rounded-control border border-border bg-bg px-2.5 text-sm text-fg outline-none focus:border-border-focus"
      @keydown.enter.prevent="save"
    />
    <template #footer>
      <button
        type="button"
        class="min-h-8 rounded-control border border-border px-3.5 py-1.5 text-xs text-fg-2 transition-colors duration-100 hover:border-border-focus hover:text-fg"
        @click="emit('cancel')"
      >
        Cancel
      </button>
      <button
        type="button"
        data-test="rename-save"
        class="min-h-8 rounded-control bg-accent px-3.5 py-1.5 text-xs font-semibold text-on-accent hover:brightness-105 active:translate-y-px"
        @click="save"
      >
        Save
      </button>
    </template>
  </ModalPanel>
</template>
