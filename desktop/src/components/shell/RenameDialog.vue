<script setup lang="ts">
/* The one-field dialog (README §04 dialog anatomy): a name, Cancel, Save. */
import { nextTick, ref, watch } from "vue";

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
  <Teleport to="body">
    <div
      v-if="open"
      data-test="rename-dialog"
      class="fixed inset-0 z-50 flex items-center justify-center bg-scrim p-10"
      @click.self="emit('cancel')"
    >
      <div
        class="ms-fade-up flex w-[360px] max-w-full flex-col overflow-hidden rounded-window border border-border bg-surface shadow-md"
        role="dialog"
        aria-modal="true"
        :aria-label="title"
      >
        <div class="border-b border-border p-4">
          <p class="text-md font-semibold text-fg">{{ title }}</p>
        </div>
        <div class="p-4">
          <input
            ref="inputEl"
            v-model="value"
            data-selectable
            type="text"
            class="h-8 w-full rounded-control border border-border bg-bg px-2.5 text-sm text-fg outline-none focus:border-border-focus"
            @keydown.enter.prevent="save"
            @keydown.esc.prevent="emit('cancel')"
          />
        </div>
        <div class="flex justify-end gap-2 border-t border-border px-4 py-3.5">
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
        </div>
      </div>
    </div>
  </Teleport>
</template>
