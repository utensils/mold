<script setup lang="ts">
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
      class="fixed inset-0 z-50 flex justify-center bg-scrim pt-24"
      @click.self="emit('cancel')"
    >
      <div
        class="border-border h-fit w-72 rounded-window border bg-bg p-3 shadow-xl"
        role="dialog"
        aria-modal="true"
        :aria-label="title"
      >
        <p class="mb-2 text-micro font-medium text-fg-2">{{ title }}</p>
        <input
          ref="inputEl"
          v-model="value"
          data-selectable
          type="text"
          class="border-border w-full rounded-control border bg-transparent px-2 py-1 text-sm text-fg outline-none focus:border-accent"
          @keydown.enter.prevent="save"
          @keydown.esc.prevent="emit('cancel')"
        />
        <div class="mt-2 flex justify-end gap-2">
          <button
            type="button"
            class="h-7 rounded-control px-2 text-micro text-fg-2 hover:text-fg"
            @click="emit('cancel')"
          >
            Cancel
          </button>
          <button
            type="button"
            data-test="rename-save"
            class="h-7 rounded-control bg-accent px-3 text-micro font-semibold text-on-accent hover:brightness-105 active:translate-y-px"
            @click="save"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
