<script setup lang="ts">
/*
 * Desktop host for the shared ReferenceCropEditor — the same viewport-fixed
 * dialog shell the sibling ImagePickerModal / MaskEditorModal use, so the
 * crop opens over the whole window rather than inside the inspector column.
 */
import { nextTick, ref, watch } from "vue";
import ReferenceCropEditor from "@studio/components/ReferenceCropEditor.vue";
import type { ReferenceCrop } from "@studio/lib/referenceCrop";

const props = defineProps<{
  open: boolean;
  title: string;
  image: { data: string; mimeType: string; width: number; height: number } | null;
  crop: ReferenceCrop | null;
}>();

const emit = defineEmits<{
  apply: [crop: ReferenceCrop | null];
  close: [];
}>();

const dialog = ref<HTMLElement | null>(null);
watch(
  () => props.open,
  (open) => {
    if (open) void nextTick(() => dialog.value?.focus());
  },
);
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open && image"
      class="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-4"
      data-test="reference-crop-modal"
      @click.self="emit('close')"
      @keydown.esc="emit('close')"
    >
      <div
        ref="dialog"
        class="border-border flex max-h-[90vh] w-full max-w-3xl flex-col overflow-y-auto rounded-window border bg-bg p-5"
        role="dialog"
        aria-modal="true"
        :aria-label="title"
        tabindex="-1"
      >
        <h2 class="text-base mb-3 font-semibold text-fg">{{ title }}</h2>
        <ReferenceCropEditor
          :image="image"
          :crop="crop"
          @apply="emit('apply', $event)"
          @cancel="emit('close')"
        />
      </div>
    </div>
  </Teleport>
</template>
