<script setup lang="ts">
/*
 * Web host for the shared ReferenceCropEditor (spec §05 overlays): the @ui
 * ModalPanel inside a viewport-fixed host — the same pattern ImagePickerModal
 * uses, because Create is a long scrolling column and the panel fills its
 * nearest positioned ancestor — plus `useOverlayFocus` for the Tab trap and
 * opener restoration.
 */
import { computed, ref } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import ReferenceCropEditor from "@studio/components/ReferenceCropEditor.vue";
import type { ReferenceCrop } from "@studio/lib/referenceCrop";
import { useOverlayFocus } from "../composables/useOverlayFocus";

const props = defineProps<{
  open: boolean;
  title: string;
  image: {
    data: string;
    mimeType: string;
    width: number;
    height: number;
  } | null;
  crop: ReferenceCrop | null;
}>();

const emit = defineEmits<{
  apply: [crop: ReferenceCrop | null];
  close: [];
}>();

const host = ref<HTMLElement | null>(null);
const isOpen = computed(() => props.open && props.image !== null);
const { onKeydown } = useOverlayFocus(isOpen, host, () => emit("close"));
</script>

<template>
  <div
    v-if="isOpen && image"
    ref="host"
    class="rc-host"
    data-test="reference-crop-host"
    @keydown="onKeydown"
  >
    <ModalPanel :open="true" :width="720" :label="title" @close="emit('close')">
      <h2 class="rc__title">{{ title }}</h2>
      <ReferenceCropEditor
        :image="image"
        :crop="crop"
        @apply="emit('apply', $event)"
        @cancel="emit('close')"
      />
    </ModalPanel>
  </div>
</template>

<style scoped>
.rc-host {
  position: fixed;
  inset: 0;
  z-index: 40;
}
.rc__title {
  margin: 0 0 12px;
  font-size: var(--text-body-lg, 15px);
  font-weight: 600;
}
</style>
