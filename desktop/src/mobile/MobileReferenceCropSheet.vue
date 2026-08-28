<script setup lang="ts">
/*
 * Reference crop sheet — the iPhone presentation of the shared
 * @studio ReferenceCropEditor. Like MobileSeamSheet (and deliberately NOT
 * @ui/SheetPanel, whose `position: absolute; inset: 0` renders off-screen in
 * the scrolling content column): a fixed overlay whose body owns its own
 * scroll and every safe-area inset.
 */
import ReferenceCropEditor from "@studio/components/ReferenceCropEditor.vue";
import type { ReferenceCrop } from "@studio/lib/referenceCrop";

defineProps<{
  open: boolean;
  title: string;
  image: { data: string; mimeType: string; width: number; height: number } | null;
  crop: ReferenceCrop | null;
}>();

const emit = defineEmits<{
  apply: [crop: ReferenceCrop | null];
  close: [];
}>();
</script>

<template>
  <div
    class="mobile-crop-sheet"
    :class="{ 'is-open': open && image }"
    role="dialog"
    :aria-label="title"
    :aria-hidden="open && image ? undefined : 'true'"
    data-test="mobile-crop-sheet"
  >
    <button
      class="mobile-crop-sheet-backdrop"
      type="button"
      aria-label="Close crop"
      data-test="mobile-crop-sheet-backdrop"
      @click="emit('close')"
    />
    <div class="mobile-crop-sheet-panel">
      <span class="mobile-crop-sheet-grabber" aria-hidden="true" />
      <div class="mobile-crop-sheet-body">
        <p class="mobile-crop-sheet-head" data-test="mobile-crop-sheet-head">
          {{ title }}
        </p>
        <ReferenceCropEditor
          v-if="open && image"
          large
          :image="image"
          :crop="crop"
          @apply="emit('apply', $event)"
          @cancel="emit('close')"
        />
      </div>
    </div>
  </div>
</template>
