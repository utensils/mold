<script setup lang="ts">
import { onBeforeUnmount, watch } from "vue";
import {
  inspectOriginalMedia,
  dismissOriginalMedia,
} from "../lib/originalMediaPreview";
const props = defineProps<{
  data?: string | null | undefined;
  mimeType?: string | null | undefined;
  label?: string | undefined;
  load?: ((signal: AbortSignal) => Promise<Blob>) | undefined;
  authority?: string | undefined;
  actionLabel?: string;
}>();
const owner = Symbol("source inspection");
function show(event: MouseEvent) {
  inspectOriginalMedia({
    ...props,
    owner,
    opener: event.currentTarget as HTMLElement,
  });
}
watch([() => props.data, () => props.mimeType, () => props.authority], () =>
  dismissOriginalMedia(owner),
);
onBeforeUnmount(() => dismissOriginalMedia(owner));
</script>
<template>
  <button
    v-if="data || load"
    type="button"
    class="original-media__view"
    @click.stop="show"
  >
    {{ actionLabel || "View original" }}
  </button>
</template>
<style scoped>
.original-media__view {
  min-height: 48px;
  min-width: 48px;
  padding: 6px 10px;
  border: var(--mold-bw, 1px) solid var(--mold-border, #999);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg, transparent);
  color: var(--mold-text, inherit);
  cursor: pointer;
  font: inherit;
}
</style>
