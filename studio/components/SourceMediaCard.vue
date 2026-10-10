<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { retainedSourceMediaThumbnail } from "../api/sourceMediaBrowser";
import {
  sourceMemberKind,
  sourceMemberLabel,
  sourceOwnerIdentity,
  type SourceMediaSelection,
} from "../lib/sourceMediaBrowser";
const props = defineProps<{ selection: SourceMediaSelection }>();
const emit = defineEmits<{ use: [selection: SourceMediaSelection] }>();
const root = ref<HTMLElement>();
const visible = ref(false);
const url = ref<string>();
const kind = computed(() => sourceMemberKind(props.selection.member));
const label = computed(() => sourceMemberLabel(props.selection.member));
let observer: IntersectionObserver | undefined;
onMounted(() => {
  if (typeof IntersectionObserver === "undefined") {
    visible.value = true;
    return;
  }
  observer = new IntersectionObserver(([entry]) => {
    if (entry?.isIntersecting) {
      visible.value = true;
      observer?.disconnect();
    }
  });
  if (root.value) observer.observe(root.value);
});
watch(
  [
    () => sourceOwnerIdentity(props.selection.owner),
    () => props.selection.member.member_id,
    visible,
  ],
  async (_, __, cleanup) => {
    const controller = new AbortController();
    let ownedURL: string | undefined;
    url.value = undefined;
    cleanup(() => {
      controller.abort();
      if (ownedURL) URL.revokeObjectURL(ownedURL);
    });
    if (!visible.value || kind.value !== "image") return;
    const { owner, member } = props.selection;
    try {
      const blob = await retainedSourceMediaThumbnail(
        owner.filename,
        member,
        owner.target,
        controller.signal,
      );
      if (controller.signal.aborted) return;
      ownedURL = URL.createObjectURL(blob);
      url.value = ownedURL;
    } catch {
      /* The role tile remains usable even when its preview is unavailable. */
    }
  },
  { immediate: true },
);
onBeforeUnmount(() => observer?.disconnect());
</script>
<template>
  <article ref="root" class="source-card" data-test="source-media-card">
    <div class="source-card__preview">
      <img v-if="url" :src="url" :alt="label" /><span v-else>{{
        kind === "image"
          ? "Preview unavailable"
          : kind === "unknown"
            ? "Original file"
            : kind
      }}</span>
    </div>
    <strong>{{ label }}</strong>
    <span class="source-card__name" :title="selection.member.display_name">{{
      selection.member.display_name
    }}</span>
    <span
      v-if="selection.member.role.startsWith('matting_processed')"
      class="source-card__notice"
      >Processed image. Reuse adds it as a new input.</span
    >
    <button
      v-if="['image', 'video', 'audio'].includes(kind)"
      type="button"
      @click="emit('use', selection)"
    >
      Use as source
    </button>
    <span v-else class="source-card__notice"
      >This file type cannot be attached to a new creation.</span
    >
  </article>
</template>
<style scoped>
.source-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  border: 1px solid var(--mold-border, #7775);
  border-radius: var(--mold-radius-3);
  padding: 10px;
  background: var(--mold-surface, #8881);
  font-size: var(--mold-fs-xs);
}
.source-card__preview {
  aspect-ratio: 4/3;
  display: grid;
  place-items: center;
  background: var(--mold-bg, #8881);
  border-radius: var(--mold-radius-2);
  overflow: hidden;
  color: var(--mold-text-dim, #888);
}
.source-card img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.source-card__name {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--mold-text-dim, #888);
}
.source-card button {
  min-height: 44px;
  border: 1px solid var(--mold-border, #7775);
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: inherit;
  cursor: pointer;
  margin-top: auto;
}
.source-card button:hover {
  background: var(--mold-accent-muted, #8882);
}
.source-card__notice {
  color: var(--mold-text-dim, #888);
}
</style>
