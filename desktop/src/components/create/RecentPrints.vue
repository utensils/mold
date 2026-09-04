<script setup lang="ts">
import { computed } from "vue";
import { displayTitle } from "@studio/lib/libraryOrganization";
import AuthedMedia from "../gallery/AuthedMedia.vue";
import { galleryMediaPath } from "../../lib/gallery/media";
import { modelDisplayNameForId, type DisplayableModel } from "../../lib/models";
import { useGalleryStore, type MergedPrint } from "../../stores/gallery";

/**
 * The Recent tab: the pictures already made, newest first, each restoring the
 * whole recipe on click. The door that opens this tab says "Use these settings
 * again", so a row hands the print back to the same reuse path the Lightbox
 * uses — never a bare prompt string.
 *
 * There is no "took 4.0s" beside the style: `OutputMetadata` records no
 * generation time, so the mono line says only what the print actually knows.
 */
const props = defineProps<{
  prints: readonly MergedPrint[];
  models?: DisplayableModel[] | undefined;
}>();
const emit = defineEmits<{ reuse: [print: MergedPrint] }>();

const gallery = useGalleryStore();

const rows = computed(() =>
  props.prints.map((entry) => ({
    entry,
    key: `${entry.sourceKey}::${entry.item.filename}`,
    title: displayTitle(entry.item),
    meta: modelDisplayNameForId(entry.item.metadata.model, props.models ?? []),
    path: galleryMediaPath(entry.item.filename, gallery.mediaSourceOf(entry.sourceKey), true),
    target: gallery.targetOf(entry.sourceKey),
    mediaVersion:
      entry.item.media_version ?? `${entry.item.timestamp}:${entry.item.size_bytes ?? "unknown"}`,
  })),
);
</script>

<template>
  <div data-test="recent-prints">
    <button
      v-for="row in rows"
      :key="row.key"
      type="button"
      class="ms-recent"
      data-test="recent-print"
      :title="row.title"
      @click="emit('reuse', row.entry)"
    >
      <span class="ms-recent__thumb">
        <AuthedMedia
          :path="row.path"
          :target="row.target"
          :cache-key="row.entry.sourceKey"
          :media-version="row.mediaVersion"
          :alt="row.title"
        />
      </span>
      <span class="ms-recent__body">
        <span class="ms-recent__title">{{ row.title }}</span>
        <span class="ms-recent__meta">{{ row.meta }}</span>
      </span>
    </button>
  </div>
</template>

<style scoped>
.ms-recent {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 9px;
  margin-bottom: 8px;
  padding: 8px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: transparent;
  text-align: left;
  cursor: pointer;
}
.ms-recent:hover {
  background: var(--mold-row-hover);
}
.ms-recent__thumb {
  display: block;
  width: 36px;
  height: 36px;
  flex-shrink: 0;
  overflow: hidden;
  border: var(--mold-bw) solid var(--mold-border);
  background: var(--mold-bg-crust);
}
.ms-recent__thumb :deep(img) {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.ms-recent__body {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  gap: 2px;
}
.ms-recent__title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--mold-fs-micro);
  font-weight: 500;
  color: var(--mold-text);
}
.ms-recent__meta {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
</style>
