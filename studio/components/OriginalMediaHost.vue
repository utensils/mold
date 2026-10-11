<script setup lang="ts">
import { onBeforeUnmount } from "vue";
import OriginalMediaViewer from "./OriginalMediaViewer.vue";
import {
  originalMediaPreview,
  dismissOriginalMedia,
} from "../lib/originalMediaPreview";
onBeforeUnmount(() => dismissOriginalMedia());
</script>
<template>
  <OriginalMediaViewer
    v-if="originalMediaPreview"
    :key="originalMediaPreview.id"
    :data="originalMediaPreview.data"
    :mime-type="originalMediaPreview.mimeType"
    :label="originalMediaPreview.label"
    :load="originalMediaPreview.load"
    :authority="originalMediaPreview.authority"
    @close="dismissOriginalMedia()"
  />
</template>
