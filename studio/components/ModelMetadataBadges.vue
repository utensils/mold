<script setup lang="ts">
import { computed } from "vue";
import BadgePill from "@ui/components/BadgePill.vue";
import {
  modelKindLabel,
  modelKindValue,
  modelModalityLabel,
} from "../lib/modelMetadata";

const props = withDefaults(
  defineProps<{
    kind?: string | null;
    family?: string | null;
    modality?: string | null;
    nsfw?: boolean | null;
    showModality?: boolean;
  }>(),
  {
    kind: null,
    family: null,
    modality: null,
    nsfw: null,
    showModality: true,
  },
);

const kindLabel = computed(() =>
  props.kind || props.family
    ? modelKindLabel(modelKindValue({ kind: props.kind, family: props.family }))
    : null,
);
const modalityLabel = computed(() =>
  props.modality ? modelModalityLabel(props.modality) : null,
);
</script>

<template>
  <span class="model-metadata-badges">
    <BadgePill
      v-if="showModality && modalityLabel"
      tone="info"
      outline
      data-test="model-modality-badge"
    >
      {{ modalityLabel }}
    </BadgePill>
    <BadgePill
      v-if="kindLabel"
      tone="neutral"
      outline
      data-test="model-kind-badge"
    >
      {{ kindLabel }}
    </BadgePill>
    <BadgePill
      v-if="nsfw"
      tone="danger"
      data-test="model-nsfw-badge"
      aria-label="Mature content (NSFW)"
      title="Mature content"
    >
      18+ NSFW
    </BadgePill>
  </span>
</template>

<style scoped>
.model-metadata-badges {
  display: inline-flex;
  min-width: 0;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}
</style>
