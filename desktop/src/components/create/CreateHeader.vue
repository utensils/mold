<script setup lang="ts">
import { computed } from "vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { GenerateForm } from "../../lib/generateForm";
import { outputFamilyLabel } from "@studio/lib/outputShape";
import HostChip from "./HostChip.vue";

/**
 * Create header (Mold Studio): the print/sequence title, a live summary
 * pill, and the shared generation-host chip. Output (One shot | Sequence)
 * is a setting in the inspector, not a place — the old Single | Sequence
 * route switch is gone.
 */
const props = defineProps<{ form: GenerateForm }>();

const draft = useSequenceDraftStore();
const isSequence = computed(() => draft.output === "sequence");

const title = computed(() => (isSequence.value ? "Untitled sequence" : "Untitled print"));

const summary = computed(() => {
  const { width, height, steps } = props.form;
  const aspect = outputFamilyLabel(width, height);
  if (isSequence.value) {
    return `${aspect} · ${width}×${height} · ${draft.clips.length} clips · ${props.form.fps} fps`;
  }
  return `${aspect} · ${width}×${height} · ${steps} steps`;
});
</script>

<template>
  <header data-test="create-header" class="ms-header">
    <span class="ms-header__title">{{ title }}</span>
    <span class="ms-header__summary data-mono">{{ summary }}</span>
    <div class="ms-header__spacer" />
    <HostChip />
  </header>
</template>

<style scoped>
.ms-header {
  height: 52px;
  flex: 0 0 52px;
  border-bottom: 1px solid var(--edge);
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 22px;
}
.ms-header__title {
  font-family: var(--f-display);
  font-size: 15px;
  font-weight: 600;
}
.ms-header__summary {
  font-size: 10px;
  color: var(--ink-3);
  padding: 3px 8px;
  border: 1px solid var(--edge);
  border-radius: 20px;
}
.ms-header__spacer {
  flex: 1;
}
</style>
