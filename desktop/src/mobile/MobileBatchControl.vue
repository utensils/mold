<script setup lang="ts">
import { computed, watch } from "vue";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import { generationCapabilitiesForFamily } from "../lib/capabilities";
import type { ModelEntry } from "../lib/api/types";
import type { GenerateForm } from "../lib/generateForm";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    selectedModel?: ModelEntry | null;
  }>(),
  { selectedModel: null },
);

const MAX_BATCH_SIZE = 10_000;
const caps = computed(() =>
  generationCapabilitiesForFamily(
    props.form.family,
    props.form.model,
    props.form.pipeline,
    props.selectedModel?.guidance_capabilities,
    props.selectedModel?.source_image ?? props.form.sourceImageCapability,
    effectiveGenerationRecipe(props.selectedModel, props.form.pipeline),
  ),
);
const batchLocked = computed(
  () =>
    caps.value.forcesBatchSizeOne ||
    (caps.value.sourceImageMode === "references" && props.form.imageAttachments.length > 0),
);

watch(
  () => [batchLocked.value, props.form.batchSize] as const,
  ([locked, batchSize]) => {
    const normalized = locked
      ? 1
      : Math.min(MAX_BATCH_SIZE, Math.max(1, Math.round(batchSize) || 1));
    if (batchSize !== normalized) props.form.batchSize = normalized;
  },
  { immediate: true },
);

function stepBatch(delta: -1 | 1): void {
  if (batchLocked.value) return;
  props.form.batchSize = Math.min(MAX_BATCH_SIZE, Math.max(1, props.form.batchSize + delta));
}

function setBatch(raw: string): void {
  if (batchLocked.value) return;
  const value = Number(raw);
  props.form.batchSize = Number.isFinite(value)
    ? Math.min(MAX_BATCH_SIZE, Math.max(1, Math.round(value)))
    : 1;
}
</script>

<template>
  <section class="mobile-batch-control" data-test="mobile-batch-control">
    <div class="mobile-generate-label-row">
      <span class="mobile-generate-label">Batch</span>
      <span v-if="batchLocked" class="mobile-generate-note" data-test="mobile-batch-locked">
        Edit models render one at a time
      </span>
    </div>
    <div class="mobile-generate-stepper" role="group" aria-label="Batch size">
      <button
        type="button"
        class="mobile-generate-stepper-button"
        data-test="mobile-batch-decrement"
        aria-label="Decrease batch size"
        :disabled="batchLocked || form.batchSize <= 1"
        @click="stepBatch(-1)"
      >
        −
      </button>
      <input
        class="mobile-generate-stepper-value"
        data-test="mobile-batch-value"
        type="number"
        inputmode="numeric"
        min="1"
        step="1"
        aria-label="Batch size"
        :value="batchLocked ? 1 : form.batchSize"
        :disabled="batchLocked"
        @change="setBatch(($event.target as HTMLInputElement).value)"
      />
      <button
        type="button"
        class="mobile-generate-stepper-button"
        data-test="mobile-batch-increment"
        aria-label="Increase batch size"
        :disabled="batchLocked || form.batchSize >= MAX_BATCH_SIZE"
        @click="stepBatch(1)"
      >
        +
      </button>
    </div>
  </section>
</template>
