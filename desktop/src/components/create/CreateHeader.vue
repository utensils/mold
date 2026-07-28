<script setup lang="ts">
import { computed } from "vue";
import { useRouter } from "vue-router";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import type { GenerateForm } from "../../lib/generateForm";
import { aspectRatioLabel } from "../../lib/resolutions";
import HostChip from "./HostChip.vue";

/**
 * Create header (Mold Studio): the print title, a live summary pill, the
 * Single | Sequence composer switch, and the shared generation-host chip.
 */
const props = defineProps<{ form: GenerateForm }>();

const router = useRouter();

/** Create is the Single side of the composer; Sequence lives at /create/chain. */
function setComposerMode(mode: string | number) {
  if (mode === "sequence") void router.push("/create/chain");
}

const summary = computed(() => {
  const { width, height, steps, family } = props.form;
  return `${aspectRatioLabel(width, height, family)} · ${width}×${height} · ${steps} steps`;
});
</script>

<template>
  <header data-test="create-header" class="ms-header">
    <span class="ms-header__title">Untitled print</span>
    <span class="ms-header__summary data-mono">{{ summary }}</span>
    <SegmentedControl
      model-value="single"
      compact
      :options="[
        { value: 'single', label: 'Single' },
        { value: 'sequence', label: 'Sequence' },
      ]"
      label="Composer mode"
      data-test="composer-mode"
      @update:model-value="setComposerMode"
    />
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
