<script setup lang="ts">
/*
 * Model picker (Mold Studio Create) — the model selector that lives in the
 * controls region so it's reachable at every width (the old left rail was
 * xl-only, leaving tablet/narrow-desktop web with no model selector). A native
 * select grouped by family, a one-line description (family · footprint · load
 * state), and a link into the full Models workspace — mirroring the desktop
 * inspector's model picker.
 */
import { computed } from "vue";
import { RouterLink } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import { modelDisplayName } from "@mold/studio";
import type { ModelInfoExtended } from "../../types";
import { modelDisplayName } from "../../lib/modelName";

const props = defineProps<{
  models: ModelInfoExtended[];
  model: string;
}>();

const emit = defineEmits<{ select: [model: ModelInfoExtended] }>();

const current = computed(
  () => props.models.find((m) => m.name === props.model) ?? null,
);

// Group installed models by family for the <optgroup> headers.
const groups = computed(() => {
  const byFamily = new Map<string, ModelInfoExtended[]>();
  for (const m of props.models) {
    const list = byFamily.get(m.family) ?? [];
    list.push(m);
    byFamily.set(m.family, list);
  }
  return [...byFamily.entries()].map(([family, list]) => ({ family, list }));
});

const description = computed(() => {
  const m = current.value;
  if (!m) return "";
  const parts = [m.family];
  if (m.size_gb > 0) parts.push(`${m.size_gb.toFixed(1)} GB`);
  parts.push(m.is_loaded ? "loaded" : "on disk");
  return parts.join(" · ");
});

function onChange(event: Event) {
  const name = (event.target as HTMLSelectElement).value;
  const m = props.models.find((x) => x.name === name);
  if (m) emit("select", m);
}
</script>

<template>
  <div class="mp" data-test="create-model-picker">
    <div class="mp__head">
      <span class="mp__kicker">Model</span>
      <RouterLink to="/models" class="mp__browse">
        Browse all
        <Icon name="chevron-right" :size="12" />
      </RouterLink>
    </div>
    <select
      class="mp__select"
      data-test="model-select"
      :value="model"
      aria-label="Model"
      @change="onChange"
    >
      <optgroup
        v-for="group in groups"
        :key="group.family"
        :label="group.family"
      >
        <option v-for="m in group.list" :key="m.name" :value="m.name">
          {{ modelDisplayName(m) }}
        </option>
      </optgroup>
    </select>
    <p v-if="description" class="mp__desc" data-test="model-desc">
      {{ description }}
    </p>
  </div>
</template>

<style scoped>
.mp {
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card-lg);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 16px 18px;
}
.mp__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.mp__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.mp__browse {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
  text-decoration: none;
}
.mp__browse:hover {
  color: var(--safelight);
}
.mp__select {
  width: 100%;
  box-sizing: border-box;
  height: 40px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  padding: 0 12px;
  font-family: var(--f-mono);
  font-size: 13px;
}
.mp__desc {
  margin: 8px 0 0;
  font-size: 11px;
  color: var(--ink-3);
  line-height: 1.4;
}
</style>
