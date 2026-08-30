<script setup lang="ts">
import { computed } from "vue";
import type { QueuePlan, QueueWorkItem } from "../api/queuePlan";
import { queuePlanOnlyWork } from "../lib/queuePlanPresentation";
import {
  preparationForWorkItem,
  preparationLabel,
} from "../lib/queuePosition";

const props = withDefaults(
  defineProps<{
    plan?: QueuePlan | null;
    excludeIds?: readonly string[];
  }>(),
  { plan: null, excludeIds: () => [] },
);

const work = computed(() => queuePlanOnlyWork(props.plan, props.excludeIds));

defineExpose({ work });

function shortId(id: string): string {
  const tail = id.split(":").at(-1) ?? id;
  return tail.length > 10 ? `…${tail.slice(-8)}` : tail;
}

function words(value: string): string {
  return value.replaceAll("_", " ");
}

function kindLabel(item: QueueWorkItem): string {
  const base = words(item.work_kind || "work");
  const label = `${base[0]?.toUpperCase() ?? ""}${base.slice(1)}`;
  return item.chain_stage == null
    ? label
    : `${label} · stage ${item.chain_stage + 1}`;
}

function phaseLabel(item: QueueWorkItem): string {
  const preparation = preparationForWorkItem(item);
  if (preparation) return preparationLabel(preparation);
  return words(item.activity_phase ?? "scheduled");
}

function laneLabel(item: QueueWorkItem): string {
  if (item.gpu != null) return `GPU ${item.gpu}`;
  if (item.planned_lane_kind === "host_utility") return "CPU";
  if (item.planned_device_id) return shortId(item.planned_device_id);
  return "Auto";
}
</script>

<template>
  <ul v-if="work.length" class="plan-work" data-test="planned-queue-list">
    <li
      v-for="item in work"
      :key="item.work_id"
      class="plan-work__row"
      data-test="planned-queue-row"
    >
      <span class="plan-work__phase">{{ phaseLabel(item) }}</span>
      <span class="plan-work__main">
        <strong>{{ kindLabel(item) }}</strong>
        <code :title="item.parent_id || item.work_id">
          {{ shortId(item.parent_id || item.work_id) }}
        </code>
      </span>
      <span class="plan-work__lane">{{ laneLabel(item) }}</span>
    </li>
  </ul>
</template>

<style scoped>
.plan-work {
  display: grid;
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.plan-work__row {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 10px;
  min-width: 0;
  padding: 8px 10px;
  border: 1px solid var(--line, var(--ce));
  border-radius: 7px;
  color: var(--ink-2, currentColor);
  font-size: 12px;
}
.plan-work__phase {
  padding: 2px 6px;
  border: 1px solid var(--line, var(--ce));
  border-radius: 999px;
  font-size: 10px;
  text-transform: uppercase;
  white-space: nowrap;
}
.plan-work__main {
  display: flex;
  min-width: 0;
  gap: 8px;
  align-items: baseline;
}
.plan-work__main strong,
.plan-work__main code {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.plan-work__lane {
  color: var(--ink-3, currentColor);
  white-space: nowrap;
}
@media (max-width: 639px) {
  .plan-work__row {
    grid-template-columns: auto minmax(0, 1fr);
  }
  .plan-work__lane {
    grid-column: 2;
  }
}
</style>
