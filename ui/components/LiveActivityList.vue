<script setup lang="ts">
import type { FleetActiveWork } from "@studio/api/activity";

defineProps<{ rows: FleetActiveWork[] }>();

function title(row: FleetActiveWork): string {
  if (row.kind === "download")
    return row.model ? `Pulling ${row.model}` : "Model download";
  if (row.kind === "sequence")
    return row.model ? `${row.model} sequence` : "Sequence";
  if (row.kind === "generation") return row.model ?? "Generation";
  return row.model ?? row.kind.replaceAll("_", " ");
}

function progress(row: FleetActiveWork): number | null {
  if (!row.total || row.current == null) return null;
  return Math.max(
    0,
    Math.min(100, Math.round((row.current / row.total) * 100)),
  );
}
</script>

<template>
  <ol
    v-if="rows.length"
    class="live-activity-list"
    data-test="shared-live-activity"
  >
    <li
      v-for="row in rows"
      :key="row.key"
      class="live-activity-row"
      :data-stale="row.stale"
    >
      <span
        class="live-activity-dot"
        :data-phase="row.phase"
        aria-hidden="true"
      />
      <span class="live-activity-copy">
        <strong>{{ title(row) }}</strong>
        <span>
          {{ row.hostLabel }} · {{ row.phase.replaceAll("_", " ") }}
          <template v-if="progress(row) !== null">
            · {{ progress(row) }}%</template
          >
          <template v-if="row.stale">
            · Last seen active ·
            {{ row.hostError ?? "Waiting to reconnect" }}</template
          >
        </span>
      </span>
    </li>
  </ol>
</template>

<style scoped>
.live-activity-list {
  display: grid;
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.live-activity-row {
  display: flex;
  align-items: center;
  gap: 9px;
  min-height: 44px;
  padding: 7px 10px;
  border: 1px solid var(--edge);
  border-radius: 9px;
  background: var(--surface);
}
.live-activity-row[data-stale="true"] {
  opacity: 0.72;
  border-style: dashed;
}
.live-activity-dot {
  width: 7px;
  height: 7px;
  flex: none;
  border-radius: 50%;
  background: var(--safelight);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--safelight) 16%, transparent);
}
.live-activity-dot[data-phase="queued"],
.live-activity-dot[data-phase="preparing"] {
  background: var(--ink-3);
  box-shadow: none;
}
.live-activity-copy {
  display: grid;
  min-width: 0;
  gap: 2px;
}
.live-activity-copy strong {
  overflow: hidden;
  font-size: 12px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.live-activity-copy span {
  color: var(--ink-3);
  font-size: 11px;
  text-transform: capitalize;
}
</style>
