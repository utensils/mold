<script setup lang="ts">
/*
 * Host queue management card (spec §08 G2). The relocated home of queue
 * controls for a machine: each entry shows its state + model, a per-GPU lane
 * selector when the host has multiple GPUs, a cancel action, and — only when
 * the host advertises `capabilities.queue.can_reorder` — up/down reorder
 * buttons for queued jobs. Purely presentational: the parent owns the host
 * transport and reloads after each emitted action.
 */
import { computed } from "vue";
import CardSurface from "@ui/components/CardSurface.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import type { QueueEntry } from "../../types";

const props = withDefaults(
  defineProps<{
    entries: QueueEntry[];
    gpuCount: number;
    canReorder?: boolean;
    /** Stale data (host offline) — dim and disable controls. */
    dimmed?: boolean;
  }>(),
  { canReorder: false, dimmed: false },
);

const emit = defineEmits<{
  cancel: [id: string];
  setLane: [id: string, gpu: number | null];
  move: [id: string, position: number];
}>();

const queuedIds = computed(() =>
  props.entries.filter((e) => e.state === "queued").map((e) => e.id),
);

function laneValue(entry: QueueEntry): string {
  const lane = entry.target_gpu ?? entry.preferred_gpu ?? null;
  return lane == null ? "" : String(lane);
}

function onLane(entry: QueueEntry, event: Event) {
  const raw = (event.target as HTMLSelectElement).value;
  emit("setLane", entry.id, raw === "" ? null : Number(raw));
}

function isFirstQueued(id: string): boolean {
  return queuedIds.value[0] === id;
}
function isLastQueued(id: string): boolean {
  return queuedIds.value[queuedIds.value.length - 1] === id;
}
</script>

<template>
  <CardSurface :padded="false">
    <div class="qc" :data-dimmed="dimmed ? 'true' : undefined">
      <div class="qc__label">Queue</div>

      <p v-if="entries.length === 0" class="qc__empty" data-test="queue-empty">
        Nothing queued.
      </p>

      <ul v-else class="qc__list">
        <li
          v-for="entry in entries"
          :key="entry.id"
          class="qc__row"
          data-test="queue-row"
        >
          <BadgePill
            :tone="entry.state === 'running' ? 'accent' : 'neutral'"
            outline
          >
            {{ entry.state }}
          </BadgePill>
          <span class="qc__model">{{ entry.model }}</span>

          <select
            v-if="gpuCount > 1"
            class="qc__lane"
            data-test="queue-lane"
            :disabled="dimmed"
            :value="laneValue(entry)"
            :aria-label="`GPU lane for ${entry.model}`"
            @change="onLane(entry, $event)"
          >
            <option value="">Auto</option>
            <option v-for="g in gpuCount" :key="g - 1" :value="g - 1">
              GPU {{ g - 1 }}
            </option>
          </select>

          <div
            v-if="canReorder && entry.state === 'queued'"
            class="qc__reorder"
          >
            <button
              type="button"
              class="qc__icon"
              data-test="queue-up"
              :disabled="dimmed || isFirstQueued(entry.id)"
              aria-label="Move up"
              @click="emit('move', entry.id, entry.position - 1)"
            >
              <Icon name="chevron-up" :size="14" />
            </button>
            <button
              type="button"
              class="qc__icon"
              data-test="queue-down"
              :disabled="dimmed || isLastQueued(entry.id)"
              aria-label="Move down"
              @click="emit('move', entry.id, entry.position + 1)"
            >
              <Icon name="chevron-down" :size="14" />
            </button>
          </div>

          <button
            type="button"
            class="qc__icon qc__icon--danger"
            data-test="queue-cancel"
            :disabled="dimmed"
            :aria-label="`Cancel ${entry.model}`"
            @click="emit('cancel', entry.id)"
          >
            <Icon name="trash" :size="14" />
          </button>
        </li>
      </ul>
    </div>
  </CardSurface>
</template>

<style scoped>
.qc {
  padding: 16px;
}

.qc[data-dimmed="true"] {
  opacity: 0.55;
}

.qc__label {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
  margin-bottom: 12px;
}

.qc__empty {
  margin: 0;
  font-size: 12.5px;
  color: var(--ink-3);
}

.qc__list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.qc__row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 6px;
  border-radius: 7px;
}

.qc__row:hover {
  background: color-mix(in srgb, var(--rebate) 5%, transparent);
}

.qc__model {
  flex: 1;
  min-width: 0;
  font-family: var(--f-mono);
  font-size: 12.5px;
  color: var(--rebate);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.qc__lane {
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: 7px;
  color: var(--ink-2);
  font-family: var(--f-mono);
  font-size: 11px;
  padding: 4px 6px;
}

.qc__reorder {
  display: flex;
  gap: 2px;
}

.qc__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: 7px;
  cursor: pointer;
}

.qc__icon:hover:not(:disabled) {
  border-color: var(--ink-3);
}

.qc__icon:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.qc__icon--danger:hover:not(:disabled) {
  color: var(--stop);
  border-color: var(--stop);
}
</style>
