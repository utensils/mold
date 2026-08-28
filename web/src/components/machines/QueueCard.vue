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
import type { ModelInfoExtended } from "../../types";
import { modelDisplayNameForId } from "@studio/lib/modelDisplay";
import QueuePlanWorkList from "@studio/components/QueuePlanWorkList.vue";
import type { QueuePlan } from "@studio/api/queuePlan";
import { queuePlanOnlyWork } from "@studio/lib/queuePlanPresentation";
import { queueWaitLabel, resolveQueueWait } from "@studio/lib/queuePosition";

const props = withDefaults(
  defineProps<{
    entries: QueueEntry[];
    plan?: QueuePlan | null;
    models?: ModelInfoExtended[];
    /** Exact schedulable CUDA ordinals advertised by this host. */
    gpuOrdinals: number[];
    canReorder?: boolean;
    canPause?: boolean;
    canCancelAll?: boolean;
    canCancelRunning?: boolean;
    cancellingIds?: string[];
    paused?: boolean;
    /** Stale data (host offline) — dim and disable controls. */
    dimmed?: boolean;
  }>(),
  {
    canReorder: false,
    canPause: false,
    canCancelAll: false,
    canCancelRunning: false,
    paused: false,
    dimmed: false,
    models: () => [],
    plan: null,
  },
);
const modelLabel = (name: string) => modelDisplayNameForId(name, props.models);

const emit = defineEmits<{
  cancel: [id: string];
  inspect: [id: string];
  setLane: [id: string, gpu: number | null];
  move: [id: string, position: number];
  togglePause: [];
  cancelAll: [];
}>();

const queuedIds = computed(() =>
  props.entries.filter((e) => e.state === "queued").map((e) => e.id),
);
const cancellableIds = computed(() =>
  props.entries
    .filter((entry) => entry.state === "queued" || entry.state === "paused")
    .map((entry) => entry.id),
);
const restartPaused = computed(() =>
  props.entries.some((entry) => entry.state === "paused"),
);
const resumeNeeded = computed(() => props.paused || restartPaused.value);
const entryIds = computed(() => props.entries.map((entry) => entry.id));
const hasPlanOnlyWork = computed(
  () => queuePlanOnlyWork(props.plan, entryIds.value).length > 0,
);
const visibleWorkCount = computed(
  () =>
    props.entries.length + queuePlanOnlyWork(props.plan, entryIds.value).length,
);

function laneValue(entry: QueueEntry): string {
  const lane = entry.target_gpu ?? null;
  return lane == null ? "" : String(lane);
}

function entryStateLabel(entry: QueueEntry): string {
  return queueWaitLabel(
    resolveQueueWait({
      state: entry.state,
      position: entry.position,
      blockedReason:
        props.paused && entry.state === "queued" ? "queue_paused" : null,
    }),
  );
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

// The reorder PATCH indexes `position` among QUEUED jobs only, so the target
// must be derived from this job's slot in the queued subset — not `position`,
// which counts running jobs too and would be off-by-N once anything is running.
function queuedIndexOf(id: string): number {
  return queuedIds.value.indexOf(id);
}
</script>

<template>
  <CardSurface :padded="false">
    <div class="qc" :data-dimmed="dimmed ? 'true' : undefined">
      <div class="qc__label">
        Queue<template v-if="visibleWorkCount">
          · {{ visibleWorkCount }} work</template
        >
      </div>

      <div
        v-if="
          canPause ||
          (canCancelAll && cancellableIds.length > 0) ||
          resumeNeeded
        "
        class="qc__controls"
      >
        <BadgePill
          v-if="resumeNeeded"
          tone="accent"
          outline
          data-test="paused-chip"
        >
          {{ restartPaused && !paused ? "paused after restart" : "paused" }}
        </BadgePill>
        <span class="qc__spacer" />
        <button
          v-if="canPause"
          type="button"
          class="qc__control"
          data-test="pause-toggle"
          :disabled="dimmed"
          @click="emit('togglePause')"
        >
          {{ resumeNeeded ? "Resume" : "Pause" }}
        </button>
        <button
          v-if="canCancelAll && cancellableIds.length > 0"
          type="button"
          class="qc__control qc__control--danger"
          data-test="cancel-all"
          :disabled="dimmed"
          @click="emit('cancelAll')"
        >
          Cancel all
        </button>
      </div>

      <p
        v-if="entries.length === 0 && !hasPlanOnlyWork"
        class="qc__empty"
        data-test="queue-empty"
      >
        Nothing queued.
      </p>

      <ul v-if="entries.length" class="qc__list">
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
            {{ entryStateLabel(entry) }}
          </BadgePill>
          <button
            type="button"
            class="qc__model"
            data-test="queue-inspect"
            :aria-label="`Job details for ${modelLabel(entry.model)}`"
            @click="emit('inspect', entry.id)"
          >
            {{ modelLabel(entry.model) }}
          </button>
          <span
            v-if="entry.state === 'held' && entry.held_reason"
            class="qc__held"
            :title="entry.held_reason ?? undefined"
          >
            {{ entry.held_reason }}
          </span>

          <select
            v-if="gpuOrdinals.length > 1"
            class="qc__lane"
            data-test="queue-lane"
            :disabled="dimmed || entry.state !== 'queued'"
            :value="laneValue(entry)"
            :aria-label="`GPU lane for ${modelLabel(entry.model)}`"
            @change="onLane(entry, $event)"
          >
            <option value="">Auto</option>
            <option
              v-for="ordinal in gpuOrdinals"
              :key="ordinal"
              :value="ordinal"
            >
              GPU {{ ordinal }}
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
              @click="emit('move', entry.id, queuedIndexOf(entry.id) - 1)"
            >
              <Icon name="chevron-up" :size="14" />
            </button>
            <button
              type="button"
              class="qc__icon"
              data-test="queue-down"
              :disabled="dimmed || isLastQueued(entry.id)"
              aria-label="Move down"
              @click="emit('move', entry.id, queuedIndexOf(entry.id) + 1)"
            >
              <Icon name="chevron-down" :size="14" />
            </button>
          </div>

          <button
            v-if="entry.state !== 'running' || canCancelRunning"
            type="button"
            class="qc__icon qc__icon--danger"
            data-test="queue-cancel"
            :disabled="dimmed || cancellingIds?.includes(entry.id)"
            :aria-label="`${cancellingIds?.includes(entry.id) ? 'Cancelling' : 'Cancel'} ${modelLabel(entry.model)}`"
            @click="emit('cancel', entry.id)"
          >
            <Icon name="trash" :size="14" />
          </button>
        </li>
      </ul>
      <QueuePlanWorkList
        class="qc__planned"
        :plan="plan"
        :exclude-ids="entryIds"
      />
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

/* The host's own words for why a job is parked — it will not start itself, so
   the reason is what makes the row actionable. */
.qc__held {
  overflow: hidden;
  color: var(--c-ink-3);
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.qc__label {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
  margin-bottom: 12px;
}

.qc__controls {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: -4px 0 10px;
}

.qc__spacer {
  flex: 1;
}

.qc__control {
  min-height: 36px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--ink-2);
  padding: 0 12px;
  cursor: pointer;
}

.qc__control--danger {
  color: var(--stop);
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

.qc__planned {
  margin-top: 8px;
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
  padding: 0;
  border: 0;
  background: none;
  font-family: var(--f-mono);
  font-size: 12.5px;
  color: var(--rebate);
  text-align: left;
  cursor: pointer;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.qc__model:hover,
.qc__model:focus-visible {
  color: var(--sel-ink, var(--safelight));
  text-decoration: underline;
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
