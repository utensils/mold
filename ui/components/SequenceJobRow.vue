<script setup lang="ts">
/*
 * One durable sequence job, rendered the same way everywhere it appears: the
 * Create strip's attention rows, Library ▸ History ▸ Sequences, and web's
 * history drawer. Presentational only — props in, `action` out; the caller
 * owns routing, confirmation, and the destructive half of every verb.
 */
import { computed, ref } from "vue";
import ProgressBar from "./ProgressBar.vue";
import {
  sequenceActionLabel,
  type ActivityAction,
  type ActivityJobVM,
} from "@studio/lib/activity";
import type { ChainJobState } from "@studio/lib/api/chainTypes";

type SequenceVM = ActivityJobVM & { kind: "sequence" };

const props = withDefaults(
  defineProps<{
    vm: SequenceVM;
    /** History rows are tighter: no live progress bar to make room for. */
    dense?: boolean;
    /** Narrow the row's buttons; defaults to the VM's state-derived set. */
    actions?: ActivityAction[] | null;
    /** Human model name resolved through the caller's inventory. */
    modelLabel?: string | null;
    /** Optional relative timestamp ("4m ago") rendered after the host. */
    timeLabel?: string | null;
    /** Non-destructive ✕ that removes the row from a strip for this session. */
    dismissible?: boolean;
    /** Activity summaries suppress detail when the canvas owns the full notice. */
    showError?: boolean;
  }>(),
  {
    dense: false,
    actions: null,
    modelLabel: null,
    timeLabel: null,
    dismissible: false,
    showError: true,
  },
);

const emit = defineEmits<{
  action: [action: ActivityAction, vm: SequenceVM];
  dismiss: [vm: SequenceVM];
  select: [vm: SequenceVM];
}>();

// State colour follows the development-temperature rule.
const STATE_CLASS: Record<ChainJobState, string> = {
  queued: "ms-seqrow__state--waiting",
  running: "ms-seqrow__state--active",
  paused: "ms-seqrow__state--waiting",
  interrupted: "ms-seqrow__state--waiting",
  failed: "ms-seqrow__state--failed",
  completed: "ms-seqrow__state--ink",
  cancelled: "ms-seqrow__state--muted",
};

const model = computed(() => props.modelLabel || props.vm.model);
const buttons = computed(() => props.actions ?? props.vm.actions);
const clip = computed(() =>
  Math.min(props.vm.currentStage + 1, props.vm.stageCount),
);
const percent = computed(() => {
  const p = props.vm.progress;
  return p && p.total > 0 ? Math.round((p.step / p.total) * 100) : 0;
});
const showProgress = computed(() => !props.dense && props.vm.progress !== null);
const phaseLabel = computed(() => props.vm.phase ?? props.vm.state);
const phaseClass = computed(() => {
  if (phaseLabel.value === "queued") return STATE_CLASS.queued;
  if (phaseLabel.value === "running" || phaseLabel.value === "finalizing")
    return STATE_CLASS.running;
  return STATE_CLASS[props.vm.state];
});
const errorExpanded = ref(false);
</script>

<template>
  <div
    class="ms-seqrow"
    :class="{ 'ms-seqrow--dense': dense }"
    data-test="sequence-job-row"
    :data-state="phaseLabel"
    role="button"
    tabindex="0"
    @click="emit('select', vm)"
    @keydown.enter.prevent="emit('select', vm)"
    @keydown.space.prevent="emit('select', vm)"
  >
    <div class="ms-seqrow__identity">
      <span class="ms-seqrow__state" :class="phaseClass">{{ phaseLabel }}</span>
      <span class="ms-seqrow__model" :title="model">{{ model }}</span>
      <span class="ms-seqrow__meta">
        {{ vm.stageCount }} clips · {{ clip }}/{{ vm.stageCount }} ·
        {{ vm.hostLabel }}
        <template v-if="timeLabel"> · {{ timeLabel }}</template>
      </span>
      <span
        v-if="showProgress"
        class="ms-seqrow__progress"
        data-test="seq-progress"
      >
        <ProgressBar :value="percent" :height="4" label="Sequence progress" />
      </span>
      <button
        v-if="showError && vm.error"
        class="ms-seqrow__error"
        :class="{ 'ms-seqrow__error--expanded': errorExpanded }"
        type="button"
        data-test="seq-error-disclosure"
        :aria-expanded="errorExpanded"
        :title="errorExpanded ? undefined : vm.error"
        @click.stop="errorExpanded = !errorExpanded"
      >
        <span>{{ vm.error }}</span>
        <span class="ms-seqrow__error-toggle" aria-hidden="true">
          {{ errorExpanded ? "Less" : "Details" }}
        </span>
      </button>
    </div>
    <div class="ms-seqrow__actions" data-test="sequence-job-actions">
      <button
        v-for="action in buttons"
        :key="action"
        type="button"
        class="ms-seqrow__btn"
        :class="{
          'ms-seqrow__btn--danger': action === 'cancel' || action === 'delete',
        }"
        :data-test="`seq-${action}`"
        @click.stop="emit('action', action, vm)"
      >
        {{ sequenceActionLabel(action, vm.state) }}
      </button>
      <slot name="actions" />
      <button
        v-if="dismissible"
        type="button"
        class="ms-seqrow__dismiss"
        data-test="seq-dismiss"
        title="Hide this from Activity. The sequence stays in Library ▸ History."
        :aria-label="`Dismiss ${model}`"
        @click.stop="emit('dismiss', vm)"
      >
        ✕
      </button>
    </div>
  </div>
</template>

<style scoped>
.ms-seqrow {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.ms-seqrow__identity {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  flex: 1;
}

.ms-seqrow__actions {
  display: flex;
  flex: 0 0 auto;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
}

.ms-seqrow__state {
  font-family: var(--mold-font-mono);
  font-size: 9.5px;
  text-transform: uppercase;
  flex: 0 0 auto;
}
.ms-seqrow__state--waiting {
  color: var(--mold-sapphire);
}
.ms-seqrow__state--active {
  color: var(--mold-blue);
}
.ms-seqrow__state--failed {
  color: var(--mold-error);
}
.ms-seqrow__state--ink {
  color: var(--mold-text);
}
.ms-seqrow__state--muted {
  color: var(--mold-text-dim);
}

.ms-seqrow__model {
  font-size: 12px;
  color: var(--mold-text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 220px;
}

.ms-seqrow__meta {
  font-family: var(--mold-font-mono);
  font-size: 9.5px;
  color: var(--mold-text-dim);
  flex: 0 0 auto;
}

.ms-seqrow__progress {
  flex: 0 1 160px;
  min-width: 60px;
}

.ms-seqrow__error {
  display: flex;
  align-items: baseline;
  gap: 6px;
  min-width: 0;
  border: 0;
  padding: 0;
  background: transparent;
  font-size: 10.5px;
  color: var(--mold-error);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 320px;
  text-align: left;
  cursor: pointer;
}

.ms-seqrow__error > span:first-child {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.ms-seqrow__error-toggle {
  flex: 0 0 auto;
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
  font-size: 9px;
  text-transform: uppercase;
}

.ms-seqrow__error--expanded {
  align-items: flex-start;
  white-space: normal;
  overflow: visible;
}

.ms-seqrow__error--expanded > span:first-child {
  overflow: visible;
  text-overflow: clip;
  overflow-wrap: anywhere;
}

.ms-seqrow--dense {
  align-items: stretch;
  flex-direction: column;
  gap: 6px;
}

.ms-seqrow--dense .ms-seqrow__identity {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 3px 10px;
}

.ms-seqrow--dense .ms-seqrow__model {
  max-width: none;
}

.ms-seqrow--dense .ms-seqrow__meta {
  grid-column: 2;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ms-seqrow--dense .ms-seqrow__progress,
.ms-seqrow--dense .ms-seqrow__error {
  grid-column: 1 / -1;
  max-width: none;
}

.ms-seqrow--dense .ms-seqrow__actions {
  width: 100%;
}

.ms-seqrow__btn {
  flex: 0 0 auto;
  border: 1px solid var(--mold-border-control);
  background: transparent;
  color: var(--mold-text-2);
  border-radius: 7px;
  padding: 3px 9px;
  font-size: 11px;
  cursor: pointer;
}
.ms-seqrow__btn:hover {
  color: var(--mold-text);
}
.ms-seqrow__btn--danger:hover {
  color: var(--mold-error);
  border-color: var(--mold-error);
}

.ms-seqrow__dismiss {
  flex: 0 0 auto;
  width: 20px;
  height: 20px;
  border: 0;
  background: transparent;
  color: var(--mold-text-dim);
  border-radius: 50%;
  font-size: 11px;
  cursor: pointer;
}
.ms-seqrow__dismiss:hover {
  color: var(--mold-error);
}
</style>
