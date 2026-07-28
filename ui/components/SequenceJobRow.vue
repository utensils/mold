<script setup lang="ts">
/*
 * One durable sequence job, rendered the same way everywhere it appears: the
 * Create strip's attention rows, Library ▸ History ▸ Sequences, and web's
 * history drawer. Presentational only — props in, `action` out; the caller
 * owns routing, confirmation, and the destructive half of every verb.
 */
import { computed } from "vue";
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
  }>(),
  {
    dense: false,
    actions: null,
    modelLabel: null,
    timeLabel: null,
    dismissible: false,
  },
);

const emit = defineEmits<{
  action: [action: ActivityAction, vm: SequenceVM];
  dismiss: [vm: SequenceVM];
}>();

// State color follows the development-temperature rule (spec §2.1).
const STATE_CLASS: Record<ChainJobState, string> = {
  queued: "ms-seqrow__state--halide",
  running: "ms-seqrow__state--safelight",
  interrupted: "ms-seqrow__state--halide",
  failed: "ms-seqrow__state--stop",
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
</script>

<template>
  <div class="ms-seqrow" data-test="sequence-job-row" :data-state="vm.state">
    <span class="ms-seqrow__state" :class="STATE_CLASS[vm.state]">{{
      vm.state
    }}</span>
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
    <span class="ms-seqrow__spacer" />
    <span v-if="vm.error" class="ms-seqrow__error" :title="vm.error">{{
      vm.error
    }}</span>
    <button
      v-for="action in buttons"
      :key="action"
      type="button"
      class="ms-seqrow__btn"
      :class="{
        'ms-seqrow__btn--danger': action === 'cancel' || action === 'delete',
      }"
      :data-test="`seq-${action}`"
      @click="emit('action', action, vm)"
    >
      {{ sequenceActionLabel(action, vm.state) }}
    </button>
    <button
      v-if="dismissible"
      type="button"
      class="ms-seqrow__dismiss"
      data-test="seq-dismiss"
      title="Hide this from Activity. The sequence stays in Library ▸ History."
      :aria-label="`Dismiss ${model}`"
      @click="emit('dismiss', vm)"
    >
      ✕
    </button>
  </div>
</template>

<style scoped>
.ms-seqrow {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.ms-seqrow__state {
  font-family: var(--f-mono);
  font-size: 9.5px;
  text-transform: uppercase;
  flex: 0 0 auto;
}
.ms-seqrow__state--halide {
  color: var(--halide);
}
.ms-seqrow__state--safelight {
  color: var(--safelight);
}
.ms-seqrow__state--stop {
  color: var(--stop);
}
.ms-seqrow__state--ink {
  color: var(--ink);
}
.ms-seqrow__state--muted {
  color: var(--ink-3);
}

.ms-seqrow__model {
  font-size: 12px;
  color: var(--rebate);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 220px;
}

.ms-seqrow__meta {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
  flex: 0 0 auto;
}

.ms-seqrow__progress {
  flex: 0 1 160px;
  min-width: 60px;
}

.ms-seqrow__spacer {
  flex: 1;
}

.ms-seqrow__error {
  font-size: 10.5px;
  color: var(--stop);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 320px;
}

.ms-seqrow__btn {
  flex: 0 0 auto;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: 7px;
  padding: 3px 9px;
  font-size: 11px;
  cursor: pointer;
}
.ms-seqrow__btn:hover {
  color: var(--rebate);
}
.ms-seqrow__btn--danger:hover {
  color: var(--stop);
  border-color: var(--stop);
}

.ms-seqrow__dismiss {
  flex: 0 0 auto;
  width: 20px;
  height: 20px;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  border-radius: 50%;
  font-size: 11px;
  cursor: pointer;
}
.ms-seqrow__dismiss:hover {
  color: var(--stop);
}
</style>
