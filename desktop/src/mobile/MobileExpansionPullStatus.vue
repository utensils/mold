<script setup lang="ts">
import { computed } from "vue";
import type { ExpansionPullView } from "../lib/expansionPull";
import { formatBytes, formatEta, percent } from "../lib/format";
import { modelDisplayNameForId } from "../lib/models";

const props = defineProps<{
  model: string;
  hostLabel: string;
  error: string;
  status: ExpansionPullView;
  etaSeconds: number | null;
}>();
const modelLabel = computed(() => modelDisplayNameForId(props.model, []));
defineEmits<{ pull: []; "retry-expansion": [] }>();

const amount = computed(() =>
  props.status.job
    ? Math.round(percent(props.status.job.bytes_done, props.status.job.bytes_total))
    : 0,
);
const busy = computed(() =>
  ["connecting", "starting", "queued", "pulling"].includes(props.status.kind),
);
const label = computed(() => {
  switch (props.status.kind) {
    case "missing":
      return props.error;
    case "connecting":
      return `Connecting to ${props.hostLabel} to pull ${modelLabel.value}…`;
    case "starting":
      return `Starting ${modelLabel.value} on ${props.hostLabel}…`;
    case "queued":
      return `${modelLabel.value} queued on ${props.hostLabel}`;
    case "pulling":
      return `Pulling ${modelLabel.value} on ${props.hostLabel}`;
    case "ready":
      return `${modelLabel.value} ready on ${props.hostLabel}`;
    case "failed":
      return `${modelLabel.value} pull failed on ${props.hostLabel}`;
    case "cancelled":
      return `${props.model} pull cancelled on ${props.hostLabel}`;
  }
});
</script>

<template>
  <section
    class="mobile-expansion-pull"
    role="status"
    aria-live="polite"
    aria-atomic="true"
    :aria-busy="busy || undefined"
  >
    <strong>{{ label }}</strong>
    <span v-if="error && status.kind !== 'missing'" class="error-text" role="alert">{{
      error
    }}</span>
    <span v-if="status.kind === 'failed' && status.message" class="error-text">{{
      status.message
    }}</span>
    <div v-if="status.kind === 'pulling' && status.job" class="mobile-expansion-progress">
      <div
        role="progressbar"
        aria-valuemin="0"
        aria-valuemax="100"
        :aria-valuenow="amount"
        :aria-label="label"
      >
        <span :style="{ width: `${amount}%` }" />
      </div>
      <p>
        <span>{{ amount }}%</span>
        <span
          >{{ formatBytes(status.job.bytes_done) }} /
          {{ formatBytes(status.job.bytes_total) }}</span
        >
        <span v-if="status.job.files_total"
          >{{ status.job.files_done }}/{{ status.job.files_total }} files</span
        >
        <span v-if="status.job.current_file">{{ status.job.current_file }}</span>
        <span>ETA {{ formatEta(etaSeconds) }}</span>
      </p>
    </div>
    <button
      v-if="status.kind === 'missing'"
      type="button"
      class="secondary-button mobile-touch-action"
      data-test="mobile-pull-expansion"
      @click="$emit('pull')"
    >
      Pull expansion model
    </button>
    <button
      v-else-if="status.kind === 'ready'"
      type="button"
      class="primary-button mobile-touch-action"
      data-test="mobile-retry-expansion"
      @click="$emit('retry-expansion')"
    >
      Retry expansion
    </button>
    <button
      v-else-if="status.kind === 'failed' || status.kind === 'cancelled'"
      type="button"
      class="secondary-button mobile-touch-action"
      data-test="mobile-retry-expansion-pull"
      @click="$emit('pull')"
    >
      Retry {{ modelLabel }} pull on {{ hostLabel }}
    </button>
  </section>
</template>

<style scoped>
.mobile-expansion-pull {
  display: grid;
  gap: 8px;
  padding: 12px;
  border: 1px solid var(--control-edge);
  border-radius: var(--radius-control);
  background: var(--bench);
  font-size: 0.875rem;
}
.mobile-expansion-pull > span {
  overflow-wrap: anywhere;
}
.mobile-expansion-progress {
  display: grid;
  gap: 6px;
}
.mobile-expansion-progress [role="progressbar"] {
  height: 6px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--bath);
}
.mobile-expansion-progress [role="progressbar"] span {
  display: block;
  height: 100%;
  background: var(--safelight);
  transition: width 180ms ease-out;
}
.mobile-expansion-progress p {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
  margin: 0;
  color: var(--ink-3);
}
.mobile-touch-action {
  min-height: 44px;
}
@media (prefers-reduced-motion: reduce) {
  .mobile-expansion-progress [role="progressbar"] span {
    transition: none;
  }
}
</style>
