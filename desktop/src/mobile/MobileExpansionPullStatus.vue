<script setup lang="ts">
import { computed } from "vue";
import { expansionPullPresentation, type ExpansionPullView } from "../lib/expansionPull";
import type { ModelEntry } from "../lib/api/types";

const props = withDefaults(
  defineProps<{
    model: string;
    hostLabel: string;
    error: string;
    status: ExpansionPullView;
    etaSeconds: number | null;
    models?: ModelEntry[] | undefined;
    pullLabel?: string;
    readyLabel?: string;
    retryLabel?: string;
  }>(),
  {
    pullLabel: "Pull expansion model",
    readyLabel: "Retry expansion",
    retryLabel: "",
  },
);
defineEmits<{ pull: []; "retry-expansion": [] }>();

const presentation = computed(() =>
  expansionPullPresentation(
    props.status,
    props.model,
    props.hostLabel,
    props.error,
    props.etaSeconds,
    props.models ?? [],
  ),
);
</script>

<template>
  <section
    class="mobile-expansion-pull"
    role="status"
    aria-live="polite"
    aria-atomic="true"
    :aria-busy="presentation.busy || undefined"
  >
    <strong>{{ presentation.label }}</strong>
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
        :aria-valuenow="presentation.percent"
        :aria-label="presentation.label"
      >
        <span :style="{ width: `${presentation.percent}%` }" />
      </div>
      <p>
        <span>{{ presentation.percent }}%</span>
        <span>{{ presentation.bytes }}</span>
        <span v-if="presentation.files">{{ presentation.files }}</span>
        <span v-if="status.job.current_file">{{ status.job.current_file }}</span>
        <span>ETA {{ presentation.eta }}</span>
      </p>
    </div>
    <button
      v-if="status.kind === 'missing'"
      type="button"
      class="secondary-button mobile-touch-action"
      data-test="mobile-pull-expansion"
      @click="$emit('pull')"
    >
      {{ pullLabel }}
    </button>
    <button
      v-else-if="status.kind === 'ready'"
      type="button"
      class="primary-button mobile-touch-action"
      data-test="mobile-retry-expansion"
      @click="$emit('retry-expansion')"
    >
      {{ readyLabel }}
    </button>
    <button
      v-else-if="status.kind === 'failed' || status.kind === 'cancelled'"
      type="button"
      class="secondary-button mobile-touch-action"
      data-test="mobile-retry-expansion-pull"
      @click="$emit('pull')"
    >
      {{ retryLabel || `Retry ${presentation.modelLabel} pull on ${hostLabel}` }}
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
