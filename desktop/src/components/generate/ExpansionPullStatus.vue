<script setup lang="ts">
import { computed } from "vue";
import type { ExpansionPullView } from "../../lib/expansionPull";
import { formatBytes, formatEta, percent } from "../../lib/format";

const props = defineProps<{
  model: string;
  hostLabel: string;
  error: string;
  status: ExpansionPullView;
  etaSeconds: number | null;
}>();

defineEmits<{
  (e: "pull"): void;
  (e: "retry-expansion"): void;
}>();

const pullPercent = computed(() =>
  props.status.job
    ? Math.round(percent(props.status.job.bytes_done, props.status.job.bytes_total))
    : 0,
);
const busy = computed(() =>
  ["connecting", "starting", "queued", "pulling"].includes(props.status.kind),
);
const toneClass = computed(() => {
  if (props.status.kind === "ready")
    return "border-halide/45 bg-[color-mix(in_srgb,var(--halide)_8%,transparent)]";
  if (props.status.kind === "failed" || props.status.kind === "cancelled")
    return "border-stop/45 bg-stop/10";
  return "border-edge bg-bench";
});
const stateLabel = computed(() => {
  switch (props.status.kind) {
    case "missing":
      return props.error;
    case "connecting":
      return `Connecting to ${props.hostLabel} to pull ${props.model}…`;
    case "starting":
      return `Starting ${props.model} on ${props.hostLabel}…`;
    case "queued":
      return `${props.model} queued on ${props.hostLabel}`;
    case "pulling":
      return `Pulling ${props.model} on ${props.hostLabel}`;
    case "ready":
      return `${props.model} ready on ${props.hostLabel}`;
    case "failed":
      return `${props.model} pull failed on ${props.hostLabel}`;
    case "cancelled":
      return `${props.model} pull cancelled on ${props.hostLabel}`;
  }
});
</script>

<template>
  <section
    class="mt-3 rounded-control border px-2.5 py-2"
    :class="toneClass"
    :aria-busy="busy || undefined"
    role="status"
    aria-live="polite"
    aria-atomic="true"
    data-test="expansion-pull-status"
  >
    <div class="flex min-w-0 flex-wrap items-center justify-between gap-2">
      <div class="min-w-0">
        <p class="text-caption font-semibold text-ink">{{ stateLabel }}</p>
        <p
          v-if="status.kind === 'failed' && status.message"
          class="mt-0.5 truncate text-caption text-stop"
          :title="status.message"
        >
          {{ status.message }}
        </p>
        <p v-else-if="status.kind === 'cancelled'" class="mt-0.5 text-caption text-ink-2">
          The reviewed prompts and {{ hostLabel }} route are unchanged.
        </p>
      </div>
      <button
        v-if="status.kind === 'missing'"
        type="button"
        data-test="pull-expand-model"
        class="border-safelight/55 min-h-8 rounded-control border px-2.5 text-caption font-semibold text-safelight transition-colors duration-100 hover:border-safelight hover:text-ink"
        @click="$emit('pull')"
      >
        Pull expansion model
      </button>
      <button
        v-else-if="status.kind === 'ready'"
        type="button"
        data-test="retry-expansion"
        :aria-label="`Retry expansion with ${model} on ${hostLabel}`"
        class="min-h-8 rounded-control bg-safelight px-2.5 text-caption font-semibold text-on-accent transition-[filter] duration-100 hover:brightness-105"
        @click="$emit('retry-expansion')"
      >
        Retry expansion
      </button>
      <button
        v-else-if="status.kind === 'failed' || status.kind === 'cancelled'"
        type="button"
        data-test="retry-expand-pull"
        class="border-stop/60 min-h-8 rounded-control border px-2.5 text-caption font-semibold text-ink transition-colors duration-100 hover:border-stop"
        @click="$emit('pull')"
      >
        Retry {{ model }} pull on {{ hostLabel }}
      </button>
    </div>

    <div v-if="status.kind === 'pulling' && status.job" class="mt-2">
      <div
        class="h-1.5 overflow-hidden rounded-full bg-bath"
        role="progressbar"
        aria-valuemin="0"
        aria-valuemax="100"
        :aria-valuenow="pullPercent"
        :aria-label="`Pulling ${model} on ${hostLabel}`"
      >
        <div
          class="h-full bg-safelight transition-[width] duration-300 motion-reduce:transition-none"
          :style="{ width: `${pullPercent}%` }"
        />
      </div>
      <div
        class="mt-1 flex min-w-0 flex-wrap items-center gap-x-3 gap-y-0.5 text-caption text-ink-3"
      >
        <span class="data-mono text-safelight">{{ pullPercent }}%</span>
        <span class="data-mono">
          {{ formatBytes(status.job.bytes_done) }} / {{ formatBytes(status.job.bytes_total) }}
        </span>
        <span v-if="status.job.files_total > 0">
          {{ status.job.files_done }}/{{ status.job.files_total }} files
        </span>
        <span
          v-if="status.job.current_file"
          class="data-mono min-w-0 truncate"
          :title="status.job.current_file"
        >
          {{ status.job.current_file }}
        </span>
        <span>ETA {{ formatEta(etaSeconds) }}</span>
      </div>
    </div>
  </section>
</template>
