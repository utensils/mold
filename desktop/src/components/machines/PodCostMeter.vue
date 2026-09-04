<script setup lang="ts">
/*
 * Running-cost meter for a rented RunPod instance. Accrued cost ticks
 * every 30 s from the pod's reported uptime plus wall-clock time since the last
 * overview poll, so a paid host always shows what it has cost so far. Stop is
 * always adjacent to the meter.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { formatUsd, liveAccruedCost } from "../../lib/runpod";

const props = withDefaults(
  defineProps<{
    /** Hourly rate the pod reports once running. */
    costPerHr: number;
    /** Uptime seconds as of the last overview poll. */
    uptimeSeconds: number;
    /** Whether to show the adjacent Stop button. */
    stoppable?: boolean;
    busy?: boolean;
  }>(),
  { stoppable: false, busy: false },
);

const emit = defineEmits<{ stop: [] }>();

/** Wall-clock ms this uptime snapshot was read; re-anchored whenever the poll
 *  delivers a fresh uptime so the meter never drifts from the server. */
const snapshotAtMs = ref(Date.now());
const now = ref(Date.now());
let timer: ReturnType<typeof setInterval> | null = null;

watch(
  () => props.uptimeSeconds,
  () => {
    snapshotAtMs.value = Date.now();
    now.value = Date.now();
  },
);

const accrued = computed(() =>
  liveAccruedCost(props.costPerHr, props.uptimeSeconds, snapshotAtMs.value, now.value),
);

onMounted(() => {
  timer = setInterval(() => (now.value = Date.now()), 30_000);
});
onBeforeUnmount(() => {
  if (timer) clearInterval(timer);
});
</script>

<template>
  <div data-test="pod-cost-meter" class="flex items-center gap-2">
    <span class="font-mono text-micro text-fg-dim">
      <span class="text-state-cost" data-test="pod-accrued">≈{{ formatUsd(accrued) }}</span>
      · {{ formatUsd(costPerHr) }}/hr
    </span>
    <span class="flex-1" />
    <button
      v-if="stoppable"
      type="button"
      data-test="pod-cost-stop"
      class="ms-toolbar-button ms-toolbar-button--danger-hover"
      :disabled="busy"
      @click.stop="emit('stop')"
    >
      Stop it
    </button>
  </div>
</template>
