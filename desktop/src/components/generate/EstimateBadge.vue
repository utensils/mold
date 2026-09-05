<script setup lang="ts">
import { onUnmounted, ref, watch } from "vue";
import type { ApiTarget } from "../../lib/api/client";
import type { GenerateRequest } from "../../lib/api/types";
import {
  classifyFit,
  displayEstimateMemory,
  ESTIMATE_TOOLTIP,
  estimateGeneration,
  estimateLabel,
  type EstimateFit,
} from "../../lib/api/estimate";

const props = defineProps<{
  request: GenerateRequest | null;
  /** Host the batch will route to; null/absent = the primary connection. */
  target?: ApiTarget | null;
}>();

type BadgeState = EstimateFit | "unavailable";
const fit = ref<BadgeState>("unknown");
const text = ref("");
const visible = ref(false);
let timer: ReturnType<typeof setTimeout> | null = null;
let token = 0;

async function run(req: GenerateRequest) {
  const mine = ++token;
  try {
    const est = await estimateGeneration(req, props.target);
    if (mine !== token) return;
    fit.value = classifyFit(est);
    const memory = displayEstimateMemory(est);
    text.value = estimateLabel(fit.value, memory.peakBytes, memory.capacityBytes);
    visible.value = true;
  } catch {
    // Advisory, but say so instead of vanishing — a silently missing badge
    // reads as "everything fits".
    if (mine !== token) return;
    fit.value = "unavailable";
    text.value = "VRAM · estimate unavailable";
    visible.value = true;
  }
}

// Debounced 600ms; only estimates when a model is selected. Re-runs when the
// routed host changes — a different GPU means a different verdict.
watch(
  () => [props.request, props.target] as const,
  ([req]) => {
    if (timer) clearTimeout(timer);
    if (!req || !req.model) {
      visible.value = false;
      return;
    }
    timer = setTimeout(() => void run(req), 600);
  },
  { deep: true, immediate: true },
);

onUnmounted(() => {
  if (timer) clearTimeout(timer);
});
</script>

<template>
  <!-- The colour is ENTIRELY in the bound map. A static `text-fg-dim` beside
       it is not a default: Tailwind utilities all have the same specificity,
       so the winner is emitted-rule order, and `.text-fg-dim` is emitted
       after `.text-accent` and `.text-error` — which painted a tight fit and
       a refusal in the same dim grey as an ordinary reading. -->
  <p
    v-if="visible"
    class="font-mono text-micro whitespace-nowrap"
    role="status"
    aria-live="polite"
    :title="ESTIMATE_TOOLTIP"
    :class="{
      'text-sapphire': fit === 'fits' || fit === 'unknown',
      'text-accent': fit === 'tight',
      'text-error': fit === 'wont-fit',
      'text-fg-dim': fit === 'unavailable',
    }"
  >
    {{ text }}
  </p>
</template>
