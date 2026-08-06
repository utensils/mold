<script setup lang="ts">
import { onUnmounted, ref, watch } from "vue";
import { fetchGenerationEstimate, type StreamTarget } from "../../api";
import type { GenerateRequestWire } from "../../types";
import {
  classifyGenerationFit,
  displayGenerationMemory,
  generationEstimateLabel,
  GENERATION_ESTIMATE_TOOLTIP,
  type EstimateFit,
} from "@studio/lib/generationMemoryEstimate";

const props = defineProps<{
  request: GenerateRequestWire | null;
  target?: StreamTarget | null;
}>();

type Fit = EstimateFit | "unavailable";
const fit = ref<Fit>("unknown");
const text = ref("");
const visible = ref(false);
let timer: ReturnType<typeof setTimeout> | null = null;
let token = 0;

async function run(request: GenerateRequestWire) {
  const mine = ++token;
  try {
    const estimate = await fetchGenerationEstimate(
      request,
      props.target ?? undefined,
    );
    if (mine !== token) return;
    fit.value = classifyGenerationFit(estimate);
    const memory = displayGenerationMemory(estimate);
    text.value = generationEstimateLabel(
      fit.value,
      memory.peakBytes,
      memory.capacityBytes,
    );
  } catch {
    if (mine !== token) return;
    fit.value = "unavailable";
    text.value = "VRAM · estimate unavailable";
  }
  visible.value = true;
}

watch(
  () => [props.request, props.target] as const,
  ([request]) => {
    ++token;
    if (timer) clearTimeout(timer);
    visible.value = false;
    if (!request?.model) {
      return;
    }
    timer = setTimeout(() => void run(request), 600);
  },
  { deep: true, immediate: true },
);

onUnmounted(() => {
  ++token;
  if (timer) clearTimeout(timer);
});
</script>

<template>
  <p
    v-if="visible"
    data-test="vram-estimate"
    role="status"
    aria-live="polite"
    :title="GENERATION_ESTIMATE_TOOLTIP"
    class="font-mono text-[11px]"
    :class="{
      'text-halide': fit === 'fits' || fit === 'unknown',
      'text-safelight': fit === 'tight',
      'text-stop': fit === 'wont-fit',
      'text-ink-3': fit === 'unavailable',
    }"
  >
    {{ text }}
  </p>
</template>
