<script setup lang="ts">
import { onUnmounted, ref, watch } from "vue";
import type { GenerateRequest } from "../../lib/api/types";
import { classifyFit, estimateGeneration, type EstimateFit } from "../../lib/api/estimate";
import { formatGB } from "../../lib/format";

const props = defineProps<{ request: GenerateRequest | null }>();

const fit = ref<EstimateFit>("unknown");
const text = ref("");
const visible = ref(false);
let timer: ReturnType<typeof setTimeout> | null = null;
let token = 0;

async function run(req: GenerateRequest) {
  const mine = ++token;
  try {
    const est = await estimateGeneration(req);
    if (mine !== token) return;
    fit.value = classifyFit(est);
    const peak = formatGB(est.peak_memory_bytes);
    const total = est.available_memory_bytes != null ? formatGB(est.available_memory_bytes) : null;
    if (fit.value === "fits")
      text.value = total ? `Fits · est. ${peak} of ${total}` : `Est. ${peak}`;
    else if (fit.value === "tight") text.value = "Tight — close other apps";
    else if (fit.value === "wont-fit") text.value = "Won't fit on this GPU";
    else text.value = `Est. ${peak}`;
    visible.value = true;
  } catch {
    // Estimate is advisory — hide silently on any endpoint error.
    if (mine === token) visible.value = false;
  }
}

// Debounced 600ms; only estimates when a model is selected.
watch(
  () => props.request,
  (req) => {
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
  <p
    v-if="visible"
    class="edge-code"
    role="status"
    aria-live="polite"
    :class="{
      'text-halide': fit === 'fits' || fit === 'unknown',
      'text-safelight': fit === 'tight',
      'text-stop': fit === 'wont-fit',
    }"
  >
    {{ text }}
  </p>
</template>
