<script setup lang="ts">
import { onUnmounted, ref, watch } from "vue";
import type { GenerateRequest } from "../../lib/api/types";
import {
  classifyFit,
  ESTIMATE_TOOLTIP,
  estimateGeneration,
  estimateLabel,
  type EstimateFit,
} from "../../lib/api/estimate";

const props = defineProps<{ request: GenerateRequest | null }>();

type BadgeState = EstimateFit | "unavailable";
const fit = ref<BadgeState>("unknown");
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
    text.value = estimateLabel(
      fit.value,
      est.peak_memory_bytes,
      est.available_memory_bytes ?? null,
    );
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
    :title="ESTIMATE_TOOLTIP"
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
