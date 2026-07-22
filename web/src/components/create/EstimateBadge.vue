<script setup lang="ts">
import { onUnmounted, ref, watch } from "vue";
import { fetchGenerationEstimate, type StreamTarget } from "../../api";
import type {
  GenerateRequestWire,
  GenerationMemoryEstimate,
} from "../../types";

const props = defineProps<{
  request: GenerateRequestWire | null;
  target?: StreamTarget | null;
}>();

type Fit = "fits" | "tight" | "wont-fit" | "unknown" | "unavailable";
const fit = ref<Fit>("unknown");
const text = ref("");
const visible = ref(false);
let timer: ReturnType<typeof setTimeout> | null = null;
let token = 0;

function classify(est: GenerationMemoryEstimate): Exclude<Fit, "unavailable"> {
  const available = est.available_memory_bytes;
  if (est.fits_available_memory === false) return "wont-fit";
  if (available == null) return "unknown";
  if (est.peak_memory_bytes > available) return "wont-fit";
  if (est.peak_memory_bytes > available * 0.92) return "tight";
  return "fits";
}

const gb = (bytes: number) => `${(bytes / 1_000_000_000).toFixed(1)} GB`;
function label(
  state: Exclude<Fit, "unavailable">,
  est: GenerationMemoryEstimate,
): string {
  const available = est.available_memory_bytes ?? null;
  if (state === "tight") return "VRAM · tight — close other apps";
  if (state === "wont-fit") return "VRAM · won't fit on this GPU";
  if (state === "fits" && available !== null)
    return `VRAM · fits — est. ${gb(est.peak_memory_bytes)} of ${gb(available)}`;
  return `VRAM · est. ${gb(est.peak_memory_bytes)}`;
}

async function run(request: GenerateRequestWire) {
  const mine = ++token;
  try {
    const estimate = await fetchGenerationEstimate(
      request,
      props.target ?? undefined,
    );
    if (mine !== token) return;
    fit.value = classify(estimate);
    text.value = label(fit.value, estimate);
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
    if (timer) clearTimeout(timer);
    if (!request?.model) {
      visible.value = false;
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
    title="Preflight estimate of peak GPU memory for this print on the selected machine."
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
