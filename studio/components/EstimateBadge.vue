<script
  setup
  lang="ts"
  generic="TRequest extends { model?: string | null }, TTarget"
>
import { onUnmounted, ref, watch } from "vue";
import {
  classifyGenerationFit,
  displayGenerationMemory,
  generationEstimateLabel,
  GENERATION_ESTIMATE_TOOLTIP,
  type EstimateFit,
  type GenerationCapacityEstimate,
} from "../lib/generationMemoryEstimate";

/*
 * The advisory VRAM badge, shared by the browser, the desktop app and the
 * phone. The three surfaces have their own HTTP clients and their own target
 * shapes, so the fetch arrives as a prop: this component owns the debounce,
 * the staleness token, the placeholder line and the verdict colour, and
 * nothing else.
 */
const props = defineProps<{
  request: TRequest | null;
  /** Machine the batch will route to; null/absent = the serving origin. */
  target?: TTarget | null;
  estimate: (
    request: TRequest,
    target?: TTarget | null,
  ) => Promise<GenerationCapacityEstimate>;
}>();

type Verdict = EstimateFit | "unavailable";
const fit = ref<Verdict>("unknown");
const text = ref("");
const visible = ref(false);
const refreshing = ref(false);
let timer: ReturnType<typeof setTimeout> | null = null;
let token = 0;

async function run(request: TRequest) {
  const mine = ++token;
  try {
    const estimate = await props.estimate(request, props.target ?? null);
    if (mine !== token) return;
    fit.value = classifyGenerationFit(estimate);
    const memory = displayGenerationMemory(estimate);
    text.value = generationEstimateLabel(
      fit.value,
      memory.peakBytes,
      memory.capacityBytes,
    );
  } catch {
    // Advisory, but say so instead of vanishing — a silently missing badge
    // reads as "everything fits".
    if (mine !== token) return;
    fit.value = "unavailable";
    text.value = "VRAM · estimate unavailable";
  }
  visible.value = true;
  refreshing.value = false;
}

// Debounced 600ms; only estimates when a style is selected. Re-runs when the
// routed machine changes — a different GPU means a different verdict.
watch(
  () => [props.request, props.target] as const,
  ([request]) => {
    ++token;
    if (timer) clearTimeout(timer);
    if (!request?.model) {
      // Only a style-less form hides the row; a mere refresh keeps the last
      // estimate on screen so the page never grows and shrinks around it.
      visible.value = false;
      refreshing.value = false;
      return;
    }
    if (!visible.value) {
      // Reserve the line immediately so the debounced first estimate doesn't
      // push the layout when it lands.
      fit.value = "unknown";
      text.value = "VRAM · estimating…";
      visible.value = true;
    }
    refreshing.value = true;
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
    class="estimate-badge"
    data-test="vram-estimate"
    role="status"
    aria-live="polite"
    :title="GENERATION_ESTIMATE_TOOLTIP"
    :data-fit="fit"
    :data-refreshing="refreshing || undefined"
  >
    {{ text }}
  </p>
</template>

<style scoped>
/*
 * The verdict colour lives here, keyed off `data-fit`, and nowhere else. Both
 * surfaces used to paint it with Tailwind utilities, where every colour has
 * the same specificity and the winner is emitted-rule order: desktop's static
 * `text-fg-dim` beat the bound `text-accent`/`text-error` and painted a tight
 * fit and a refusal in the same dim grey as an ordinary reading.
 */
.estimate-badge {
  margin: 0;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  white-space: nowrap;
  color: var(--mold-sapphire);
}
.estimate-badge[data-fit="fits"],
.estimate-badge[data-fit="unknown"] {
  color: var(--mold-sapphire);
}
.estimate-badge[data-fit="tight"] {
  color: var(--mold-blue);
}
.estimate-badge[data-fit="wont-fit"] {
  color: var(--mold-error);
}
.estimate-badge[data-fit="unavailable"] {
  color: var(--mold-text-dim);
}
.estimate-badge[data-refreshing] {
  opacity: 0.7;
}
</style>
