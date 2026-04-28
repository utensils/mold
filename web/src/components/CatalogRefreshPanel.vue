<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { useCatalog } from "../composables/useCatalog";

const cat = useCatalog();

// Drives the "elapsed Xm Ys" string. We re-render once a second instead
// of binding a `Date.now()` computed (which would only update on other
// reactive changes — meaning the timer would freeze whenever the scan
// stage isn't ticking).
const tick = ref(Date.now());
let tickHandle: ReturnType<typeof setInterval> | null = null;
onMounted(() => {
  tickHandle = setInterval(() => (tick.value = Date.now()), 1000);
});
onBeforeUnmount(() => {
  if (tickHandle) clearInterval(tickHandle);
});

const status = computed(() => cat.refreshStatus.value);
const state = computed(() => status.value?.state ?? "idle");
const isBusy = computed(
  () => state.value === "pending" || state.value === "running",
);

const familiesTotal = computed(() =>
  status.value?.state === "running" ? (status.value.families_total ?? 0) : 0,
);
const familiesDone = computed(() =>
  status.value?.state === "running" ? (status.value.families_done ?? 0) : 0,
);
const currentFamily = computed(() =>
  status.value?.state === "running"
    ? (status.value.current_family ?? null)
    : null,
);
const currentStage = computed(() =>
  status.value?.state === "running"
    ? (status.value.current_stage ?? null)
    : null,
);
const currentSeed = computed(() =>
  status.value?.state === "running"
    ? (status.value.current_seed ?? null)
    : null,
);
const pagesDone = computed(() =>
  status.value?.state === "running" ? (status.value.pages_done ?? 0) : 0,
);
const entriesSoFar = computed(() =>
  status.value?.state === "running" ? (status.value.entries_so_far ?? 0) : 0,
);
// Strip the org prefix so "stabilityai/stable-diffusion-xl-base-1.0"
// renders as "stable-diffusion-xl-base-1.0" — keeps the subline tight
// without losing the seed identity.
const seedShort = computed(() => {
  const seed = currentSeed.value;
  if (!seed) return null;
  const slash = seed.lastIndexOf("/");
  return slash >= 0 ? seed.slice(slash + 1) : seed;
});

// Determinate when we know how many families are in scope; spinner-only
// before the scanner has reported families_total.
const percent = computed(() => {
  if (state.value !== "running" || familiesTotal.value === 0) return null;
  return Math.min(
    100,
    Math.round((familiesDone.value / familiesTotal.value) * 100),
  );
});

const elapsedLabel = computed(() => {
  if (status.value?.state !== "running") return "";
  const startedAt = status.value.started_at_ms;
  if (!startedAt) return "";
  const secs = Math.max(0, Math.floor((tick.value - startedAt) / 1000));
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return m > 0 ? `${m}m ${s.toString().padStart(2, "0")}s` : `${s}s`;
});

const headline = computed(() => {
  switch (state.value) {
    case "pending":
      return "Queued — waiting for scan to start";
    case "running": {
      const fam = currentFamily.value;
      const stage = currentStage.value;
      if (fam) {
        const stageLabel = stage === "civitai" ? "Civitai" : "Hugging Face";
        return `Scanning ${fam} · ${stageLabel}`;
      }
      return "Starting scan…";
    }
    case "done": {
      const total =
        status.value?.state === "done" ? status.value.total_entries : 0;
      return `Refreshed · ${total} entries`;
    }
    case "failed":
      return /in progress/i.test(
        status.value?.state === "failed" ? status.value.message : "",
      )
        ? "Another scan is already running"
        : "Refresh failed";
    default:
      return "Catalog refresh";
  }
});

const subline = computed(() => {
  if (state.value === "failed" && status.value?.state === "failed") {
    return status.value.message;
  }
  if (state.value === "running") {
    const parts: string[] = [];
    if (familiesTotal.value > 0) {
      parts.push(`${familiesDone.value}/${familiesTotal.value} families`);
    }
    // Per-seed detail only shows during the HF stage — Civitai walks
    // run a different code path and don't populate current_seed.
    if (seedShort.value && currentStage.value === "hf") {
      const detail: string[] = [`seed ${seedShort.value}`];
      if (pagesDone.value > 0) detail.push(`page ${pagesDone.value}`);
      if (entriesSoFar.value > 0) detail.push(`${entriesSoFar.value} entries`);
      parts.push(detail.join(" · "));
    }
    if (elapsedLabel.value) parts.push(`elapsed ${elapsedLabel.value}`);
    return parts.join(" · ");
  }
  if (state.value === "done" && status.value?.state === "done") {
    const entries = Object.entries(status.value.per_family);
    return entries.length > 0
      ? `${entries.length} families scanned`
      : "No new entries";
  }
  return "Re-walks Hugging Face + Civitai for every supported family. Takes 30–60 min.";
});

const perFamily = computed(() => {
  if (status.value?.state !== "done")
    return [] as { family: string; outcome: string }[];
  return Object.entries(status.value.per_family).map(([family, outcome]) => ({
    family,
    outcome,
  }));
});

async function onClick() {
  if (isBusy.value) return;
  try {
    await cat.startRefresh();
  } catch (err) {
    // useCatalog already records failed state on refreshStatus; logging
    // here keeps the devtools breadcrumb without double-surfacing the
    // message in the panel.
    console.error("catalog refresh failed", err);
  }
}
</script>

<template>
  <section
    class="glass flex flex-col gap-3 rounded-3xl px-4 py-3.5 sm:px-5"
    aria-label="Catalog refresh"
  >
    <div class="flex flex-wrap items-center gap-3">
      <!-- Status icon -->
      <div
        class="flex h-9 w-9 shrink-0 items-center justify-center rounded-full"
        :class="{
          'bg-white/5 text-ink-300': state === 'idle',
          'bg-brand-500/20 text-brand-200': isBusy,
          'bg-emerald-500/20 text-emerald-200': state === 'done',
          'bg-rose-500/20 text-rose-200': state === 'failed',
        }"
      >
        <svg
          v-if="isBusy"
          class="h-4 w-4 animate-spin"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" />
          <path d="M21 3v5h-5" />
          <path d="M21 12a9 9 0 0 1-15.5 6.3L3 16" />
          <path d="M3 21v-5h5" />
        </svg>
        <svg
          v-else-if="state === 'done'"
          class="h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.4"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M5 13l4 4L19 7" />
        </svg>
        <svg
          v-else-if="state === 'failed'"
          class="h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="9" />
          <path d="M12 8v4" />
          <path d="M12 16h.01" />
        </svg>
        <svg
          v-else
          class="h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" />
          <path d="M21 3v5h-5" />
          <path d="M21 12a9 9 0 0 1-15.5 6.3L3 16" />
          <path d="M3 21v-5h5" />
        </svg>
      </div>

      <!-- Headline + subline -->
      <div class="min-w-0 flex-1">
        <div class="text-[14px] font-semibold tracking-tight text-ink-50">
          {{ headline }}
        </div>
        <div class="truncate text-[12px] text-ink-300" :title="subline">
          {{ subline }}
        </div>
      </div>

      <!-- Action button -->
      <button
        type="button"
        class="inline-flex h-10 items-center gap-2 rounded-full border px-4 text-[13px] font-medium transition disabled:cursor-not-allowed disabled:opacity-60"
        :class="
          isBusy
            ? 'border-white/5 bg-white/5 text-ink-200'
            : 'border-brand-400/40 bg-brand-500/15 text-brand-100 hover:bg-brand-500/25'
        "
        :disabled="isBusy"
        :aria-busy="isBusy"
        :aria-label="isBusy ? 'Catalog refresh in progress' : 'Refresh catalog'"
        @click="onClick"
      >
        <svg
          class="h-4 w-4"
          :class="{ 'animate-spin': isBusy }"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" />
          <path d="M21 3v5h-5" />
          <path d="M21 12a9 9 0 0 1-15.5 6.3L3 16" />
          <path d="M3 21v-5h5" />
        </svg>
        <span>{{
          isBusy
            ? state === "pending"
              ? "Queued…"
              : "Scanning…"
            : "Refresh catalog"
        }}</span>
      </button>
    </div>

    <!-- Determinate progress bar, shown only when running with known total. -->
    <div
      v-if="state === 'running'"
      class="relative h-1.5 w-full overflow-hidden rounded-full bg-white/5"
      role="progressbar"
      :aria-valuemin="0"
      :aria-valuemax="100"
      :aria-valuenow="percent ?? undefined"
    >
      <div
        v-if="percent !== null"
        class="h-full rounded-full bg-brand-400 transition-[width] duration-500 ease-out"
        :style="{ width: percent + '%' }"
      />
      <!-- Indeterminate marquee while we don't yet know families_total. -->
      <div
        v-else
        class="absolute inset-y-0 w-1/3 rounded-full bg-brand-400/70 [animation:catalogScanMarquee_1.6s_ease-in-out_infinite]"
      />
    </div>

    <!-- Per-family outcome table after a Done. Compact list, two columns
         on wider screens. Reads like the CLI's --json summary. -->
    <ul
      v-if="state === 'done' && perFamily.length > 0"
      class="grid grid-cols-1 gap-x-6 gap-y-1 text-[12px] text-ink-300 sm:grid-cols-2"
    >
      <li
        v-for="row in perFamily"
        :key="row.family"
        class="flex items-baseline justify-between gap-2"
      >
        <span class="truncate font-medium text-ink-100">{{ row.family }}</span>
        <span class="truncate text-right">{{ row.outcome }}</span>
      </li>
    </ul>
  </section>
</template>

<style scoped>
@keyframes catalogScanMarquee {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(400%);
  }
}
</style>
