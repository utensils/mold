<script setup lang="ts">
import { computed } from "vue";
import { STARTER_MODELS } from "@studio/lib/starterModels";
import { useDownloadsStore } from "../../stores/downloads";
import { useToastStore } from "../../stores/toasts";
import { percent } from "../../lib/format";
import type { DownloadJob } from "../../lib/api/types";

const emit = defineEmits<{ (e: "browse"): void }>();

const downloads = useDownloadsStore();
const toasts = useToastStore();

// Cold start (§08 G10): the shortest path to a first print. Three small,
// fast-to-pull models — the recommended one first — each a one-click pull.
const STARTERS = STARTER_MODELS;

// Match an in-flight download to a starter by model base name (the server may
// canonicalize the tag we posted).
function jobFor(model: string): DownloadJob | null {
  const base = model.split(":")[0]!;
  return downloads.inFlight.find((j) => j.model === model || j.model.startsWith(base)) ?? null;
}
const pulling = computed(() => new Set(STARTERS.map((s) => s.model).filter((m) => jobFor(m))));

async function pull(model: string) {
  try {
    await downloads.createDownload(model);
    void downloads.subscribe().catch((error) => {
      const detail = error instanceof Error ? error.message : String(error);
      toasts.push(`Pull started, but progress is unavailable — ${detail}`, "error");
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    toasts.push(`Couldn't pull ${model} — ${detail}`, "error");
  }
}
</script>

<template>
  <div data-test="starter-cards" class="flex h-full flex-col items-center justify-center p-8">
    <h1 class="font-display text-display-lg font-bold text-ink" style="font-stretch: 90%">
      Develop your first print.
    </h1>
    <p class="mt-2 max-w-md text-center text-body-lg text-ink-2">
      mold runs models locally on your machine's GPU. Pull one to start.
    </p>

    <div class="mt-8 grid w-full max-w-3xl grid-cols-1 gap-3 sm:grid-cols-3">
      <div
        v-for="starter in STARTERS"
        :key="starter.model"
        data-test="starter-card"
        class="flex flex-col gap-2 rounded-chrome border bg-bench p-4"
        :class="
          starter.recommended
            ? 'border-safelight shadow-[inset_0_1px_0_var(--card-hi)]'
            : 'border-edge'
        "
      >
        <div class="flex items-center gap-2">
          <span class="data-mono text-body text-ink">{{ starter.model }}</span>
          <span
            v-if="starter.recommended"
            data-test="starter-recommended"
            class="edge-code ml-auto rounded-pill bg-[color-mix(in_srgb,var(--safelight)_16%,transparent)] px-1.5 !text-safelight"
          >
            Recommended
          </span>
        </div>
        <span class="text-caption text-ink-3">{{ starter.speed }}, {{ starter.size }}</span>
        <div class="flex-1" />
        <template v-if="jobFor(starter.model)">
          <div class="h-1.5 overflow-hidden rounded-full bg-bath">
            <div
              class="h-full bg-safelight transition-[width] duration-300"
              :style="{
                width: `${percent(jobFor(starter.model)!.bytes_done, jobFor(starter.model)!.bytes_total)}%`,
              }"
            />
          </div>
          <span class="edge-code" data-test="starter-pulling">
            Pulling…
            {{ percent(jobFor(starter.model)!.bytes_done, jobFor(starter.model)!.bytes_total) }}%
          </span>
        </template>
        <button
          v-else
          type="button"
          data-test="starter-pull"
          class="h-8 rounded-control text-body font-semibold transition-colors disabled:opacity-50"
          :class="
            starter.recommended
              ? 'bg-safelight text-on-accent hover:brightness-105'
              : 'border-edge border text-ink-2 hover:text-ink'
          "
          :disabled="pulling.has(starter.model)"
          @click="pull(starter.model)"
        >
          Pull
        </button>
      </div>
    </div>

    <button type="button" class="mt-6 text-body text-halide hover:text-ink" @click="emit('browse')">
      Browse all models
    </button>
  </div>
</template>
