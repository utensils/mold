<script setup lang="ts">
import { computed } from "vue";
import { useDownloadsStore } from "../../stores/downloads";
import { percent } from "../../lib/format";
import type { DownloadJob } from "../../lib/api/types";

const emit = defineEmits<{ (e: "browse"): void }>();

const downloads = useDownloadsStore();

// Curated for Apple Silicon — small enough to pull and run on a Mac's GPU.
const STARTERS = [
  { model: "flux2-klein:q4", label: "smallest · 2.4 GB weights" },
  { model: "flux-schnell:q8", label: "fast 4-step" },
  { model: "sd3.5-medium:q8", label: "classic quality" },
];

// Match an in-flight download to a starter by model base name (the server may
// canonicalize the tag we posted).
function jobFor(model: string): DownloadJob | null {
  const base = model.split(":")[0]!;
  return downloads.inFlight.find((j) => j.model === model || j.model.startsWith(base)) ?? null;
}
const pulling = computed(() => new Set(STARTERS.map((s) => s.model).filter((m) => jobFor(m))));

async function pull(model: string) {
  await downloads.createDownload(model);
  downloads.subscribe();
}
</script>

<template>
  <div class="flex h-full flex-col items-center justify-center p-8">
    <h1 class="font-display text-display-lg font-bold text-ink" style="font-stretch: 90%">
      Develop your first print.
    </h1>
    <p class="mt-2 max-w-md text-center text-body-lg text-ink-2">
      mold runs models locally on your Mac's GPU. Pull one to start.
    </p>

    <div class="mt-8 grid w-full max-w-3xl grid-cols-1 gap-3 sm:grid-cols-3">
      <div
        v-for="starter in STARTERS"
        :key="starter.model"
        class="border-edge flex flex-col gap-2 rounded-chrome border bg-bench p-4"
      >
        <span class="data-mono text-body text-ink">{{ starter.model }}</span>
        <span class="text-caption text-ink-3">{{ starter.label }}</span>
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
          <span class="edge-code">Pulling…</span>
        </template>
        <button
          v-else
          type="button"
          class="border-edge h-8 rounded-control border text-body text-ink-2 hover:text-ink disabled:opacity-50"
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
