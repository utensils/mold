<script setup lang="ts">
import { computed } from "vue";
import { STARTER_MODELS } from "@mold/studio";
import { useDownloadsStore } from "../../stores/downloads";
import { useToastStore } from "../../stores/toasts";
import { percent } from "../../lib/format";
import type { DownloadJob } from "../../lib/api/types";

const emit = defineEmits<{ (e: "browse"): void }>();

const downloads = useDownloadsStore();
const toasts = useToastStore();

// Cold start: the shortest path to a first print. Three small,
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
      toasts.push(`It is downloading, but progress is unavailable — ${detail}`, "error");
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    toasts.push(`Couldn't get ${model} — ${detail}`, "error");
  }
}
</script>

<template>
  <div data-test="starter-cards" class="flex h-full flex-col items-center justify-center p-8">
    <h1 class="font-sans font-semibold text-xl font-bold text-fg" style="font-stretch: 90%">
      Make your first picture.
    </h1>
    <p class="mt-2 max-w-md text-center text-base text-fg-2">
      mold makes pictures on this machine's own graphics card. Get a style to start.
    </p>

    <div class="mt-8 grid w-full max-w-3xl grid-cols-1 gap-3 sm:grid-cols-3">
      <div
        v-for="starter in STARTERS"
        :key="starter.model"
        data-test="starter-card"
        class="flex flex-col gap-2 rounded-window border bg-bg p-4"
        :class="starter.recommended ? 'border-accent' : 'border-border'"
      >
        <div class="flex items-center gap-2">
          <span class="font-mono text-sm text-fg">{{ starter.model }}</span>
          <span
            v-if="starter.recommended"
            data-test="starter-recommended"
            class="font-mono text-micro text-fg-dim whitespace-nowrap ml-auto rounded-control bg-accent-tint px-1.5 !text-accent"
          >
            Recommended
          </span>
        </div>
        <span class="text-micro text-fg-dim">{{ starter.speed }}, {{ starter.size }}</span>
        <div class="flex-1" />
        <template v-if="jobFor(starter.model)">
          <div class="h-1.5 overflow-hidden bg-bg-deep">
            <div
              class="h-full bg-accent transition-[width] duration-300"
              :style="{
                width: `${percent(jobFor(starter.model)!.bytes_done, jobFor(starter.model)!.bytes_total)}%`,
              }"
            />
          </div>
          <span
            class="font-mono text-micro text-fg-dim whitespace-nowrap"
            data-test="starter-pulling"
          >
            Getting it…
            {{
              Math.round(
                percent(jobFor(starter.model)!.bytes_done, jobFor(starter.model)!.bytes_total),
              )
            }}%
          </span>
        </template>
        <button
          v-else
          type="button"
          data-test="starter-pull"
          class="h-8 rounded-control text-sm font-semibold transition-colors disabled:opacity-50"
          :class="
            starter.recommended
              ? 'bg-accent text-on-accent hover:brightness-105'
              : 'border-border border text-fg-2 hover:text-fg'
          "
          :disabled="pulling.has(starter.model)"
          @click="pull(starter.model)"
        >
          Get it
        </button>
      </div>
    </div>

    <button type="button" class="mt-6 text-sm text-sapphire hover:text-fg" @click="emit('browse')">
      Browse more
    </button>
  </div>
</template>
