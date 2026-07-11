<script setup lang="ts">
import { useDownloadsStore } from "../../stores/downloads";
import { formatGB, percent } from "../../lib/format";

const downloads = useDownloadsStore();
</script>

<template>
  <div v-if="downloads.hasActivity" class="border-edge border-t bg-bench px-4 py-2">
    <div class="edge-code mb-2">Downloads</div>
    <div class="flex flex-col gap-2">
      <div
        v-for="job in downloads.inFlight"
        :key="job.id"
        class="-mx-1 flex items-center gap-3 rounded-control px-1 transition-colors duration-100 hover:bg-bath"
      >
        <span class="w-40 shrink-0 truncate text-body text-ink" :title="job.model">
          {{ job.model }}
        </span>
        <div
          class="h-1.5 flex-1 overflow-hidden rounded-full bg-bath"
          role="progressbar"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-valuenow="percent(job.bytes_done, job.bytes_total)"
          :aria-label="`Downloading ${job.model}`"
        >
          <div
            class="h-full bg-safelight transition-[width] duration-300"
            :style="{ width: `${percent(job.bytes_done, job.bytes_total)}%` }"
          />
        </div>
        <span class="data-mono w-28 shrink-0 text-right text-ink-3">
          {{ formatGB(job.bytes_done) }} / {{ formatGB(job.bytes_total) }}
        </span>
        <button
          type="button"
          class="text-ink-3 hover:text-stop active:translate-y-px"
          title="Cancel download"
          :aria-label="`Cancel download of ${job.model}`"
          :disabled="downloads.cancelling.includes(job.id)"
          @click="downloads.cancel(job.id)"
        >
          {{ downloads.cancelling.includes(job.id) ? "…" : "✕" }}
        </button>
      </div>
    </div>
  </div>
</template>
