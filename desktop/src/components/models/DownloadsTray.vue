<script setup lang="ts">
import { useDownloadsStore } from "../../stores/downloads";
import { useHostsStore } from "../../stores/hosts";
import { modelSource } from "../../lib/modelSource";
import { formatGB, percent } from "../../lib/format";
import { PLATFORM_UI } from "../../lib/platform";
import SourceGlyph from "../generate/SourceGlyph.vue";

const downloads = useDownloadsStore();
const hosts = useHostsStore();

/** Primary rows carry a null label; resolve them to the primary host's name. */
function hostLabel(label: string | null): string {
  return label ?? hosts.primaryHost?.label ?? PLATFORM_UI.deviceLabel;
}
</script>

<template>
  <div v-if="downloads.hasActivity" class="border-edge border-b bg-bench px-4 py-2">
    <div class="edge-code mb-2">Downloads</div>
    <div class="flex flex-col gap-2">
      <div
        v-for="row in downloads.hostedInFlight"
        :key="`${row.hostId}:${row.job.id}`"
        class="-mx-1 flex items-center gap-3 rounded-control px-1 transition-colors duration-100 hover:bg-bath"
      >
        <span class="flex w-52 shrink-0 items-center gap-1.5">
          <SourceGlyph :source="modelSource({ name: row.job.model })" class="text-ink-3" />
          <span class="min-w-0 truncate text-body text-ink" :title="row.job.model">
            {{ row.job.model }}
          </span>
          <span class="edge-code shrink-0" data-test="download-host">
            · {{ hostLabel(row.hostLabel) }}
          </span>
        </span>
        <div
          class="h-1.5 flex-1 overflow-hidden rounded-full bg-bath"
          role="progressbar"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-valuenow="percent(row.job.bytes_done, row.job.bytes_total)"
          :aria-label="`Downloading ${row.job.model} on ${hostLabel(row.hostLabel)}`"
        >
          <div
            class="h-full bg-safelight transition-[width] duration-300"
            :style="{ width: `${percent(row.job.bytes_done, row.job.bytes_total)}%` }"
          />
        </div>
        <span class="data-mono w-28 shrink-0 text-right text-ink-3">
          {{ formatGB(row.job.bytes_done) }} / {{ formatGB(row.job.bytes_total) }}
        </span>
        <button
          type="button"
          class="text-ink-3 hover:text-stop active:translate-y-px"
          title="Cancel download"
          :aria-label="`Cancel download of ${row.job.model} on ${hostLabel(row.hostLabel)}`"
          :disabled="downloads.isCancelling(row.hostId, row.job.id)"
          @click="downloads.cancel(row.job.id, row.hostId)"
        >
          {{ downloads.isCancelling(row.hostId, row.job.id) ? "…" : "✕" }}
        </button>
      </div>
    </div>
  </div>
</template>
