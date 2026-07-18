<script setup lang="ts">
import { ref } from "vue";
import { useDownloadsStore, type HostedDownloadJob } from "../../stores/downloads";
import { useHostsStore } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import { modelSource } from "../../lib/modelSource";
import { formatEta, formatGB, percent } from "../../lib/format";
import { PLATFORM_UI } from "../../lib/platform";
import SourceGlyph from "../generate/SourceGlyph.vue";
import type { DownloadJobStatus } from "../../lib/api/types";

const downloads = useDownloadsStore();
const hosts = useHostsStore();
const toasts = useToastStore();

/** History stays collapsed by default — the tray is a strip, not a page. */
const historyOpen = ref(false);
const retrying = ref<string[]>([]);

/** Primary rows carry a null label; resolve them to the primary host's name. */
function hostLabel(label: string | null): string {
  return label ?? hosts.primaryHost?.label ?? PLATFORM_UI.deviceLabel;
}

/** Safelight chip palette per download status. */
const STATUS_CHIP: Record<DownloadJobStatus, string> = {
  active: "border-safelight/40 text-safelight",
  queued: "border-edge text-ink-2",
  completed: "border-halide/60 text-halide",
  failed: "border-stop/60 text-stop",
  cancelled: "border-edge text-ink-3",
};

function rowKey(row: HostedDownloadJob): string {
  return `${row.hostId}:${row.job.id}`;
}

function isRetrying(row: HostedDownloadJob): boolean {
  return retrying.value.includes(rowKey(row));
}

async function retry(row: HostedDownloadJob) {
  const key = rowKey(row);
  if (retrying.value.includes(key)) return;
  retrying.value.push(key);
  try {
    await downloads.retry(row.hostId, row.job);
  } catch (error) {
    toasts.push(
      error instanceof Error ? error.message : `Could not retry ${row.job.model}`,
      "error",
    );
  } finally {
    retrying.value = retrying.value.filter((candidate) => candidate !== key);
  }
}
</script>

<template>
  <div
    v-if="downloads.hasActivity || downloads.hostedHistory.length > 0"
    class="border-edge border-b bg-bench px-4 py-2"
  >
    <div class="edge-code mb-2">Downloads</div>
    <div class="flex flex-col gap-2">
      <div
        v-for="row in downloads.hostedInFlight"
        :key="rowKey(row)"
        class="-mx-1 flex flex-col gap-1 rounded-control px-1 transition-colors duration-100 hover:bg-bath"
      >
        <div class="flex items-center gap-3">
          <span class="flex w-52 shrink-0 items-center gap-1.5">
            <SourceGlyph :source="modelSource({ name: row.job.model })" class="text-ink-3" />
            <span class="min-w-0 truncate text-body text-ink" :title="row.job.model">
              {{ row.job.model }}
            </span>
            <span class="edge-code shrink-0" data-test="download-host">
              · {{ hostLabel(row.hostLabel) }}
            </span>
          </span>
          <span
            class="data-mono shrink-0 rounded-full border px-1.5 text-caption"
            :class="STATUS_CHIP[row.job.status]"
            data-test="download-status"
          >
            {{ row.job.status }}
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
        <div
          v-if="row.job.status === 'active'"
          class="flex items-center gap-3 pl-6 text-caption text-ink-3"
        >
          <span
            v-if="row.job.current_file"
            class="data-mono min-w-0 truncate"
            :title="row.job.current_file"
            data-test="download-current-file"
          >
            {{ row.job.current_file }}
          </span>
          <span v-if="row.job.files_total > 0" class="shrink-0" data-test="download-files">
            {{ row.job.files_done }}/{{ row.job.files_total }} files
          </span>
          <span class="shrink-0" data-test="download-eta">
            ETA {{ formatEta(downloads.etaByJob[row.job.id] ?? null) }}
          </span>
        </div>
      </div>
    </div>
    <div v-if="downloads.hostedHistory.length > 0" :class="downloads.hasActivity ? 'mt-2' : ''">
      <button
        type="button"
        class="edge-code flex items-center gap-1 text-ink-3 hover:text-ink"
        :aria-expanded="historyOpen"
        data-test="history-toggle"
        @click="historyOpen = !historyOpen"
      >
        <span class="inline-block transition-transform" :class="historyOpen ? 'rotate-90' : ''">
          ▸
        </span>
        History ({{ downloads.hostedHistory.length }})
      </button>
      <ul v-if="historyOpen" class="mt-1 flex flex-col gap-1" data-test="history-list">
        <li
          v-for="row in downloads.hostedHistory"
          :key="rowKey(row)"
          class="-mx-1 flex items-center gap-3 rounded-control px-1 transition-colors duration-100 hover:bg-bath"
        >
          <span
            class="data-mono shrink-0 rounded-full border px-1.5 text-caption"
            :class="STATUS_CHIP[row.job.status]"
            data-test="history-status"
          >
            {{ row.job.status }}
          </span>
          <span class="flex min-w-0 items-center gap-1.5">
            <SourceGlyph :source="modelSource({ name: row.job.model })" class="text-ink-3" />
            <span class="min-w-0 truncate text-body text-ink" :title="row.job.model">
              {{ row.job.model }}
            </span>
            <span class="edge-code shrink-0" data-test="download-host">
              · {{ hostLabel(row.hostLabel) }}
            </span>
          </span>
          <span
            v-if="row.job.error"
            class="min-w-0 flex-1 truncate text-caption text-stop"
            :title="row.job.error"
          >
            {{ row.job.error }}
          </span>
          <span v-else class="flex-1" />
          <button
            v-if="row.job.status === 'failed'"
            type="button"
            class="border-edge shrink-0 rounded-full border px-2 text-caption text-ink-2 hover:border-safelight hover:text-safelight active:translate-y-px"
            :aria-label="`Retry download of ${row.job.model} on ${hostLabel(row.hostLabel)}`"
            :disabled="isRetrying(row)"
            data-test="download-retry"
            @click="retry(row)"
          >
            {{ isRetrying(row) ? "…" : "Retry" }}
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>
