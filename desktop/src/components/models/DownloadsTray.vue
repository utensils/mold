<script setup lang="ts">
/*
 * Downloads on their way — one banner per job (README §04: a sentence a
 * first-timer can act on, the CLI's progress line in mono beside it), then a
 * collapsed history. Rendered above the Styles shelf and, scoped by host, on
 * a machine's page.
 */
import { computed, ref } from "vue";
import { useDownloadsStore, type HostedDownloadJob } from "../../stores/downloads";
import { useHostsStore } from "../../stores/hosts";
import { useHostModelsStore } from "../../stores/hostModels";
import { useToastStore } from "../../stores/toasts";
import { modelSource } from "../../lib/modelSource";
import { formatEta, formatGB, formatRate, percent } from "../../lib/format";
import { PLATFORM_UI } from "../../lib/platform";
import { modelDisplayNameForId } from "../../lib/models";
import SourceGlyph from "../generate/SourceGlyph.vue";
import type { DownloadJobStatus } from "../../lib/api/types";

const downloads = useDownloadsStore();
const hosts = useHostsStore();
const hostModels = useHostModelsStore();
const toasts = useToastStore();
const props = defineProps<{ hostId?: string }>();

const rows = computed(() =>
  props.hostId
    ? downloads.hostedInFlight.filter((row) => row.hostId === props.hostId)
    : downloads.hostedInFlight,
);

/** History honors the same host scope — a host page must not leak other
 *  hosts' settled downloads (or appear because of them). */
const history = computed(() =>
  props.hostId
    ? downloads.hostedHistory.filter((row) => row.hostId === props.hostId)
    : downloads.hostedHistory,
);

/** History stays collapsed by default — the tray is a strip, not a page. */
const historyOpen = ref(false);
const retrying = ref<string[]>([]);

/** Primary rows carry a null label; resolve them to the primary host's name. */
function hostLabel(label: string | null): string {
  return label ?? hosts.primaryHost?.label ?? PLATFORM_UI.deviceLabel;
}

const modelLabel = (row: HostedDownloadJob) =>
  modelDisplayNameForId(row.job.model, hostModels.modelsOn(row.hostId));

/** Ink per download status — state colours only, never decoration. */
const STATUS_INK: Record<DownloadJobStatus, string> = {
  active: "text-warning",
  queued: "text-fg-2",
  completed: "text-success",
  failed: "text-error",
  cancelled: "text-fg-dim",
};

function rowKey(row: HostedDownloadJob): string {
  return `${row.hostId}:${row.job.id}`;
}

function isRetrying(row: HostedDownloadJob): boolean {
  return retrying.value.includes(rowKey(row));
}

/** Plain words for the wire's own status vocabulary (docs/design/README.md
 *  §2), so no surface ever prints `active` or `queued` at a person. */
const STATUS_WORD: Record<DownloadJobStatus, string> = {
  active: "Downloading",
  queued: "Waiting",
  completed: "Finished",
  failed: "Failed",
  cancelled: "Cancelled",
};

function statusLabel(row: HostedDownloadJob): string {
  if (row.job.status === "active" && row.job.bytes_total === 0) return "Getting ready";
  return STATUS_WORD[row.job.status];
}

function headline(row: HostedDownloadJob): string {
  return row.job.status === "queued"
    ? `${modelLabel(row)} is waiting its turn`
    : `Getting ${modelLabel(row)} ready`;
}

/** The CLI's own progress line: `[1.2 GB/6.8 GB, 12.4 MB/s, eta 8m 12s]`. */
function progressLine(row: HostedDownloadJob): string {
  const { job } = row;
  return `[${formatGB(job.bytes_done)}/${formatGB(job.bytes_total)}, ${formatRate(
    downloads.rateByJob[job.id] ?? null,
  )}, eta ${formatEta(downloads.etaByJob[job.id] ?? null)}]`;
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
    v-if="rows.length || history.length > 0"
    class="flex flex-col gap-2.5 px-3.5 pt-3.5"
    data-test="downloads-tray"
  >
    <div
      v-for="row in rows"
      :key="rowKey(row)"
      class="flex items-center gap-3 border bg-panel p-3"
      :class="row.job.status === 'active' ? 'border-warning' : 'border-border'"
    >
      <span class="font-mono text-xs" :class="STATUS_INK[row.job.status]" aria-hidden="true">
        ↓
      </span>
      <div class="flex min-w-0 flex-1 flex-col gap-1.5">
        <div class="flex min-w-0 flex-wrap items-baseline gap-x-2.5 gap-y-0.5">
          <span class="flex items-center gap-1.5 text-sm font-semibold text-fg">
            <SourceGlyph :source="modelSource({ name: row.job.model })" class="text-fg-dim" />
            {{ headline(row) }}
          </span>
          <span
            class="min-w-0 truncate font-mono text-micro text-fg-dim"
            :title="row.job.model"
            data-test="download-eta"
          >
            {{ row.job.model }} · {{ progressLine(row) }}
          </span>
          <span
            class="font-mono text-micro"
            :class="STATUS_INK[row.job.status]"
            data-test="download-status"
          >
            {{ statusLabel(row) }}
          </span>
        </div>
        <div
          class="h-1.5 overflow-hidden bg-surface"
          role="progressbar"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-valuenow="percent(row.job.bytes_done, row.job.bytes_total)"
          :aria-label="`Downloading ${modelLabel(row)} on ${hostLabel(row.hostLabel)}`"
        >
          <div
            class="h-full bg-warning transition-[width] duration-300"
            :style="{ width: `${percent(row.job.bytes_done, row.job.bytes_total)}%` }"
          />
        </div>
        <div class="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-fg-dim">
          <span>
            You can keep making images while this downloads. Going to
            <span class="font-mono" data-test="download-host">{{ hostLabel(row.hostLabel) }}</span
            >.
          </span>
          <span
            v-if="row.job.status === 'active' && row.job.current_file"
            class="min-w-0 truncate font-mono text-micro"
            :title="row.job.current_file"
            data-test="download-current-file"
          >
            {{ row.job.current_file }}
          </span>
          <span
            v-if="row.job.status === 'active' && row.job.files_total > 0"
            class="shrink-0 font-mono text-micro"
            data-test="download-files"
          >
            {{ row.job.files_done }}/{{ row.job.files_total }} files
          </span>
        </div>
      </div>
      <button
        type="button"
        class="ms-toolbar-button ms-toolbar-button--danger-hover"
        :aria-label="`Cancel download of ${modelLabel(row)} on ${hostLabel(row.hostLabel)}`"
        :disabled="downloads.isCancelling(row.hostId, row.job.id)"
        @click="downloads.cancel(row.job.id, row.hostId)"
      >
        {{ downloads.isCancelling(row.hostId, row.job.id) ? "Cancelling…" : "Cancel" }}
      </button>
    </div>

    <div v-if="history.length > 0">
      <button
        type="button"
        class="ms-group-label flex items-center gap-1 hover:text-fg"
        :aria-expanded="historyOpen"
        data-test="history-toggle"
        @click="historyOpen = !historyOpen"
      >
        <span class="inline-block transition-transform" :class="historyOpen ? 'rotate-90' : ''">
          ▸
        </span>
        History ({{ history.length }})
      </button>
      <ul
        v-if="historyOpen"
        class="mt-1.5 divide-y divide-border border border-border bg-panel"
        data-test="history-list"
      >
        <li
          v-for="row in history"
          :key="rowKey(row)"
          class="flex min-h-[38px] items-center gap-3 px-3 py-1.5 transition-colors duration-100 hover:bg-row-hover"
        >
          <span
            class="w-16 shrink-0 font-mono text-micro"
            :class="STATUS_INK[row.job.status]"
            data-test="history-status"
          >
            {{ statusLabel(row) }}
          </span>
          <span class="flex min-w-0 items-center gap-1.5">
            <SourceGlyph :source="modelSource({ name: row.job.model })" class="text-fg-dim" />
            <span class="min-w-0 truncate text-sm text-fg" :title="modelLabel(row)">
              {{ modelLabel(row) }}
            </span>
            <span class="shrink-0 font-mono text-micro text-fg-dim" data-test="download-host">
              · {{ hostLabel(row.hostLabel) }}
            </span>
          </span>
          <span
            v-if="row.job.error"
            class="min-w-0 flex-1 truncate text-micro text-error"
            :title="row.job.error"
          >
            {{ row.job.error }}
          </span>
          <span v-else class="flex-1" />
          <button
            v-if="row.job.status === 'failed'"
            type="button"
            class="ms-toolbar-button"
            :aria-label="`Retry download of ${modelLabel(row)} on ${hostLabel(row.hostLabel)}`"
            :disabled="isRetrying(row)"
            data-test="download-retry"
            @click="retry(row)"
          >
            {{ isRetrying(row) ? "Retrying…" : "Try again" }}
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>
