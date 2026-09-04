<script setup lang="ts">
/*
 * Queue (README §04): the same list the sidebar rail shows, at full width —
 * three counts, one explainer, and a table with a sentence of status per row.
 * Work in progress is context, so this view is a place to manage the line,
 * not to watch pixels arrive; selecting a row brings it to the canvas.
 */
import { computed } from "vue";
import AuthedMedia from "../components/gallery/AuthedMedia.vue";
import QueueRowMenu from "../components/shell/QueueRowMenu.vue";
import { useQueueActivity, type QueueRow } from "../composables/useQueueActivity";
import { useQueueCommands } from "../composables/useQueueCommands";
import { useQueueRowContext } from "../composables/useQueueRowContext";
import { madeTodayCount, rowGlyph, rowStatusLine, rowTitle, rowTone } from "../lib/queueRows";
import { formatEta } from "../lib/format";
import { thumbnailPath } from "../lib/gallery/media";
import { isMeshCompletion } from "@studio/lib/meshCompletion";
import { modelDisplayNameForId } from "../lib/models";
import { useGalleryStore } from "../stores/gallery";
import { useGenerationStore } from "../stores/generation";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";

const gallery = useGalleryStore();
const generation = useGenerationStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const queue = useQueueActivity();
const commands = useQueueCommands();
const rowContext = useQueueRowContext();
const title = (row: QueueRow) => rowTitle(row, hostModels.unionInstalled);
const status = (row: QueueRow) => rowStatusLine(row, rowContext.contextFor.value(row));

const doneToday = computed(() => madeTodayCount(gallery.merged));
const totalEta = computed(() => {
  const seconds = rowContext.totalEtaSeconds.value;
  return seconds === null || seconds <= 0 ? null : `about ${formatEta(seconds)} left in total`;
});
const hasFinished = computed(() =>
  generation.jobs.some((j) => j.status === "complete" || j.status === "error"),
);

function model(row: QueueRow): string {
  const id =
    row.kind === "print"
      ? row.print.model
      : row.kind === "sequence"
        ? row.sequence.model
        : (row.shared.model ?? "");
  return id ? modelDisplayNameForId(id, hostModels.unionInstalled) : "—";
}
function machine(row: QueueRow): string {
  if (row.kind === "print")
    return row.print.hostLabel ?? hosts.primaryHost?.label ?? "this machine";
  if (row.kind === "sequence") return row.sequence.hostLabel;
  return hosts.all.find((h) => h.id === row.shared.hostId)?.label ?? row.shared.hostId;
}
function progress(row: QueueRow): number | null {
  if (row.kind === "print" && row.print.status === "denoising" && row.print.total > 0) {
    return Math.round((row.print.step / row.print.total) * 100);
  }
  if (row.kind === "sequence" && row.sequence.stageCount > 0 && row.sequence.phase !== "queued") {
    return Math.round((row.sequence.currentStage / row.sequence.stageCount) * 100);
  }
  // A row waiting on its style has a meter too: the download it is waiting on.
  const preparation = rowContext.contextFor.value(row).wait?.preparation?.fraction;
  return preparation == null ? null : Math.round(preparation * 100);
}
/** A meter fills in the accent, or in the blocked tone when it measures a wait. */
function meterFill(row: QueueRow): string {
  return rowContext.contextFor.value(row).wait?.blockedReason === "preparing"
    ? "bg-state-blocked"
    : "bg-accent";
}
/** A mesh print's saved file is binary glTF, so its rendered poster is its
 * only still — checked before every raster arm. */
function meshPoster(row: QueueRow): string | null {
  if (row.kind !== "print") return null;
  const job = row.print;
  return isMeshCompletion(job.result) && job.result?.mesh_poster
    ? `data:image/png;base64,${job.result.mesh_poster}`
    : null;
}
function libraryThumb(row: QueueRow) {
  return row.kind === "print" && row.print.status === "complete" && row.print.result?.filename
    ? row.print
    : null;
}
function previewSrc(row: QueueRow): string | null {
  if (row.kind !== "print") return null;
  return row.print.previewUrl ?? (row.print.result?.video_frames ? null : row.print.resultUrl);
}
</script>

<template>
  <div class="flex h-full min-h-0 flex-col bg-bg">
    <!-- view toolbar -->
    <div
      class="flex h-[var(--mold-shell-viewbar-h)] shrink-0 items-center gap-3 border-b border-border bg-chrome px-3.5"
    >
      <span data-test="queue-headline" class="text-sm font-semibold text-fg">
        {{ queue.activeCount.value }} being made · {{ queue.waitingCount.value }} waiting
      </span>
      <span v-if="totalEta" data-test="queue-total-eta" class="text-xs text-fg-dim">
        {{ totalEta }}
      </span>
      <span class="flex-1" />
      <button
        v-if="commands.canPause.value"
        type="button"
        data-test="queue-pause"
        class="ms-toolbar-button"
        @click="commands.togglePause()"
      >
        {{ commands.paused.value ? "Resume queue" : "Pause queue" }}
      </button>
      <button
        type="button"
        data-test="queue-clear-finished"
        class="text-xs text-fg-dim hover:text-fg disabled:text-fg-faint"
        :disabled="!hasFinished"
        @click="generation.prune(0)"
      >
        Clear finished
      </button>
      <button
        type="button"
        data-test="queue-stop-all"
        class="ms-toolbar-button ms-toolbar-button--danger-hover"
        :disabled="queue.liveCount.value === 0"
        @click="commands.stopEverything()"
      >
        Stop everything
      </button>
    </div>

    <div class="flex min-h-0 flex-1 flex-col gap-3.5 overflow-y-auto p-4">
      <!-- the three counts -->
      <div class="grid grid-cols-3 gap-3">
        <div
          v-for="stat in [
            {
              label: 'Being made',
              value: queue.activeCount.value,
              note: queue.activeCount.value ? 'one at a time' : 'nothing right now',
            },
            {
              label: 'Waiting',
              value: queue.waitingCount.value,
              note: queue.waitingCount.value ? 'in the order you asked' : 'the line is empty',
            },
            { label: 'Done today', value: doneToday, note: 'all saved to My images' },
          ]"
          :key="stat.label"
          data-test="queue-stat"
          class="flex flex-col gap-1.5 rounded-control border border-border bg-panel p-3.5"
        >
          <span class="ms-group-label uppercase">{{ stat.label }}</span>
          <span class="text-lg font-semibold text-fg">{{ stat.value }}</span>
          <span class="text-xs text-fg-dim">{{ stat.note }}</span>
        </div>
      </div>

      <!-- explainer -->
      <div class="flex gap-3 rounded-control border border-border bg-panel-raised p-3">
        <span class="font-mono text-xs text-accent">•</span>
        <p class="text-xs leading-body text-fg-2" style="text-wrap: pretty">
          One image is made at a time so each gets your machine's full attention. Drag a row to
          reorder, or hit Jump the line on the one you need first. Closing the window keeps the
          queue running.
        </p>
      </div>

      <!-- the table -->
      <div class="border border-border bg-panel">
        <div class="table-grid ms-group-label border-b border-border px-3.5 py-2.5">
          <span>Image</span><span>What's happening</span><span>Style</span
          ><span class="text-right">Machine</span><span />
        </div>
        <p
          v-if="queue.rows.value.length === 0"
          data-test="queue-empty"
          class="px-3.5 py-6 text-sm text-fg-dim"
        >
          Nothing in the queue. Describe a picture in New image and press Generate.
        </p>
        <div
          v-for="row in queue.rows.value"
          :key="row.key"
          :data-test="`queue-row-${row.kind}`"
          class="table-grid min-h-[56px] cursor-pointer items-center border-b border-border px-3.5 py-3 transition-colors duration-100 hover:bg-row-hover focus-visible:outline-2 focus-visible:outline-accent"
          role="button"
          tabindex="0"
          @click="commands.open(row)"
          @keydown.enter.prevent="commands.open(row)"
          @keydown.space.prevent="commands.open(row)"
          @contextmenu.prevent="commands.contextMenu($event, row)"
        >
          <div class="flex min-w-0 items-center gap-2.5">
            <span
              class="flex h-[34px] w-[34px] shrink-0 items-center justify-center overflow-hidden border border-border bg-bg-crust font-mono text-xs"
              :class="rowTone(row)"
            >
              <img
                v-if="meshPoster(row)"
                :src="meshPoster(row)!"
                alt=""
                class="h-full w-full object-cover"
              />
              <AuthedMedia
                v-else-if="libraryThumb(row)"
                :path="thumbnailPath(libraryThumb(row)!.result!.filename!)"
                :target="generation.targetForJob(libraryThumb(row)!.clientId)"
                :cache-key="libraryThumb(row)!.hostId ?? hosts.primaryHost?.id ?? 'primary'"
                :alt="title(row)"
              />
              <img
                v-else-if="previewSrc(row)"
                :src="previewSrc(row)!"
                alt=""
                class="h-full w-full object-cover"
              />
              <span v-else>{{ rowGlyph(row) }}</span>
            </span>
            <span class="truncate text-sm text-fg">{{ title(row) }}</span>
          </div>
          <div class="flex flex-col gap-1 pr-4">
            <span class="text-xs" :class="rowTone(row)">{{ status(row) }}</span>
            <span
              v-if="progress(row) !== null"
              class="block h-[5px] overflow-hidden bg-surface"
              aria-hidden="true"
            >
              <span
                class="block h-full"
                :class="meterFill(row)"
                :style="{ width: `${progress(row)}%` }"
              />
            </span>
          </div>
          <span class="truncate font-mono text-xs text-fg-dim">{{ model(row) }}</span>
          <span class="truncate text-right font-mono text-xs text-fg-dim">{{ machine(row) }}</span>
          <span class="justify-self-end"><QueueRowMenu :row="row" /></span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.table-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 250px 150px 128px 34px;
}
</style>
