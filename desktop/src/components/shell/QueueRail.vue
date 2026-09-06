<script setup lang="ts">
/*
 * The queue, under the machine card (README §04). One active card (52px
 * thumb, sentence status, meter, pause/stop) then compact rows (38px thumb
 * or a mono glyph for an image that does not exist yet, title, one-line
 * status, ⋯). Header controls pause the whole queue, stop everything, or
 * open the full Queue view. The explainer is never open by default.
 */
import { computed, ref } from "vue";
import { useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import AuthedMedia from "../gallery/AuthedMedia.vue";
import QueueRowMenu from "./QueueRowMenu.vue";
import { useQueueActivity, type QueueRow } from "../../composables/useQueueActivity";
import { useQueueCommands } from "../../composables/useQueueCommands";
import { useQueueRowContext } from "../../composables/useQueueRowContext";
import {
  batchPositionLabel,
  railStatusLine,
  rowGlyph,
  rowTitle,
  rowTone,
} from "../../lib/queueRows";
import { thumbnailPath } from "../../lib/gallery/media";
import { isMeshCompletion } from "@studio/lib/meshCompletion";
import { useGenerationStore } from "../../stores/generation";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";

const router = useRouter();
const generation = useGenerationStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const title = (row: QueueRow) => rowTitle(row, hostModels.unionInstalled);
const queue = useQueueActivity();
const commands = useQueueCommands();
const rowContext = useQueueRowContext();
const status = (row: QueueRow) => railStatusLine(row, rowContext.contextFor.value(row));
const batchPosition = (row: QueueRow) => batchPositionLabel(row, generation.jobs);

const explainOpen = ref(false);

const active = computed(() => queue.active.value);
// The active card is fleet-wide — it shows whichever machine started most
// recently — so its pause, its label and its meter tint all belong to THAT
// machine. The header's pause above is the display host's, deliberately: it
// is the whole-queue control the status bar's Space hint describes.
const activeHostId = computed(() => (active.value ? commands.hostIdFor(active.value) : null));
const activeCanPause = computed(() => commands.canPauseFor(activeHostId.value));
const activePaused = computed(() => commands.pausedFor(activeHostId.value));
/** The card pauses the ROW's machine while the rail header pauses the display
 *  host, so the two controls a few pixels apart name their machines. */
const activeHostLabel = computed(() => {
  const id = activeHostId.value;
  return (id && hosts.all.find((host) => host.id === id)?.label) || "this machine";
});
const activePauseTitle = computed(() =>
  activePaused.value
    ? `Resume the queue on ${activeHostLabel.value}`
    : `Pause the queue on ${activeHostLabel.value} after this image`,
);
// The rail is context, not the record: it shows what fits a sidebar without
// becoming a scroll of its own. The Queue view is the full list.
const RAIL_ROWS = 12;
const rows = computed(() =>
  queue.rows.value.filter((row) => row.key !== active.value?.key).slice(0, RAIL_ROWS),
);

/** A mesh print's saved file is binary glTF, so its rendered poster is its
 * only still — checked before every raster arm. */
function meshPoster(row: QueueRow): string | null {
  if (row.kind !== "print") return null;
  const job = row.print;
  return isMeshCompletion(job.result) && job.result?.mesh_poster
    ? `data:image/png;base64,${job.result.mesh_poster}`
    : null;
}
/** Where a raster row's picture comes from, if it exists yet. */
function thumbSrc(row: QueueRow): string | null {
  if (row.kind !== "print") return null;
  const job = row.print;
  if (job.resultUrl && !job.result?.video_frames) return job.resultUrl;
  return job.previewUrl ?? null;
}
function libraryThumb(row: QueueRow) {
  return row.kind === "print" && row.print.status === "complete" && row.print.result?.filename
    ? row.print
    : null;
}
function progressPct(row: QueueRow): number | null {
  if (row.kind === "print" && row.print.status === "denoising" && row.print.total > 0) {
    return Math.round((row.print.step / row.print.total) * 100);
  }
  return null;
}
</script>

<template>
  <div data-test="queue-rail" class="flex flex-col border-t border-border pt-2">
    <div class="flex shrink-0 items-center gap-2 px-0.5 pb-2">
      <span class="ms-group-label uppercase">Queue</span>
      <span
        v-if="queue.liveCount.value > 0"
        data-test="queue-count"
        class="inline-flex h-4 min-w-[18px] items-center justify-center rounded-control bg-surface-2 px-1.5 font-mono text-micro text-fg"
      >
        {{ queue.liveCount.value }}
      </span>
      <span class="flex-1" />
      <button
        v-if="commands.canPause.value"
        type="button"
        data-test="queue-pause"
        class="ms-toolbar-button ms-toolbar-button--icon"
        :class="commands.paused.value ? 'text-accent border-accent' : ''"
        :title="commands.paused.value ? 'Resume the queue' : 'Pause the whole queue'"
        :aria-label="commands.paused.value ? 'Resume the queue' : 'Pause the whole queue'"
        @click="commands.togglePause()"
      >
        <Icon :name="commands.paused.value ? 'play' : 'pause'" :size="12" />
      </button>
      <button
        type="button"
        data-test="queue-stop-all"
        class="ms-toolbar-button ms-toolbar-button--icon ms-toolbar-button--danger-hover"
        title="Stop everything"
        aria-label="Stop everything"
        :disabled="queue.liveCount.value === 0"
        @click="commands.askStopEverything()"
      >
        <Icon name="stop" :size="12" />
      </button>
      <button
        type="button"
        data-test="queue-open"
        class="ms-toolbar-button ms-toolbar-button--icon"
        title="Open the full queue"
        aria-label="Open the full queue"
        @click="router.push('/queue')"
      >
        <Icon name="expand" :size="12" />
      </button>
    </div>

    <div data-test="queue-rows" class="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto pr-0.5">
      <!-- active card -->
      <div
        v-if="active"
        data-test="queue-active"
        class="flex shrink-0 cursor-pointer flex-col gap-2 rounded-control border border-accent bg-panel-raised p-2.5"
        role="button"
        tabindex="0"
        @click="commands.open(active)"
        @keydown.enter.prevent="commands.open(active)"
        @keydown.space.prevent="commands.open(active)"
        @contextmenu.prevent="commands.contextMenu($event, active)"
      >
        <div class="flex gap-2.5">
          <span
            class="flex h-[52px] w-[52px] shrink-0 items-center justify-center overflow-hidden border border-accent bg-bg-crust font-mono text-xs text-accent"
          >
            <img
              v-if="meshPoster(active)"
              :src="meshPoster(active)!"
              alt=""
              class="h-full w-full object-cover"
            />
            <AuthedMedia
              v-else-if="libraryThumb(active)"
              :path="thumbnailPath(libraryThumb(active)!.result!.filename!)"
              :target="generation.targetForJob(libraryThumb(active)!.clientId)"
              :cache-key="libraryThumb(active)!.hostId ?? hosts.primaryHost?.id ?? 'primary'"
              :alt="title(active)"
            />
            <img
              v-else-if="thumbSrc(active)"
              :src="thumbSrc(active)!"
              alt=""
              class="h-full w-full object-cover"
              style="filter: blur(1.4px)"
            />
            <span v-else class="ms-pulse">⠂</span>
          </span>
          <div class="flex min-w-0 flex-1 flex-col gap-1">
            <span class="truncate text-xs font-semibold text-fg">{{ title(active) }}</span>
            <span class="truncate text-micro text-fg-2">{{ status(active) }}</span>
            <div class="flex items-center gap-1.5">
              <span
                class="block h-1.5 flex-1 overflow-hidden bg-bg-crust"
                :class="activePaused ? 'opacity-50' : ''"
                aria-hidden="true"
              >
                <span
                  class="block h-full bg-accent"
                  :style="{ width: `${progressPct(active) ?? 8}%` }"
                  :class="progressPct(active) === null && !activePaused ? 'ms-pulse' : ''"
                />
              </span>
              <button
                v-if="activeCanPause"
                type="button"
                data-test="queue-active-pause"
                class="ms-toolbar-button ms-toolbar-button--icon"
                :class="activePaused ? 'border-accent text-accent' : ''"
                :title="activePauseTitle"
                :aria-label="activePauseTitle"
                @click.stop="commands.togglePauseFor(activeHostId)"
              >
                <Icon :name="activePaused ? 'play' : 'pause'" :size="11" />
              </button>
              <button
                v-if="commands.canCancel(active)"
                type="button"
                data-test="queue-active-stop"
                class="ms-toolbar-button ms-toolbar-button--icon ms-toolbar-button--danger-hover"
                title="Stop this image"
                aria-label="Stop this image"
                @click.stop="commands.cancel(active)"
              >
                <Icon name="close" :size="11" />
              </button>
            </div>
          </div>
        </div>
        <div class="flex items-center gap-1.5">
          <span
            v-if="batchPosition(active)"
            data-test="queue-active-batch"
            class="font-mono text-micro text-fg-dim"
          >
            {{ batchPosition(active) }}
          </span>
          <span class="flex-1" />
          <button
            type="button"
            data-test="queue-explain"
            class="text-micro font-medium text-accent"
            @click.stop="explainOpen = !explainOpen"
          >
            What's this?
          </button>
        </div>
        <p v-if="explainOpen" class="text-micro leading-body text-fg-2" style="text-wrap: pretty">
          Your picture starts as random static and gets cleared up pass by pass — big shapes first,
          fine detail last. That's why the thumbnail looks soft until the end.
        </p>
      </div>

      <!-- waiting and finished rows -->
      <div
        v-for="row in rows"
        :key="row.key"
        :data-test="`queue-row-${row.kind}`"
        class="flex shrink-0 cursor-pointer items-center gap-2 rounded-control border border-border p-2 transition-colors duration-100 hover:bg-row-hover focus-visible:outline-2 focus-visible:outline-accent"
        role="button"
        tabindex="0"
        @click="commands.open(row)"
        @keydown.enter.prevent="commands.open(row)"
        @keydown.space.prevent="commands.open(row)"
        @contextmenu.prevent="commands.contextMenu($event, row)"
      >
        <span
          class="flex h-[38px] w-[38px] shrink-0 items-center justify-center overflow-hidden border border-border bg-bg-crust font-mono text-xs"
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
            v-else-if="thumbSrc(row)"
            :src="thumbSrc(row)!"
            alt=""
            class="h-full w-full object-cover"
          />
          <span v-else>{{ rowGlyph(row) }}</span>
        </span>
        <div class="flex min-w-0 flex-1 flex-col gap-0.5">
          <span class="truncate text-micro font-medium text-fg">{{ title(row) }}</span>
          <span class="truncate text-micro" :class="rowTone(row)">{{ status(row) }}</span>
        </div>
        <QueueRowMenu :row="row" />
      </div>
    </div>

    <p class="shrink-0 px-0.5 pt-2 text-micro leading-snug text-fg-dim">
      Finished images go to <strong class="font-semibold text-fg-2">My images</strong>. Closing the
      window keeps the queue running.
    </p>
  </div>
</template>
