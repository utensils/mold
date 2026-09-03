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
import { rowGlyph, rowStatusLine, rowTitle, rowTone } from "../../lib/queueRows";
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

const explainOpen = ref(false);

const active = computed(() => queue.active.value);
const rows = computed(() =>
  queue.rows.value.filter((row) => row.key !== active.value?.key).slice(0, 12),
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
  if (row.kind === "sequence" && row.sequence.stageCount > 0) {
    return Math.round((row.sequence.currentStage / row.sequence.stageCount) * 100);
  }
  return null;
}
</script>

<template>
  <div data-test="queue-rail" class="flex flex-col border-t border-border pt-2">
    <div class="flex shrink-0 items-center gap-2 px-0.5 pb-2">
      <span class="group-label">Queue</span>
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
        class="rail-button"
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
        class="rail-button hover:border-error hover:text-error"
        title="Stop everything"
        aria-label="Stop everything"
        :disabled="queue.liveCount.value === 0"
        @click="commands.stopEverything()"
      >
        <Icon name="stop" :size="12" />
      </button>
      <button
        type="button"
        data-test="queue-open"
        class="rail-button"
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
        class="flex shrink-0 cursor-pointer flex-col gap-2 rounded-control bg-panel-raised p-2.5 shadow-[inset_0_0_0_1px_var(--mold-blue)]"
        @click="commands.open(active)"
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
            <span class="truncate text-micro text-fg-2">{{ rowStatusLine(active) }}</span>
            <div class="flex items-center gap-1.5">
              <span class="block h-1.5 flex-1 overflow-hidden bg-bg-crust" aria-hidden="true">
                <span
                  class="block h-full bg-accent"
                  :style="{ width: `${progressPct(active) ?? 8}%` }"
                  :class="progressPct(active) === null ? 'ms-pulse' : ''"
                />
              </span>
              <button
                v-if="commands.canCancel(active)"
                type="button"
                data-test="queue-active-stop"
                class="rail-button h-6 w-6 hover:border-error hover:text-error"
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
          <span class="font-mono text-micro text-fg-dim">{{ rowGlyph(active) }}</span>
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
        <p
          v-if="explainOpen"
          class="text-micro leading-relaxed text-fg-2"
          style="text-wrap: pretty"
        >
          Your picture starts as random static and gets cleared up pass by pass — big shapes first,
          fine detail last. That's why the thumbnail looks soft until the end.
        </p>
      </div>

      <!-- waiting and finished rows -->
      <div
        v-for="row in rows"
        :key="row.key"
        :data-test="`queue-row-${row.kind}`"
        class="flex shrink-0 cursor-pointer items-center gap-2 rounded-control border border-border p-2 transition-colors duration-100 hover:bg-row-hover"
        @click="commands.open(row)"
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
          <span class="truncate text-micro" :class="rowTone(row)">{{ rowStatusLine(row) }}</span>
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

<style scoped>
.group-label {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--mold-text-dim);
}

.rail-button {
  display: inline-flex;
  width: 26px;
  height: 24px;
  align-items: center;
  justify-content: center;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  color: var(--mold-text-2);
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}
.rail-button:hover:not(:disabled) {
  border-color: var(--mold-border-focus);
  color: var(--mold-text);
}
.rail-button:disabled {
  color: var(--mold-text-faint);
}
</style>
