<script setup lang="ts">
/*
 * The Machines overview queue column (spec §06 · G2) — running + queued work
 * across every connected host, read from the shared jobs.queueSurface so it
 * shows exactly what the Create activity strip shows. Queued rows can be
 * cancelled, and reordered up/down when the owning host advertises
 * queue.can_reorder. Read-only mirror otherwise; the server is the authority.
 */
import { computed } from "vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import Icon from "@ui/components/Icon.vue";
import { useGenerationStore, jobProgress } from "../../stores/generation";
import { useJobsStore, type QueueSurfaceRow } from "../../stores/jobs";
import { useToastStore } from "../../stores/toasts";

const generation = useGenerationStore();
const jobs = useJobsStore();
const toasts = useToastStore();

const rows = computed(() => jobs.queueSurface);

function ownJob(row: QueueSurfaceRow) {
  const clientId = row.entry.clientId;
  if (clientId === null) return null;
  return generation.jobs.find((j) => j.clientId === clientId) ?? null;
}

function promptFor(row: QueueSurfaceRow): string {
  return ownJob(row)?.prompt ?? row.entry.model;
}

/** developing 18/28 · This Mac (own denoising) → developing/queued · host. */
function statusLine(row: QueueSurfaceRow): string {
  const job = ownJob(row);
  const state =
    row.entry.state === "running"
      ? job && job.total > 0 && job.status === "denoising"
        ? `developing ${job.step}/${job.total}`
        : "developing"
      : "queued";
  return `${state} · ${row.hostLabel}`;
}

function progressPct(row: QueueSurfaceRow): number {
  const job = ownJob(row);
  return job ? Math.round(jobProgress(job) * 100) : 0;
}

async function cancel(row: QueueSurfaceRow) {
  try {
    if (row.entry.clientId !== null) await generation.cancel(row.entry.clientId);
    else await jobs.cancelJob(row.hostId, row.entry.id);
    toasts.push("Cancelled");
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

/** Nudge a queued job earlier/later; the server clamps out-of-range indices. */
async function reorder(row: QueueSurfaceRow, delta: number) {
  const target = Math.max(0, row.entry.position + delta);
  await jobs.reorderQueued(row.hostId, row.entry.id, target);
}
</script>

<template>
  <div data-test="queue-column" class="flex flex-col gap-3">
    <div class="edge-code uppercase">Queue</div>
    <template v-if="rows.length">
      <div
        v-for="row in rows"
        :key="`${row.hostId}:${row.entry.id}`"
        data-test="queue-surface-row"
        class="border-edge rounded-chrome border bg-bench p-3.5 shadow-[inset_0_1px_0_var(--card-hi)]"
      >
        <div class="flex items-center gap-3">
          <span
            class="h-9 w-9 shrink-0 overflow-hidden rounded-media"
            :class="row.entry.state === 'running' ? 'ms-shimmer' : 'border-edge border bg-bath'"
          >
            <img
              v-if="ownJob(row)?.previewUrl"
              :src="ownJob(row)!.previewUrl!"
              alt=""
              class="h-full w-full object-cover"
              style="filter: blur(1px)"
            />
          </span>
          <div class="min-w-0 flex-1">
            <div class="truncate text-body text-ink" :title="promptFor(row)">
              {{ promptFor(row) }}
            </div>
            <div
              class="data-mono mt-0.5 truncate text-caption"
              :class="row.entry.state === 'running' ? 'text-safelight' : 'text-ink-3'"
            >
              {{ statusLine(row) }}
            </div>
          </div>
          <div v-if="row.entry.state === 'queued'" class="flex shrink-0 items-center gap-0.5">
            <template v-if="row.canReorder">
              <button
                type="button"
                data-test="queue-reorder-up"
                class="rounded-control p-1 text-ink-3 transition-colors hover:text-ink"
                aria-label="Move earlier"
                @click="reorder(row, -1)"
              >
                <Icon name="chevron-up" :size="16" :stroke-width="2" />
              </button>
              <button
                type="button"
                data-test="queue-reorder-down"
                class="rounded-control p-1 text-ink-3 transition-colors hover:text-ink"
                aria-label="Move later"
                @click="reorder(row, 1)"
              >
                <Icon name="chevron-down" :size="16" :stroke-width="2" />
              </button>
            </template>
            <button
              type="button"
              data-test="queue-cancel"
              class="ml-1 rounded-control px-2 py-1 text-caption text-ink-3 transition-colors hover:text-stop"
              @click="cancel(row)"
            >
              Cancel
            </button>
          </div>
        </div>
        <ProgressBar
          v-if="row.entry.state === 'running'"
          :value="progressPct(row)"
          tone="accent"
          :height="5"
          class="mt-3"
        />
      </div>
    </template>
    <p v-else data-test="queue-empty" class="text-caption text-ink-3">
      Nothing running or queued right now.
    </p>
  </div>
</template>
