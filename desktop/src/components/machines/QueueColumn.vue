<script setup lang="ts">
/*
 * The Machines overview queue column (spec §06 · G2) — running + queued work
 * across every connected host, read from the shared jobs.queueSurface so it
 * shows exactly what the Create activity strip shows. Queued rows can be
 * cancelled, and reordered up/down when the owning host advertises
 * queue.can_reorder. Read-only mirror otherwise; the server is the authority.
 */
import { computed, ref } from "vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import Icon from "@ui/components/Icon.vue";
import { useGenerationStore, jobProgress } from "../../stores/generation";
import { useJobsStore, type QueueSurfaceRow } from "../../stores/jobs";
import { useHostsStore } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import { useContextMenuStore, type MenuEntry } from "../../stores/contextMenu";

const generation = useGenerationStore();
const jobs = useJobsStore();
const toasts = useToastStore();
const contextMenu = useContextMenuStore();
const cancellingIds = ref<string[]>([]);
const retryingIds = ref<string[]>([]);

const rows = computed(() => jobs.queueSurface);
const hostsWithMore = computed(() => {
  const hosts = useHostsStore();
  return hosts.all.filter((host) => jobs.queues[host.id]?.nextCursor);
});

function ownJob(row: QueueSurfaceRow) {
  const clientId = row.entry.clientId;
  if (clientId === null) return null;
  return generation.jobs.find((j) => j.clientId === clientId) ?? null;
}

function promptFor(row: QueueSurfaceRow): string {
  return ownJob(row)?.prompt ?? row.entry.model;
}

/** developing 18/28 · This Mac (own denoising) → developing/finalizing/queued/held · host.
 *  Held is not waiting for a turn — the host parked it and will not start it —
 *  so it must never borrow the word that means "your turn is coming". */
function statusLine(row: QueueSurfaceRow): string {
  const job = ownJob(row);
  if (row.entry.state === "held") {
    const reason = row.entry.held_reason?.trim();
    return `held${reason ? ` (${reason})` : ""} · ${row.hostLabel}`;
  }
  if (row.entry.state === "paused") return `paused after restart · ${row.hostLabel}`;
  if (row.entry.state === "queued" && jobs.queues[row.hostId]?.paused === true) {
    return `paused · ${row.hostLabel}`;
  }
  const state =
    row.entry.state === "running"
      ? job?.status === "finishing"
        ? "finalizing"
        : job && job.total > 0 && job.status === "denoising"
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
  if (cancellingIds.value.includes(row.entry.id)) return;
  cancellingIds.value = [...cancellingIds.value, row.entry.id];
  try {
    const cancelled =
      row.entry.clientId !== null
        ? await generation.cancel(row.entry.clientId)
        : await jobs.cancelJob(row.hostId, row.entry.id).then(() => true);
    if (cancelled) toasts.push("Cancelled");
  } catch (err) {
    toasts.push(String(err), "error");
  } finally {
    cancellingIds.value = cancellingIds.value.filter((id) => id !== row.entry.id);
  }
}

async function togglePause(row: QueueSurfaceRow) {
  const snapshot = jobs.queues[row.hostId];
  try {
    if (snapshot?.paused || snapshot?.entries.some((entry) => entry.state === "paused")) {
      await jobs.resume(row.hostId);
      toasts.push(`Queue resumed on ${row.hostLabel}`);
    } else {
      await jobs.pause(row.hostId);
      toasts.push(`Queue paused on ${row.hostLabel} — running job finishes`);
    }
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  }
}

async function retry(row: QueueSurfaceRow) {
  if (retryingIds.value.includes(row.entry.id)) return;
  retryingIds.value = [...retryingIds.value, row.entry.id];
  try {
    const entry = await jobs.queueJob(row.hostId, row.entry.id);
    await jobs.retryJob(row.hostId, entry);
    toasts.push("Retry queued");
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  } finally {
    retryingIds.value = retryingIds.value.filter((id) => id !== row.entry.id);
  }
}

function openQueueMenu(row: QueueSurfaceRow, event: MouseEvent) {
  const snapshot = jobs.queues[row.hostId];
  const paused =
    snapshot?.paused === true || snapshot?.entries.some((entry) => entry.state === "paused");
  const items: MenuEntry[] = [];
  if (snapshot?.caps?.canPause || paused) {
    items.push({
      label: paused ? "Resume queue" : "Pause queue",
      action: () => void togglePause(row),
    });
  }
  if (row.entry.state === "held" && row.entry.retryable === true) {
    items.push({ label: "Retry job", action: () => void retry(row) });
  }
  items.push({
    label: row.entry.state === "running" ? "Stop job" : "Cancel job",
    danger: true,
    disabled: row.entry.state === "running" && !row.canCancelRunning,
    action: () => void cancel(row),
  });
  contextMenu.open(event, items);
}

/** This job's slot among its host's QUEUED rows — the index space the reorder
 *  PATCH uses. `entry.position` counts running jobs too, so nudging against it
 *  is off-by-N (or a no-op) the moment anything on the host is running. */
function queuedIndexOf(row: QueueSurfaceRow): number {
  return (
    jobs.queues[row.hostId]?.entries
      .filter((entry) => entry.state === "queued")
      .findIndex((entry) => entry.id === row.entry.id) ?? -1
  );
}

/** Nudge a queued job earlier/later; the server clamps out-of-range indices. */
async function reorder(row: QueueSurfaceRow, delta: number) {
  const target = Math.max(0, queuedIndexOf(row) + delta);
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
        @contextmenu="openQueueMenu(row, $event)"
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
          <div
            v-if="
              row.entry.state === 'queued' ||
              row.entry.state === 'paused' ||
              row.entry.state === 'held' ||
              (row.entry.state === 'running' && row.canCancelRunning)
            "
            class="flex shrink-0 items-center gap-0.5"
          >
            <template v-if="row.entry.state === 'queued' && row.canReorder">
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
              :disabled="cancellingIds.includes(row.entry.id)"
              class="ml-1 rounded-control px-2 py-1 text-caption text-ink-3 transition-colors hover:text-stop"
              @click="cancel(row)"
            >
              {{ cancellingIds.includes(row.entry.id) ? "Cancelling…" : "Cancel" }}
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
    <div v-if="hostsWithMore.length" class="flex flex-col gap-1.5">
      <button
        v-for="host in hostsWithMore"
        :key="host.id"
        type="button"
        data-test="queue-column-load-more"
        class="border-edge h-8 rounded-control border px-3 text-caption text-ink-2 hover:text-ink disabled:opacity-50"
        :disabled="jobs.queues[host.id]?.loadingMore"
        @click="jobs.loadMoreHost(host.id)"
      >
        {{
          jobs.queues[host.id]?.loadingMore
            ? `Loading from ${host.label}…`
            : `Load more from ${host.label}`
        }}
      </button>
      <p
        v-for="host in hostsWithMore.filter((host) => jobs.queues[host.id]?.loadMoreError)"
        :key="`error:${host.id}`"
        class="text-caption text-stop"
      >
        {{ jobs.queues[host.id]?.loadMoreError }}
      </p>
    </div>
  </div>
</template>
