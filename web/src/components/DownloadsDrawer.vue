<script setup lang="ts">
import { computed } from "vue";
import type { DownloadJobWire } from "../types";

const props = defineProps<{
  open: boolean;
  active: DownloadJobWire[];
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
  etaSeconds: number | null;
}>();

const emit = defineEmits<{
  (e: "close"): void;
  (e: "cancel", id: string): void;
  (e: "retry", model: string): void;
}>();

type DownloadGroup = {
  id: string;
  jobs: DownloadJobWire[];
  bytesDone: number;
  bytesTotal: number;
  filesDone: number;
  filesTotal: number;
  status: DownloadJobWire["status"];
  currentFile: string | null;
};

function formatGb(bytes: number): string {
  if (!bytes) return "—";
  const gb = bytes / 1_073_741_824;
  return gb >= 1
    ? `${gb.toFixed(1)} GB`
    : `${(bytes / 1_048_576).toFixed(0)} MB`;
}

function formatEta(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

const looseActive = computed(() =>
  props.active.filter((job) => !job.catalog_id),
);

const looseQueued = computed(() =>
  props.queued.filter((job) => !job.catalog_id),
);

const looseHistory = computed(() =>
  props.history.filter((job) => !job.catalog_id || job.status !== "completed"),
);

function activePct(job: DownloadJobWire): number {
  if (job.bytes_total === 0) return 0;
  return Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100));
}

const catalogGroups = computed<DownloadGroup[]>(() => {
  const byId = new Map<string, DownloadJobWire[]>();
  for (const job of [...props.active, ...props.queued, ...props.history]) {
    if (!job.catalog_id) continue;
    const jobs = byId.get(job.catalog_id) ?? [];
    jobs.push(job);
    byId.set(job.catalog_id, jobs);
  }

  const rank: Record<DownloadJobWire["status"], number> = {
    active: 0,
    queued: 1,
    failed: 2,
    cancelled: 3,
    completed: 4,
  };

  return [...byId.entries()]
    .map(([id, jobs]) => {
      const ordered = [...jobs].sort((a, b) => rank[a.status] - rank[b.status]);
      const status: DownloadJobWire["status"] = ordered.some(
        (job) => job.status === "active",
      )
        ? "active"
        : ordered.some((job) => job.status === "queued")
          ? "queued"
          : ordered.some((job) => job.status === "failed")
            ? "failed"
            : ordered.some((job) => job.status === "cancelled")
              ? "cancelled"
              : "completed";
      return {
        id,
        jobs: ordered,
        bytesDone: jobs.reduce((sum, job) => sum + job.bytes_done, 0),
        bytesTotal: jobs.reduce((sum, job) => sum + job.bytes_total, 0),
        filesDone: jobs.reduce((sum, job) => sum + job.files_done, 0),
        filesTotal: jobs.reduce((sum, job) => sum + job.files_total, 0),
        status,
        currentFile:
          ordered.find((job) => job.status === "active")?.current_file ?? null,
      };
    })
    .filter((group) => group.status !== "completed")
    .sort((a, b) => rank[a.status] - rank[b.status]);
});

function groupPct(group: DownloadGroup): number {
  if (group.bytesTotal === 0) return 0;
  return Math.min(100, Math.round((group.bytesDone / group.bytesTotal) * 100));
}

function jobLabel(job: DownloadJobWire): string {
  if (job.catalog_id && job.model === job.catalog_id) return "Primary model";
  return job.model;
}
</script>

<template>
  <aside
    v-if="open"
    class="glass fixed inset-y-4 right-4 z-40 flex w-[min(420px,90vw)] flex-col gap-4 overflow-y-auto rounded-3xl p-5"
    aria-label="Downloads"
  >
    <header class="flex items-center justify-between">
      <h2 class="text-sm font-semibold uppercase tracking-wider text-ink-200">
        Downloads
      </h2>
      <button
        class="rounded-full border border-white/5 bg-white/5 px-3 py-1 text-xs text-ink-200 hover:text-white"
        @click="emit('close')"
      >
        Close
      </button>
    </header>

    <!-- Catalog Groups -->
    <section
      v-for="group in catalogGroups"
      :key="group.id"
      class="rounded-2xl border border-white/5 bg-white/5 p-3"
    >
      <div class="mb-2 flex items-center justify-between">
        <div class="text-xs uppercase tracking-wider text-ink-300">
          Catalog model
        </div>
        <span
          class="rounded-full px-1.5 py-0.5 text-[10px] uppercase"
          :class="{
            'bg-brand-500/20 text-brand-100': group.status === 'active',
            'bg-white/10 text-ink-300': group.status === 'queued',
            'bg-red-500/20 text-red-200': group.status === 'failed',
            'bg-slate-500/30 text-slate-200': group.status === 'cancelled',
          }"
        >
          {{ group.status }}
        </span>
      </div>
      <div class="flex items-center justify-between">
        <div class="min-w-0 truncate text-sm font-medium text-ink-50">
          {{ group.id }}
        </div>
        <button
          v-if="group.status === 'active'"
          class="rounded-full bg-red-500/20 px-2 py-0.5 text-xs text-red-200 hover:bg-red-500/40"
          @click="
            emit(
              'cancel',
              group.jobs.find((job) => job.status === 'active')?.id ??
                group.jobs[0].id,
            )
          "
        >
          Cancel
        </button>
      </div>
      <div class="mt-1 text-xs text-ink-300">
        {{ formatGb(group.bytesDone) }} / {{ formatGb(group.bytesTotal) }} ·
        {{ group.filesDone }}/{{ group.filesTotal }} files ·
        {{ groupPct(group) }}%
        <template v-if="active.some((job) => job.catalog_id === group.id)">
          · ETA {{ formatEta(etaSeconds) }}
        </template>
      </div>
      <div
        class="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/10"
        role="progressbar"
        :aria-valuenow="groupPct(group)"
        aria-valuemin="0"
        aria-valuemax="100"
      >
        <div
          class="h-full bg-brand-400 transition-[width]"
          :style="{ width: groupPct(group) + '%' }"
        />
      </div>
      <div v-if="group.currentFile" class="mt-2 truncate text-xs text-ink-400">
        {{ group.currentFile }}
      </div>
      <ul class="mt-3 flex flex-col gap-1 border-t border-white/5 pt-2">
        <li
          v-for="job in group.jobs"
          :key="job.id"
          class="grid grid-cols-[1fr_auto] gap-2 text-xs"
        >
          <span class="min-w-0 truncate text-ink-200">{{ jobLabel(job) }}</span>
          <span class="text-ink-400">
            {{ formatGb(job.bytes_total) }} · {{ job.status }}
          </span>
        </li>
      </ul>
    </section>

    <!-- Active -->
    <section
      v-for="job in looseActive"
      :key="job.id"
      class="rounded-2xl border border-white/5 bg-white/5 p-3"
    >
      <div class="mb-2 text-xs uppercase tracking-wider text-ink-300">
        Active
      </div>
      <div class="flex items-center justify-between">
        <div class="text-sm font-medium text-ink-50">
          {{ job.model }}
        </div>
        <button
          class="rounded-full bg-red-500/20 px-2 py-0.5 text-xs text-red-200 hover:bg-red-500/40"
          @click="emit('cancel', job.id)"
        >
          Cancel
        </button>
      </div>
      <div class="mt-1 text-xs text-ink-300">
        {{ formatGb(job.bytes_done) }} / {{ formatGb(job.bytes_total) }} ·
        {{ job.files_done }}/{{ job.files_total }} · ETA
        {{ formatEta(etaSeconds) }}
      </div>
      <div
        class="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/10"
        role="progressbar"
        :aria-valuenow="activePct(job)"
        aria-valuemin="0"
        aria-valuemax="100"
      >
        <div
          class="h-full bg-brand-400 transition-[width]"
          :style="{ width: activePct(job) + '%' }"
        />
      </div>
      <div v-if="job.current_file" class="mt-2 truncate text-xs text-ink-400">
        {{ job.current_file }}
      </div>
    </section>

    <!-- Queued -->
    <section
      v-if="looseQueued.length"
      class="rounded-2xl border border-white/5 bg-white/5 p-3"
    >
      <div class="mb-2 text-xs uppercase tracking-wider text-ink-300">
        Queued ({{ looseQueued.length }})
      </div>
      <ul class="flex flex-col gap-1">
        <li
          v-for="(job, idx) in looseQueued"
          :key="job.id"
          class="flex items-center justify-between text-sm"
        >
          <span class="truncate text-ink-100">{{ job.model }}</span>
          <span class="text-xs text-ink-400">#{{ idx + 1 }}</span>
          <button
            class="ml-2 rounded-full bg-white/10 px-2 py-0.5 text-xs text-ink-200 hover:text-white"
            :aria-label="'Cancel queued ' + job.model"
            @click="emit('cancel', job.id)"
          >
            ×
          </button>
        </li>
      </ul>
    </section>

    <!-- Recent -->
    <section
      v-if="looseHistory.length"
      class="rounded-2xl border border-white/5 bg-white/5 p-3"
    >
      <div class="mb-2 text-xs uppercase tracking-wider text-ink-300">
        Recent
      </div>
      <ul class="flex flex-col gap-1">
        <li
          v-for="job in [...looseHistory].reverse()"
          :key="job.id"
          class="flex items-center justify-between gap-2 text-sm"
        >
          <span class="flex min-w-0 items-center gap-2">
            <span
              class="rounded-full px-1.5 py-0.5 text-[10px] uppercase"
              :class="{
                'bg-emerald-500/20 text-emerald-200':
                  job.status === 'completed',
                'bg-red-500/20 text-red-200': job.status === 'failed',
                'bg-slate-500/30 text-slate-200': job.status === 'cancelled',
              }"
            >
              {{ job.status }}
            </span>
            <span class="truncate text-ink-100">{{ job.model }}</span>
          </span>
          <button
            v-if="job.status === 'failed'"
            :data-test="'retry-' + job.id"
            class="rounded-full bg-brand-500/20 px-2 py-0.5 text-xs text-brand-100 hover:bg-brand-500/40"
            @click="emit('retry', job.model)"
          >
            Retry
          </button>
        </li>
      </ul>
    </section>

    <p
      v-if="
        !looseActive.length &&
        !looseQueued.length &&
        !looseHistory.length &&
        !catalogGroups.length
      "
      class="text-center text-sm text-ink-400"
    >
      No downloads yet.
    </p>
  </aside>
</template>
