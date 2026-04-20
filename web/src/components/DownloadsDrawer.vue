<script setup lang="ts">
import { computed } from "vue";
import type { DownloadJobWire } from "../types";

const props = defineProps<{
  open: boolean;
  active: DownloadJobWire | null;
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
  etaSeconds: number | null;
}>();

const emit = defineEmits<{
  (e: "close"): void;
  (e: "cancel", id: string): void;
  (e: "retry", model: string): void;
}>();

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

const activePct = computed(() => {
  const a = props.active;
  if (!a || a.bytes_total === 0) return 0;
  return Math.min(100, Math.round((a.bytes_done / a.bytes_total) * 100));
});
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

    <!-- Active -->
    <section
      v-if="active"
      class="rounded-2xl border border-white/5 bg-white/5 p-3"
    >
      <div class="mb-2 text-xs uppercase tracking-wider text-ink-300">
        Active
      </div>
      <div class="flex items-center justify-between">
        <div class="text-sm font-medium text-ink-50">{{ active.model }}</div>
        <button
          class="rounded-full bg-red-500/20 px-2 py-0.5 text-xs text-red-200 hover:bg-red-500/40"
          @click="emit('cancel', active.id)"
        >
          Cancel
        </button>
      </div>
      <div class="mt-1 text-xs text-ink-300">
        {{ formatGb(active.bytes_done) }} / {{ formatGb(active.bytes_total) }} ·
        {{ active.files_done }}/{{ active.files_total }} · ETA
        {{ formatEta(etaSeconds) }}
      </div>
      <div
        class="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/10"
        role="progressbar"
        :aria-valuenow="activePct"
        aria-valuemin="0"
        aria-valuemax="100"
      >
        <div
          class="h-full bg-brand-400 transition-[width]"
          :style="{ width: activePct + '%' }"
        />
      </div>
      <div
        v-if="active.current_file"
        class="mt-2 truncate text-xs text-ink-400"
      >
        {{ active.current_file }}
      </div>
    </section>

    <!-- Queued -->
    <section
      v-if="queued.length"
      class="rounded-2xl border border-white/5 bg-white/5 p-3"
    >
      <div class="mb-2 text-xs uppercase tracking-wider text-ink-300">
        Queued ({{ queued.length }})
      </div>
      <ul class="flex flex-col gap-1">
        <li
          v-for="(job, idx) in queued"
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
      v-if="history.length"
      class="rounded-2xl border border-white/5 bg-white/5 p-3"
    >
      <div class="mb-2 text-xs uppercase tracking-wider text-ink-300">
        Recent
      </div>
      <ul class="flex flex-col gap-1">
        <li
          v-for="job in [...history].reverse()"
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
      v-if="!active && !queued.length && !history.length"
      class="text-center text-sm text-ink-400"
    >
      No downloads yet.
    </p>
  </aside>
</template>
