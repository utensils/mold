<script setup lang="ts">
/*
 * Downloads panel body (spec §06, prototype web downloads popover). Three
 * sections — Downloading (name · % · progress bar · rate/eta), Queued, and a
 * divider then Recent downloads (status glyph · name · size). Cancel on
 * active/queued,
 * retry on a failed history row. Presentational: the parent owns the download
 * state and wraps this in either an anchored panel or a bottom sheet.
 */
import Icon from "@ui/components/Icon.vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import type { DownloadJobWire, ModelInfoExtended } from "../../types";
import { modelDisplayNameForId } from "@studio/lib/modelDisplay";

const props = defineProps<{
  active: DownloadJobWire[];
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
  /** Seconds remaining, keyed by job id. */
  etaByJob: Record<string, number | null>;
  /** Bytes/sec, keyed by job id. */
  rateByJob: Record<string, number | null>;
  models?: ModelInfoExtended[];
  loaded?: boolean;
  loading?: boolean;
  error?: string | null;
  actionError?: string | null;
  actionBusy?: boolean;
}>();
const modelLabel = (name: string) =>
  modelDisplayNameForId(name, props.models ?? []);

const actionLabel = (name: string) => {
  const label = modelLabel(name);
  return label === name ? name : label + " (" + name + ")";
};

const emit = defineEmits<{
  (e: "refresh"): void;
  (e: "cancel", id: string): void;
  (e: "retry", model: string): void;
}>();

function formatSize(bytes: number): string {
  if (!bytes) return "—";
  const gb = bytes / 1_073_741_824;
  return gb >= 1
    ? `${gb.toFixed(1)} GB`
    : `${(bytes / 1_048_576).toFixed(0)} MB`;
}

function formatEta(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s}s`;
}

function formatRate(bps: number | null | undefined): string | null {
  if (bps == null || !Number.isFinite(bps) || bps <= 0) return null;
  return bps >= 1_000_000
    ? `${(bps / 1_000_000).toFixed(1)} MB/s`
    : `${Math.max(1, Math.round(bps / 1000))} KB/s`;
}

function pct(job: DownloadJobWire): number {
  if (!job.bytes_total) return 0;
  return Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100));
}

function subLine(job: DownloadJobWire): string {
  if (!job.bytes_total) return "Preparing download…";
  const parts = [`${job.files_done}/${job.files_total} files`];
  const rate = formatRate(props.rateByJob[job.id]);
  if (rate) parts.push(rate);
  parts.push(`eta ${formatEta(props.etaByJob[job.id])}`);
  return parts.join(" · ");
}

const isEmpty = () =>
  !props.active.length && !props.queued.length && !props.history.length;
</script>

<template>
  <div class="dl-body">
    <div v-if="error" class="dl-recovery" role="alert">
      <p>{{ error }}</p>
      <p v-if="loaded">Showing the last known downloads.</p>
      <button
        class="dl-retry"
        type="button"
        :disabled="loading"
        @click="emit('refresh')"
      >
        Retry loading
      </button>
    </div>
    <p v-else-if="loading || loaded === false" role="status">
      Loading downloads…
    </p>
    <p v-if="actionError" class="dl-error" role="alert">{{ actionError }}</p>
    <!-- Downloading -->
    <template v-if="active.length">
      <div class="dl-kicker">Downloading</div>
      <div
        v-for="job in active"
        :key="job.id"
        class="dl-active"
        :data-test="`dl-active-${job.id}`"
      >
        <div class="dl-active__head">
          <span class="dl-dot dl-dot--active" />
          <span class="dl-active__name"
            >{{ modelLabel(job.model)
            }}<small v-if="modelLabel(job.model) !== job.model">{{
              job.model
            }}</small></span
          >
          <span class="dl-active__pct">{{ pct(job) }}%</span>
          <button
            type="button"
            class="dl-cancel"
            :disabled="actionBusy"
            :data-test="`dl-cancel-${job.id}`"
            :aria-label="`Cancel ${actionLabel(job.model)}`"
            @click="emit('cancel', job.id)"
          >
            <Icon name="close" :size="12" />
          </button>
        </div>
        <ProgressBar :value="pct(job)" tone="accent" :height="6" />
        <div class="dl-active__sub">{{ subLine(job) }}</div>
        <div v-if="job.current_file" class="dl-active__file">
          {{ job.current_file }}
        </div>
      </div>
    </template>

    <!-- Queued -->
    <template v-if="queued.length">
      <div class="dl-kicker">Queued</div>
      <div
        v-for="(job, idx) in queued"
        :key="job.id"
        class="dl-row"
        :data-test="`dl-queued-${job.id}`"
      >
        <span class="dl-dot dl-dot--queued" />
        <span class="dl-row__name"
          >{{ modelLabel(job.model)
          }}<small v-if="modelLabel(job.model) !== job.model">{{
            job.model
          }}</small
          ><small v-if="job.error" class="dl-error">{{
            job.error
          }}</small></span
        >
        <span class="dl-row__meta">#{{ idx + 1 }}</span>
        <button
          type="button"
          class="dl-cancel"
          :disabled="actionBusy"
          :data-test="`dl-cancel-${job.id}`"
          :aria-label="`Cancel queued ${actionLabel(job.model)}`"
          @click="emit('cancel', job.id)"
        >
          <Icon name="close" :size="12" />
        </button>
      </div>
    </template>

    <!-- Recent downloads -->
    <template v-if="history.length">
      <div v-if="active.length || queued.length" class="dl-divider" />
      <div class="dl-kicker">Recent downloads</div>
      <div
        v-for="job in [...history].reverse()"
        :key="job.id"
        class="dl-row"
        :data-test="`dl-history-${job.id}`"
      >
        <span
          class="dl-glyph"
          :class="{
            'dl-glyph--ok': job.status === 'completed',
            'dl-glyph--bad':
              job.status === 'failed' || job.status === 'cancelled',
          }"
        >
          <Icon v-if="job.status === 'completed'" name="check" :size="12" />
          <Icon v-else name="close" :size="12" />
        </span>
        <span class="dl-row__name"
          >{{ modelLabel(job.model)
          }}<small v-if="modelLabel(job.model) !== job.model">{{
            job.model
          }}</small
          ><small v-if="job.error" class="dl-error">{{
            job.error
          }}</small></span
        >
        <span v-if="job.status === 'completed'" class="dl-row__meta">
          {{ formatSize(job.bytes_total) }}
        </span>
        <span v-else class="dl-row__meta">{{ job.status }}</span>
        <button
          v-if="job.status === 'failed'"
          type="button"
          class="dl-retry"
          :disabled="actionBusy"
          :data-test="`retry-${job.id}`"
          :aria-label="`Retry ${actionLabel(job.model)}`"
          @click="emit('retry', job.model)"
        >
          Retry
        </button>
      </div>
    </template>

    <p
      v-if="isEmpty() && loaded !== false && !loading && !error"
      class="dl-empty"
    >
      No downloads yet.
    </p>
  </div>
</template>

<style scoped>
.dl-body {
  display: flex;
  flex-direction: column;
}

.dl-kicker {
  font-family: var(--f-mono);
  font-size: 0.75rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
  margin: 2px 0 9px;
}

.dl-kicker:not(:first-child) {
  margin-top: 6px;
}

.dl-divider {
  height: 1px;
  background: var(--edge);
  margin: 10px 0;
}

/* ── Active row ─────────────────────────────────────────────────────── */
.dl-active {
  margin-bottom: 14px;
}

.dl-active__head {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px;
  font-size: 0.875rem;
  margin-bottom: 8px;
}

.dl-active__name {
  font-family: var(--f-mono);
  flex: 1;
  min-width: 0;
  overflow-wrap: anywhere;
  color: var(--rebate);
}

.dl-active__pct {
  font-family: var(--f-mono);
  font-size: 0.8125rem;
  color: var(--safelight);
}

.dl-active__sub,
.dl-active__file {
  font-family: var(--f-mono);
  font-size: 0.75rem;
  color: var(--ink-3);
  margin-top: 7px;
}

.dl-active__file {
  margin-top: 3px;
  overflow-wrap: anywhere;
}

/* ── Compact rows (queued / history) ────────────────────────────────── */
.dl-row {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 9px;
  padding: 7px 0;
}

.dl-row__name {
  font-family: var(--f-mono);
  font-size: 0.875rem;
  flex: 1;
  min-width: 0;
  overflow-wrap: anywhere;
  color: var(--rebate);
}

.dl-row__meta {
  font-family: var(--f-mono);
  font-size: 0.75rem;
  color: var(--ink-3);
}

/* ── Dots + glyphs ──────────────────────────────────────────────────── */
.dl-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex: 0 0 7px;
}

.dl-dot--active {
  background: var(--warning);
}

.dl-dot--queued {
  background: var(--ce);
}

.dl-glyph {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  line-height: 1;
  color: var(--ink-3);
}

.dl-glyph--ok {
  color: var(--safelight);
}

.dl-glyph--bad {
  color: var(--stop);
}

/* ── Actions ────────────────────────────────────────────────────────── */
.dl-cancel {
  min-width: 44px;
  min-height: 44px;
  justify-content: center;
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  font-size: 0.875rem;
  line-height: 1;
  padding: 2px 4px;
  border-radius: var(--radius-control-sm);
  cursor: pointer;
  transition: color var(--dur-quick) var(--ease);
}

.dl-cancel:hover {
  color: var(--stop);
}

.dl-retry {
  min-width: 44px;
  min-height: 44px;
  flex: 0 0 auto;
  border: 1px solid var(--sel-border);
  background: var(--sel-bg);
  color: var(--sel-ink);
  font-family: var(--f-body);
  font-size: 0.8125rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: var(--radius-pill);
  cursor: pointer;
}

.dl-cancel:focus-visible,
.dl-retry:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.dl-empty {
  text-align: center;
  font-size: 0.875rem;
  color: var(--ink-3);
  padding: 18px 0;
}
.dl-active__name small,
.dl-row__name small {
  display: block;
  font-size: 0.75rem;
  color: var(--ink-3);
  margin-top: 0.25rem;
}
.dl-error,
.dl-row__name .dl-error {
  color: var(--stop);
}
.dl-row__name,
.dl-active__name {
  grid-column: 2 / -1;
}
.dl-row__meta,
.dl-active__pct {
  grid-column: 2;
}
.dl-cancel,
.dl-retry {
  justify-self: end;
}
.dl-recovery .dl-retry {
  justify-self: start;
}
</style>
