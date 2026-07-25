<script setup lang="ts">
/*
 * Downloads panel body (spec §06, prototype web downloads popover). Three
 * sections — Downloading (name · % · progress bar · rate/eta), Queued, and a
 * divider then Recently installed (status glyph · name · size). Cancel on
 * active/queued,
 * retry on a failed history row. Presentational: the parent owns the download
 * state and wraps this in either an anchored panel or a bottom sheet.
 */
import Icon from "@ui/components/Icon.vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import type { DownloadJobWire, ModelInfoExtended } from "../../types";
import { modelDisplayNameForId } from "../../lib/modelName";

const props = defineProps<{
  active: DownloadJobWire[];
  queued: DownloadJobWire[];
  history: DownloadJobWire[];
  /** Seconds remaining, keyed by job id. */
  etaByJob: Record<string, number | null>;
  /** Bytes/sec, keyed by job id. */
  rateByJob: Record<string, number | null>;
  models?: ModelInfoExtended[];
}>();
const modelLabel = (name: string) =>
  modelDisplayNameForId(name, props.models ?? []);

const emit = defineEmits<{
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
          <span class="dl-active__name">{{ modelLabel(job.model) }}</span>
          <span class="dl-active__pct">{{ pct(job) }}%</span>
          <button
            type="button"
            class="dl-cancel"
            :data-test="`dl-cancel-${job.id}`"
            :aria-label="`Cancel ${modelLabel(job.model)}`"
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
        <span class="dl-row__name">{{ modelLabel(job.model) }}</span>
        <span class="dl-row__meta">#{{ idx + 1 }}</span>
        <button
          type="button"
          class="dl-cancel"
          :data-test="`dl-cancel-${job.id}`"
          :aria-label="`Cancel queued ${modelLabel(job.model)}`"
          @click="emit('cancel', job.id)"
        >
          <Icon name="close" :size="12" />
        </button>
      </div>
    </template>

    <!-- Recently installed -->
    <template v-if="history.length">
      <div v-if="active.length || queued.length" class="dl-divider" />
      <div class="dl-kicker">Recently installed</div>
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
        <span class="dl-row__name">{{ modelLabel(job.model) }}</span>
        <span v-if="job.status === 'completed'" class="dl-row__meta">
          {{ formatSize(job.bytes_total) }}
        </span>
        <span v-else class="dl-row__meta">{{ job.status }}</span>
        <button
          v-if="job.status === 'failed'"
          type="button"
          class="dl-retry"
          :data-test="`retry-${job.id}`"
          @click="emit('retry', job.model)"
        >
          Retry
        </button>
      </div>
    </template>

    <p v-if="isEmpty()" class="dl-empty">No downloads yet.</p>
  </div>
</template>

<style scoped>
.dl-body {
  display: flex;
  flex-direction: column;
}

.dl-kicker {
  font-family: var(--f-mono);
  font-size: 9px;
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
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12.5px;
  margin-bottom: 8px;
}

.dl-active__name {
  font-family: var(--f-mono);
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--rebate);
}

.dl-active__pct {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--safelight);
}

.dl-active__sub,
.dl-active__file {
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
  margin-top: 7px;
}

.dl-active__file {
  margin-top: 3px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ── Compact rows (queued / history) ────────────────────────────────── */
.dl-row {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 7px 0;
}

.dl-row__name {
  font-family: var(--f-mono);
  font-size: 12px;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--rebate);
}

.dl-row__meta {
  font-family: var(--f-mono);
  font-size: 10px;
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
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  font-size: 12px;
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
  flex: 0 0 auto;
  border: 1px solid var(--sel-border);
  background: var(--sel-bg);
  color: var(--sel-ink);
  font-family: var(--f-body);
  font-size: 11px;
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
  font-size: 12.5px;
  color: var(--ink-3);
  padding: 18px 0;
}
</style>
