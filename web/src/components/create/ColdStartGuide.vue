<script setup lang="ts">
/*
 * Cold start (spec §08 G10). When this machine has no installed models the
 * develop bed can't do anything useful, so the empty canvas becomes a guide:
 * one line of what's happening plus a few one-tap starter pulls that get a
 * first print within three steps. A starter that is downloading shows its
 * inline progress in place of its Pull button; when it lands the canvas
 * returns to the normal empty state (the parent stops rendering this).
 */
import { computed } from "vue";
import { STARTER_MODELS } from "@mold/studio";
import Icon from "@ui/components/Icon.vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import { useDownloads } from "../../composables/useDownloads";
import { toast } from "../../lib/toasts";
import type { DownloadJobWire } from "../../types";

// Recommended-first: a fast, capable default, then the smallest/fastest, then a
// familiar classic. Names are canonical manifest ids the downloads API accepts.
const starters = STARTER_MODELS;

const downloads = useDownloads();

function jobFor(name: string): DownloadJobWire | null {
  return (
    downloads.activeJobs.value.find((j) => j.model === name) ??
    downloads.queued.value.find((j) => j.model === name) ??
    null
  );
}

function pct(job: DownloadJobWire): number {
  if (!job.bytes_total) return 0;
  return Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100));
}

function statusLabel(job: DownloadJobWire): string {
  if (job.status === "queued") return "queued";
  if (!job.bytes_total) return "starting";
  return `pulling ${pct(job)}%`;
}

const anyPulling = computed(() =>
  starters.some((s) => jobFor(s.model) !== null),
);

async function pull(model: string) {
  try {
    await downloads.enqueue(model);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    toast("error", `couldn't pull ${model} — ${detail}`);
  }
}
</script>

<template>
  <div class="cold" data-test="cold-start">
    <div class="cold__frame">
      <Icon name="download" :size="38" :stroke-width="1.3" />
    </div>
    <h3 class="cold__headline">pull a model to start</h3>
    <p class="cold__guidance">
      mold runs models locally on this machine's GPU.
    </p>

    <div class="cold__starters">
      <div
        v-for="s in starters"
        :key="s.model"
        class="cold__card"
        :data-test="`starter-${s.model}`"
      >
        <div class="cold__meta">
          <div class="cold__name-row">
            <span class="cold__name">{{ s.displayName }}</span>
            <span v-if="s.recommended" class="cold__rec">recommended</span>
          </div>
          <span class="cold__note">{{ s.speed }} · {{ s.size }}</span>
        </div>

        <div
          v-if="jobFor(s.model)"
          class="cold__progress"
          :data-test="`starter-progress-${s.model}`"
        >
          <ProgressBar
            :value="pct(jobFor(s.model)!)"
            tone="accent"
            :height="6"
          />
          <span class="cold__status">{{ statusLabel(jobFor(s.model)!) }}</span>
        </div>
        <button
          v-else
          type="button"
          class="cold__pull"
          :data-test="`starter-pull-${s.model}`"
          @click="pull(s.model)"
        >
          Pull
        </button>
      </div>
    </div>

    <p v-if="anyPulling" class="cold__hint">
      keep this tab open — models land in your library when the pull finishes.
    </p>
  </div>
</template>

<style scoped>
.cold {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  width: 100%;
  max-width: 420px;
  margin: 0 auto;
}

.cold__frame {
  width: 92px;
  height: 92px;
  margin-bottom: 18px;
  border: 1.5px dashed var(--ce);
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--ink-3);
}

.cold__headline {
  margin: 0 0 7px;
  font-family: var(--f-display);
  font-size: 20px;
  font-weight: 600;
  color: var(--rebate);
}

.cold__guidance {
  margin: 0 0 20px;
  font-size: 13.5px;
  line-height: 1.5;
  color: var(--ink-3);
}

.cold__starters {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 9px;
}

.cold__card {
  display: flex;
  align-items: center;
  gap: 12px;
  text-align: left;
  background: var(--bath);
  border: 1px solid var(--edge);
  border-radius: var(--radius-control-lg);
  padding: 12px 14px;
}

.cold__meta {
  flex: 1;
  min-width: 0;
}

.cold__name-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.cold__name {
  font-family: var(--f-display);
  font-size: 14px;
  font-weight: 600;
  color: var(--rebate);
}

.cold__rec {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--sel-ink);
  background: var(--sel-bg);
  padding: 1px 7px;
  border-radius: var(--radius-pill);
}

.cold__note {
  display: block;
  margin-top: 3px;
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}

.cold__progress {
  flex: 0 0 132px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.cold__status {
  font-family: var(--f-mono);
  font-size: 10.5px;
  color: var(--safelight);
}

.cold__pull {
  flex: 0 0 auto;
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 700;
  padding: 8px 18px;
  border-radius: var(--radius-control);
  cursor: pointer;
  transition: filter var(--dur-quick) var(--ease);
}

.cold__pull:hover {
  filter: brightness(1.06);
}

.cold__pull:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.cold__hint {
  margin: 16px 0 0;
  font-size: 12px;
  color: var(--ink-3);
}
</style>
