<script setup lang="ts">
import {
  cancelChainJob,
  chainJobStagePreviewUrl,
  resumeChainJob,
  retakeChainJob,
} from "../api";
import type { ChainJobDetail } from "../types";

const props = defineProps<{
  job: ChainJobDetail;
}>();

const emit = defineEmits<{
  updated: [];
}>();

function nextTransition(stageIdx: number): string {
  return props.job.script.stage[stageIdx + 1]?.transition ?? "cut";
}

function spliceDisabled(stageIdx: number): boolean {
  return nextTransition(stageIdx) === "smooth";
}

async function resume() {
  await resumeChainJob(props.job.id);
  emit("updated");
}

async function cancel() {
  await cancelChainJob(props.job.id);
  emit("updated");
}

async function retake(stageIdx: number, mode: "cascade" | "splice") {
  if (mode === "splice" && spliceDisabled(stageIdx)) return;
  await retakeChainJob(props.job.id, { stage_idx: stageIdx, mode });
  emit("updated");
}
</script>

<template>
  <article
    data-test="chain-job-card"
    class="rounded-lg border border-slate-700 bg-slate-950/60 p-3"
  >
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div>
        <div class="text-sm font-medium text-slate-100">{{ job.id }}</div>
        <div class="text-xs text-slate-400">
          {{ job.state }} - {{ job.current_stage }}/{{ job.stage_count }}
        </div>
      </div>
      <div class="flex gap-2">
        <button
          data-test="chain-job-resume"
          class="rounded border border-slate-600 px-2 py-1 text-xs text-slate-100 disabled:opacity-40"
          :disabled="job.state === 'running' || job.state === 'completed'"
          @click="resume"
        >
          Resume
        </button>
        <button
          data-test="chain-job-cancel"
          class="rounded border border-slate-600 px-2 py-1 text-xs text-slate-100 disabled:opacity-40"
          :disabled="job.state === 'completed' || job.state === 'cancelled'"
          @click="cancel"
        >
          Cancel
        </button>
      </div>
    </div>

    <ol data-test="chain-job-timeline" class="mt-3 grid gap-2">
      <li
        v-for="stage in job.stages"
        :key="stage.idx"
        class="grid grid-cols-[56px_1fr_auto] items-center gap-3 rounded border border-slate-800 p-2"
        :data-test="`chain-job-stage-${stage.idx}`"
      >
        <img
          v-if="stage.has_preview"
          :src="chainJobStagePreviewUrl(job.id, stage.idx)"
          alt=""
          class="h-10 w-14 rounded object-cover"
          :data-test="`chain-job-stage-preview-${stage.idx}`"
        />
        <div
          v-else
          class="h-10 w-14 rounded bg-slate-800"
          :data-test="`chain-job-stage-preview-empty-${stage.idx}`"
        />
        <div class="min-w-0">
          <div class="text-xs font-medium text-slate-100">
            Stage {{ stage.idx }} - {{ stage.state }}
          </div>
          <div class="truncate text-xs text-slate-400">
            {{ job.script.stage[stage.idx]?.prompt }}
          </div>
        </div>
        <div class="flex gap-2">
          <button
            class="rounded border border-slate-700 px-2 py-1 text-xs text-slate-200"
            :data-test="`chain-job-retake-cascade-${stage.idx}`"
            @click="retake(stage.idx, 'cascade')"
          >
            Retake
          </button>
          <button
            class="rounded border border-slate-700 px-2 py-1 text-xs text-slate-200 disabled:opacity-40"
            :data-test="`chain-job-retake-splice-${stage.idx}`"
            :disabled="spliceDisabled(stage.idx)"
            :title="
              spliceDisabled(stage.idx)
                ? 'Splice retake requires the next transition to be cut or fade.'
                : 'Retake only this stage'
            "
            @click="retake(stage.idx, 'splice')"
          >
            Splice
          </button>
        </div>
      </li>
    </ol>
  </article>
</template>
