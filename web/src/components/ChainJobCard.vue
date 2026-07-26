<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import {
  cancelChainJob,
  chainJobStagePreviewUrl,
  resumeChainJob,
  retakeChainJob,
  type StreamTarget,
} from "../api";
import type { ChainJobDetail } from "../types";

const props = defineProps<{
  job: ChainJobDetail;
  target?: StreamTarget;
}>();

const emit = defineEmits<{
  updated: [];
  dismiss: [];
}>();

const actionBusy = ref<"resume" | "cancel" | "retake" | null>(null);
const actionError = ref<string | null>(null);
const previewUrls = ref<Record<number, string>>({});
const active = computed(
  () => props.job.state === "queued" || props.job.state === "running",
);
const resumable = computed(() =>
  ["failed", "interrupted", "cancelled"].includes(props.job.state),
);
const dismissible = computed(() =>
  ["completed", "failed", "cancelled"].includes(props.job.state),
);

watch(
  () => props.job.state,
  () => {
    actionBusy.value = null;
    actionError.value = null;
  },
);

watch(
  () =>
    props.job.stages
      .filter((stage) => stage.has_preview)
      .map((stage) => stage.idx)
      .join(","),
  () => {
    for (const stage of props.job.stages) {
      if (!stage.has_preview || previewUrls.value[stage.idx]) continue;
      const headers = props.target?.apiKey
        ? { "x-api-key": props.target.apiKey }
        : undefined;
      void fetch(
        chainJobStagePreviewUrl(props.job.id, stage.idx, props.target),
        {
          ...(headers ? { headers } : {}),
        },
      )
        .then(async (response) => {
          if (!response.ok)
            throw new Error(`Preview failed: ${response.status}`);
          const url = URL.createObjectURL(await response.blob());
          previewUrls.value = { ...previewUrls.value, [stage.idx]: url };
        })
        .catch(() => {});
    }
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  for (const url of Object.values(previewUrls.value)) URL.revokeObjectURL(url);
});

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function nextTransition(stageIdx: number): string {
  return props.job.script.stage[stageIdx + 1]?.transition ?? "cut";
}

function spliceDisabled(stageIdx: number): boolean {
  return nextTransition(stageIdx) === "smooth";
}

async function resume() {
  if (actionBusy.value) return;
  actionBusy.value = "resume";
  actionError.value = null;
  try {
    await resumeChainJob(props.job.id, props.target);
    emit("updated");
  } catch (error) {
    actionError.value = errorMessage(error);
  } finally {
    actionBusy.value = null;
  }
}

async function cancel() {
  if (!active.value || actionBusy.value) return;
  actionBusy.value = "cancel";
  actionError.value = null;
  try {
    await cancelChainJob(props.job.id, props.target);
    emit("updated");
  } catch (error) {
    actionError.value = errorMessage(error);
  } finally {
    actionBusy.value = null;
  }
}

async function retake(stageIdx: number, mode: "cascade" | "splice") {
  if (actionBusy.value || (mode === "splice" && spliceDisabled(stageIdx)))
    return;
  actionBusy.value = "retake";
  actionError.value = null;
  try {
    await retakeChainJob(
      props.job.id,
      { stage_idx: stageIdx, mode },
      props.target,
    );
    emit("updated");
  } catch (error) {
    actionError.value = errorMessage(error);
  } finally {
    actionBusy.value = null;
  }
}
</script>

<template>
  <article
    data-test="chain-job-card"
    class="rounded-lg border border-edge bg-bath/60 p-3"
  >
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div>
        <div class="text-sm font-medium text-rebate">{{ job.id }}</div>
        <div class="text-xs text-ink-3">
          {{ job.state }} - {{ job.current_stage }}/{{ job.stage_count }}
        </div>
      </div>
      <div class="flex gap-2">
        <button
          v-if="resumable"
          data-test="chain-job-resume"
          class="rounded border border-ce px-2 py-1 text-xs text-rebate disabled:opacity-40"
          :disabled="actionBusy !== null"
          @click="resume"
        >
          {{ actionBusy === "resume" ? "Resuming…" : "Resume" }}
        </button>
        <button
          v-if="active"
          data-test="chain-job-cancel"
          class="rounded border border-ce px-2 py-1 text-xs text-rebate disabled:opacity-40"
          :disabled="actionBusy !== null"
          @click="cancel"
        >
          {{ actionBusy === "cancel" ? "Cancelling…" : "Cancel" }}
        </button>
        <button
          v-else-if="dismissible"
          data-test="chain-job-dismiss"
          class="rounded border border-ce px-2 py-1 text-xs text-rebate"
          @click="emit('dismiss')"
        >
          Dismiss
        </button>
      </div>
    </div>

    <p
      v-if="job.error"
      data-test="chain-job-error"
      class="mt-3 rounded border border-red-400/30 bg-red-400/10 px-3 py-2 text-xs leading-relaxed text-red-200"
      role="alert"
    >
      {{ job.error }}
    </p>
    <p
      v-if="actionError"
      data-test="chain-job-action-error"
      class="mt-3 rounded border border-red-400/30 bg-red-400/10 px-3 py-2 text-xs text-red-200"
      role="alert"
    >
      {{ actionError }}
    </p>

    <ol data-test="chain-job-timeline" class="mt-3 grid gap-2">
      <li
        v-for="stage in job.stages"
        :key="stage.idx"
        class="grid grid-cols-[56px_1fr_auto] items-center gap-3 rounded border border-edge p-2"
        :data-test="`chain-job-stage-${stage.idx}`"
      >
        <img
          v-if="previewUrls[stage.idx]"
          :src="previewUrls[stage.idx]"
          alt=""
          class="h-10 w-14 rounded object-cover"
          :data-test="`chain-job-stage-preview-${stage.idx}`"
        />
        <div
          v-else
          class="h-10 w-14 rounded bg-surface"
          :data-test="`chain-job-stage-preview-empty-${stage.idx}`"
        />
        <div class="min-w-0">
          <div class="text-xs font-medium text-rebate">
            Clip {{ stage.idx + 1 }} · {{ stage.state }}
          </div>
          <div class="truncate text-xs text-ink-3">
            {{ job.script.stage[stage.idx]?.prompt }}
          </div>
          <div
            v-if="stage.error && stage.error !== job.error"
            class="mt-1 text-xs leading-relaxed text-red-200"
            :data-test="`chain-job-stage-error-${stage.idx}`"
          >
            {{ stage.error }}
          </div>
        </div>
        <div class="flex gap-2">
          <button
            class="rounded border border-edge px-2 py-1 text-xs text-ink-2"
            :data-test="`chain-job-retake-cascade-${stage.idx}`"
            :disabled="active || actionBusy !== null"
            @click="retake(stage.idx, 'cascade')"
          >
            Retake
          </button>
          <button
            class="rounded border border-edge px-2 py-1 text-xs text-ink-2 disabled:opacity-40"
            :data-test="`chain-job-retake-splice-${stage.idx}`"
            :disabled="
              active || actionBusy !== null || spliceDisabled(stage.idx)
            "
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
