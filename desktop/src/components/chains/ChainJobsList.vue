<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { useChainJobsStore } from "../../stores/chainJobs";
import { useToastStore } from "../../stores/toasts";
import { useHostModelsStore } from "../../stores/hostModels";
import type { ChainJobState, ChainJobSummary, RetakeMode } from "../../lib/api/types";
import { modelDisplayNameForId } from "../../lib/models";

const chains = useChainJobsStore();
const toasts = useToastStore();
const hostModels = useHostModelsStore();
const modelLabel = (name: string) => modelDisplayNameForId(name, hostModels.unionInstalled);

// State color follows the development-temperature rule (design spec §2.1).
const STATE_CLASS: Record<ChainJobState, string> = {
  queued: "text-halide",
  running: "text-safelight",
  interrupted: "text-halide",
  failed: "text-stop",
  completed: "text-ink",
  cancelled: "text-ink-3",
};

const resumable = (s: ChainJobState) => s === "failed" || s === "interrupted" || s === "cancelled";
const cancellable = (s: ChainJobState) => s === "running" || s === "queued";

// Per-job retake control: which job's retake chooser is open, and its stage.
const retakeFor = ref<string | null>(null);
const retakeStage = reactive<Record<string, number>>({});

function openRetake(job: ChainJobSummary) {
  retakeFor.value = retakeFor.value === job.id ? null : job.id;
  if (retakeStage[job.id] == null) retakeStage[job.id] = 0;
}

async function doRetake(job: ChainJobSummary, mode: RetakeMode) {
  const idx = Math.min(Math.max(0, retakeStage[job.id] ?? 0), job.stage_count - 1);
  retakeFor.value = null;
  try {
    await chains.retake(job.id, { stage_idx: idx, mode });
    toasts.push(`Retaking stage ${idx + 1} (${mode})`);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

async function runGc() {
  try {
    const outcome = await chains.gc();
    toasts.push(
      `Swept ${outcome.swept_ephemeral_jobs} jobs, pruned ${outcome.pruned_artifact_dirs} dirs`,
    );
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

const clearableCount = computed(
  () => chains.jobs.filter((j) => j.state !== "running" && j.state !== "queued").length,
);

async function clearInactive() {
  try {
    const { cleared, failed } = await chains.clearInactive();
    if (failed > 0) toasts.push(`Cleared ${cleared}, ${failed} could not be deleted`, "error");
    else toasts.push(`Cleared ${cleared} job${cleared === 1 ? "" : "s"}`);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}
</script>

<template>
  <div>
    <div class="mb-2 flex items-center gap-2">
      <span class="edge-code">Jobs</span>
      <div class="border-edge h-px flex-1 border-t" />
      <button
        v-if="clearableCount > 0"
        type="button"
        data-test="clear-inactive"
        class="edge-code text-ink-3 hover:text-ink active:translate-y-px"
        :title="`Delete all ${clearableCount} jobs that aren't running or queued — including resumable failed or interrupted ones`"
        @click="clearInactive"
      >
        Clear inactive
      </button>
      <button
        type="button"
        data-test="jobs-cleanup"
        class="edge-code text-ink-3 hover:text-ink active:translate-y-px"
        title="Sweep ephemeral jobs and prune artifact directories on the host"
        @click="runGc"
      >
        Clean up disk
      </button>
    </div>

    <p v-if="chains.jobs.length === 0" class="text-caption text-ink-3">No chain jobs yet.</p>

    <div
      v-for="job in chains.jobs"
      :key="job.id"
      class="border-edge -mx-1 flex flex-col gap-1 rounded-control border-b px-1 py-2 transition-colors duration-100 last:border-b-0 hover:bg-bath"
    >
      <div class="flex items-center gap-2">
        <span class="edge-code uppercase" :class="STATE_CLASS[job.state]">{{ job.state }}</span>
        <span class="truncate text-body text-ink" :title="modelLabel(job.model)">{{
          modelLabel(job.model)
        }}</span>
        <span class="data-mono ml-auto text-caption text-ink-3">
          stage {{ Math.min(job.current_stage + 1, job.stage_count) }}/{{ job.stage_count }}
        </span>
      </div>

      <p v-if="job.error" class="text-caption text-stop">{{ job.error }}</p>

      <div class="flex flex-wrap items-center gap-1">
        <button
          type="button"
          class="border-edge h-7 rounded-control border px-2 text-caption text-ink-2 hover:text-ink active:translate-y-px"
          @click="chains.watch(job.id)"
        >
          Watch
        </button>
        <button
          v-if="resumable(job.state)"
          type="button"
          class="border-edge h-7 rounded-control border px-2 text-caption text-ink-2 hover:text-ink active:translate-y-px"
          @click="chains.resume(job.id)"
        >
          Resume
        </button>
        <button
          v-if="cancellable(job.state)"
          type="button"
          class="border-edge h-7 rounded-control border px-2 text-caption text-ink-2 hover:text-stop active:translate-y-px"
          @click="chains.cancel(job.id)"
        >
          Cancel
        </button>
        <button
          v-if="job.state === 'completed'"
          type="button"
          class="border-edge h-7 rounded-control border px-2 text-caption text-ink-2 hover:text-ink active:translate-y-px"
          @click="openRetake(job)"
        >
          Retake
        </button>
        <button
          type="button"
          class="border-edge h-7 rounded-control border px-2 text-caption text-ink-2 hover:text-stop active:translate-y-px"
          @click="chains.remove(job.id)"
        >
          Delete
        </button>
      </div>

      <!-- Retake chooser -->
      <div v-if="retakeFor === job.id" class="border-edge mt-1 rounded-control border bg-bath p-2">
        <label class="flex items-center gap-2 text-caption text-ink-2">
          Stage
          <input
            v-model.number="retakeStage[job.id]"
            type="number"
            min="0"
            :max="job.stage_count - 1"
            class="border-edge data-mono h-7 w-16 rounded-control border bg-bench px-1.5 text-ink"
          />
        </label>
        <div class="mt-2 flex flex-col gap-1">
          <button
            type="button"
            class="border-edge rounded-control border px-2 py-1 text-left text-caption text-ink-2 hover:text-ink"
            @click="doRetake(job, 'cascade')"
          >
            Cascade — re-renders this stage and everything after it.
          </button>
          <button
            type="button"
            class="border-edge rounded-control border px-2 py-1 text-left text-caption text-ink-2 hover:text-ink"
            @click="doRetake(job, 'splice')"
          >
            Splice — replaces this stage in place.
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
