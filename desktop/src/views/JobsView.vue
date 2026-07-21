<script setup lang="ts">
import { computed, onMounted, onUnmounted } from "vue";
import { useRouter } from "vue-router";
import DevelopCanvas from "../lib/develop/DevelopCanvas.vue";
import HostQueuePanel from "../components/machines/HostQueuePanel.vue";
import {
  useGenerationStore,
  jobPhase,
  jobProgress,
  jobStatusCode,
  type Job,
} from "../stores/generation";
import { useHostsStore } from "../stores/hosts";
import { useJobsStore } from "../stores/jobs";
import { useComposerStore } from "../stores/composer";

const router = useRouter();
const generation = useGenerationStore();
const hosts = useHostsStore();
const jobs = useJobsStore();
const composer = useComposerStore();

onMounted(() => jobs.startPolling());
onUnmounted(() => jobs.stopPolling());

/** Finished jobs from this session, freshest first. */
const finished = computed(() =>
  generation.jobs
    .filter((j) => j.status === "complete" || j.status === "error")
    .slice()
    .reverse(),
);

function reuse(job: Job) {
  composer.set({
    prompt: job.prompt,
    model: job.model,
    seed: null,
    width: job.width,
    height: job.height,
    steps: job.total,
    guidance: job.guidance,
  });
  void router.push("/generate");
}
</script>

<template>
  <div class="h-full overflow-y-auto p-6">
    <div class="mx-auto max-w-3xl">
      <div class="flex items-center gap-3">
        <h1 class="font-display text-display-md font-bold text-ink" style="font-stretch: 90%">
          Jobs
        </h1>
        <div class="flex-1" />
        <button
          v-if="finished.length"
          type="button"
          class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
          @click="generation.prune(0)"
        >
          Clear finished
        </button>
      </div>

      <!-- One queue section per connected host — the full management panel is
           the shared HostQueuePanel, reused verbatim by the Machines host
           detail so both surfaces behave identically. -->
      <section v-for="host in hosts.all" :key="host.id" data-test="host-queue" class="mt-6">
        <div class="flex items-center gap-2">
          <span class="edge-code">{{ host.label }}</span>
          <div class="border-edge h-px flex-1 border-t" />
          <span v-if="host.queueDepth !== null" class="edge-code">
            {{ host.queueDepth
            }}<template v-if="host.queueCapacity">/{{ host.queueCapacity }}</template>
          </span>
        </div>
        <HostQueuePanel :host="host" class="mt-2" />
      </section>

      <!-- This session's finished prints -->
      <section v-if="finished.length" class="mt-8">
        <div class="flex items-center gap-2">
          <span class="edge-code">Finished this session</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <ul class="mt-2 space-y-1.5">
          <li
            v-for="job in finished"
            :key="job.clientId"
            data-test="finished-row"
            class="border-edge flex items-center gap-3 rounded-control border bg-bench px-3 py-2"
          >
            <div
              class="h-12 w-12 shrink-0 overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_14%,transparent)] bg-print-surface"
            >
              <img
                v-if="job.resultUrl && !job.result?.video_frames"
                :src="job.resultUrl"
                alt=""
                class="h-full w-full object-cover"
              />
              <DevelopCanvas
                v-else
                :seed="job.visualSeed"
                :progress="jobProgress(job)"
                :phase="jobPhase(job)"
              />
            </div>
            <div class="min-w-0 flex-1">
              <div class="truncate text-body text-ink" :title="job.prompt">{{ job.prompt }}</div>
              <div class="mt-0.5 flex items-center gap-2">
                <span class="edge-code" :class="job.status === 'error' ? 'text-stop' : ''">
                  {{ jobStatusCode(job) }}
                </span>
                <span class="text-caption text-ink-3">
                  {{ job.model }}<template v-if="job.hostLabel"> · {{ job.hostLabel }}</template>
                </span>
                <span class="data-mono text-caption text-ink-3">S {{ job.visualSeed }}</span>
              </div>
            </div>
            <button
              type="button"
              class="border-edge h-7 shrink-0 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
              @click="reuse(job)"
            >
              Reuse
            </button>
          </li>
        </ul>
      </section>
    </div>
  </div>
</template>
