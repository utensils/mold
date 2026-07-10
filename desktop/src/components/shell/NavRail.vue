<script setup lang="ts">
import { computed } from "vue";
import { RouterLink, useRouter } from "vue-router";
import DevelopCanvas from "../../lib/develop/DevelopCanvas.vue";
import {
  useGenerationStore,
  jobPhase,
  jobProgress,
  railOrder,
  type Job,
} from "../../stores/generation";
import { useComposerStore } from "../../stores/composer";
import { useContextMenuStore, type MenuEntry } from "../../stores/contextMenu";
import { useToastStore } from "../../stores/toasts";

const router = useRouter();
const generation = useGenerationStore();
const composer = useComposerStore();
const contextMenu = useContextMenuStore();
const toasts = useToastStore();

const destinations = [
  { route: "/generate", label: "Generate", key: "⌘1" },
  { route: "/gallery", label: "Gallery", key: "⌘2" },
  { route: "/chains", label: "Chains", key: "⌘3" },
  { route: "/models", label: "Models", key: "⌘4" },
  { route: "/history", label: "History", key: "⌘5" },
];

/** Queue order: every live job first (submission order), then the freshest
 *  finished prints — the rail is a working queue, not a full history. */
const railJobs = computed<Job[]>(() => {
  const live = railOrder(
    generation.jobs.filter((j) => j.status !== "complete" && j.status !== "error"),
  );
  const done = generation.jobs
    .filter((j) => j.status === "complete" || j.status === "error")
    .slice(-3)
    .reverse();
  return [...live, ...done];
});

function statusCode(job: Job): string {
  switch (job.status) {
    case "denoising":
      return `${job.step}/${job.total}`;
    case "finishing":
      return "FIXING";
    case "loading":
      return "LOADING";
    case "queued":
      return job.queuePosition && job.queuePosition > 0 ? `QUEUED #${job.queuePosition}` : "QUEUED";
    case "complete":
      return "FIXED";
    case "error":
      return job.error === "Cancelled" ? "CANCELLED" : "STOPPED";
  }
}

function jobMenu(job: Job): MenuEntry[] {
  const live = job.status !== "complete" && job.status !== "error";
  return [
    {
      label: "Cancel",
      danger: true,
      disabled: !live,
      action: () => void generation.cancel(job.clientId).then(() => toasts.push("Cancelled")),
    },
    { separator: true },
    {
      label: "Use prompt",
      action: () => {
        composer.set({
          prompt: job.prompt,
          model: job.model,
          seed: null,
          width: job.width,
          height: job.height,
          steps: job.total,
          guidance: 1.0,
        });
        void router.push("/generate");
      },
    },
    {
      label: "Show in Gallery",
      disabled: job.status !== "complete",
      action: () => void router.push("/gallery"),
    },
    { separator: true },
    {
      label: "Clear finished",
      disabled: !generation.jobs.some((j) => j.status === "complete" || j.status === "error"),
      action: () => generation.prune(0),
    },
  ];
}
</script>

<template>
  <nav class="border-edge flex w-[208px] flex-col border-r bg-bench pt-2 pb-2" aria-label="Primary">
    <RouterLink
      v-for="d in destinations"
      :key="d.route"
      :to="d.route"
      class="group mx-2 flex h-7 items-center justify-between rounded-control px-2.5 text-body text-ink-2 transition-colors duration-100 hover:text-ink"
      active-class="!text-ink bg-[color-mix(in_srgb,var(--safelight)_14%,transparent)]"
    >
      <span class="font-medium">{{ d.label }}</span>
      <kbd
        class="kbd-hint text-ink-3 opacity-0 transition-opacity duration-100 group-hover:opacity-100"
      >
        {{ d.key }}
      </kbd>
    </RouterLink>

    <div class="mx-4 mt-4 mb-1 flex items-center gap-2">
      <span class="edge-code">JOBS</span>
      <div class="border-edge h-px flex-1 border-t" />
      <span v-if="generation.pending.length > 1" class="edge-code">
        {{ generation.pending.length }}
      </span>
    </div>
    <div v-if="railJobs.length > 0" class="min-h-0 overflow-y-auto">
      <RouterLink
        v-for="job in railJobs"
        :key="job.clientId"
        to="/generate"
        class="mx-2 flex items-center gap-2 rounded-control px-2 py-1.5 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)]"
        @contextmenu="contextMenu.open($event, jobMenu(job))"
      >
        <div
          class="h-8 w-8 shrink-0 overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_14%,transparent)] bg-print-surface"
        >
          <img
            v-if="job.resultUrl && !job.result?.video_frames"
            :src="job.resultUrl"
            alt=""
            class="h-full w-full object-cover"
          />
          <img
            v-else-if="job.previewUrl"
            :src="job.previewUrl"
            alt=""
            class="h-full w-full object-cover"
            style="filter: blur(1px)"
          />
          <DevelopCanvas
            v-else
            :seed="job.visualSeed"
            :progress="jobProgress(job)"
            :phase="jobPhase(job)"
          />
        </div>
        <div class="min-w-0">
          <div class="truncate text-caption text-ink-2" :title="job.prompt">{{ job.model }}</div>
          <div class="edge-code" :class="job.status === 'error' ? 'text-stop' : ''">
            {{ statusCode(job) }}
          </div>
        </div>
      </RouterLink>
    </div>
    <p v-else class="mx-4 text-caption text-ink-3">Nothing developing</p>

    <div class="flex-1" />

    <RouterLink
      to="/settings"
      class="mx-2 flex h-7 items-center justify-between rounded-control px-2.5 text-body text-ink-2 transition-colors duration-100 hover:text-ink"
      active-class="!text-ink bg-[color-mix(in_srgb,var(--safelight)_14%,transparent)]"
    >
      <span class="font-medium">Settings</span>
      <kbd class="kbd-hint text-ink-3">⌘,</kbd>
    </RouterLink>
  </nav>
</template>
