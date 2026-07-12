<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
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
import { useHostsStore, type HostView } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import { hostIdFromUrl } from "../../lib/hosts";
import { ipc, type DiscoveredHost } from "../../lib/ipc";

const router = useRouter();
const generation = useGenerationStore();
const composer = useComposerStore();
const contextMenu = useContextMenuStore();
const hosts = useHostsStore();
const toasts = useToastStore();

// Quiet background mDNS scan so nearby `mold serve` instances surface in the
// rail without a trip to Settings.
const discovered = ref<DiscoveredHost[]>([]);
let scanTimer: ReturnType<typeof setInterval> | null = null;

async function scanNetwork() {
  try {
    discovered.value = await ipc.discoverServers();
  } catch {
    // Discovery is best-effort; the section simply stays as-is.
  }
}

onMounted(() => {
  void scanNetwork();
  scanTimer = setInterval(() => void scanNetwork(), 60_000);
});
onUnmounted(() => {
  if (scanTimer) clearInterval(scanTimer);
});

/** Detected on the network but not connected (and not this machine). */
const detectedHosts = computed(() => {
  const connected = new Set(hosts.all.map((h) => h.id));
  return discovered.value.filter((d) => !d.isThisMachine && !connected.has(hostIdFromUrl(d.url)));
});

function hostDot(host: HostView): string {
  switch (host.status) {
    case "ready":
      return "bg-safelight";
    case "connecting":
      return "bg-halide animate-pulse";
    default:
      return "bg-stop";
  }
}

/** Connect a detected host in place when its key is already stored. */
async function connectDetected(host: DiscoveredHost) {
  const id = hostIdFromUrl(host.url);
  const key = await ipc.secretGet(`remote-api-key.${id}`);
  if (host.authRequired && !key) {
    toasts.push(`${host.name} needs an API key — add it in Settings → Engine.`);
    void router.push("/settings");
    return;
  }
  try {
    await hosts.connect(host.url, key, host.name);
    toasts.push(`Connected to ${host.name}`);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

function hostMenu(host: HostView): MenuEntry[] {
  if (host.primary) {
    return [{ label: "Manage in Settings", action: () => void router.push("/settings") }];
  }
  const entries: MenuEntry[] = [];
  if (host.status === "error") {
    entries.push({ label: "Reconnect", action: () => void hosts.reconnect(host.id) });
  }
  entries.push({
    label: "Disconnect",
    danger: true,
    action: () => void hosts.disconnect(host.id),
  });
  return entries;
}

const destinations = [
  { route: "/generate", label: "Generate", key: "⌘1" },
  { route: "/gallery", label: "Gallery", key: "⌘2" },
  { route: "/chains", label: "Chains", key: "⌘3" },
  { route: "/models", label: "Models", key: "⌘4" },
  { route: "/history", label: "History", key: "⌘5" },
  { route: "/runpod", label: "RunPod", key: "" },
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
      <span class="edge-code">HOSTS</span>
      <div class="border-edge h-px flex-1 border-t" />
    </div>
    <div data-test="hosts-section">
      <div
        v-for="host in hosts.all"
        :key="host.id"
        data-test="host-row"
        class="mx-2 flex h-7 items-center gap-2 rounded-control px-2.5 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)]"
        :title="host.baseUrl ?? undefined"
        @contextmenu.prevent="contextMenu.open($event, hostMenu(host))"
      >
        <span class="h-1.5 w-1.5 shrink-0 rounded-full" :class="hostDot(host)" />
        <span class="min-w-0 truncate text-caption text-ink-2">{{ host.label }}</span>
        <span v-if="host.queueDepth !== null" class="edge-code ml-auto shrink-0">
          {{ host.queueDepth }}
        </span>
      </div>
      <div
        v-for="d in detectedHosts"
        :key="d.url"
        data-test="detected-host-row"
        class="mx-2 flex h-7 items-center gap-2 rounded-control px-2.5"
        :title="d.url"
      >
        <span class="border-edge h-1.5 w-1.5 shrink-0 rounded-full border" />
        <span class="min-w-0 truncate text-caption text-ink-3">{{ d.name }}</span>
        <button
          type="button"
          data-test="detected-host-connect"
          class="ml-auto shrink-0 rounded-control px-1 text-caption text-ink-3 hover:text-ink"
          :aria-label="`Connect to ${d.name}`"
          @click="connectDetected(d)"
        >
          +
        </button>
      </div>
      <p
        v-if="hosts.all.length === 0 && detectedHosts.length === 0"
        class="mx-4 text-caption text-ink-3"
      >
        No hosts
      </p>
    </div>

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
          <div class="truncate text-caption text-ink-2" :title="job.prompt">
            {{ job.model }}<template v-if="job.hostLabel"> · {{ job.hostLabel }}</template>
          </div>
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
