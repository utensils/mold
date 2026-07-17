<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { RouterLink, useRouter } from "vue-router";
import DevelopCanvas from "../../lib/develop/DevelopCanvas.vue";
import PanelResizeHandle from "./PanelResizeHandle.vue";
import {
  useGenerationStore,
  jobPhase,
  jobProgress,
  railOrder,
  type Job,
} from "../../stores/generation";
import RenameDialog from "./RenameDialog.vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useComposerStore } from "../../stores/composer";
import { useContextMenuStore, type MenuEntry } from "../../stores/contextMenu";
import { useGalleryStore } from "../../stores/gallery";
import { useHostsStore, type HostView } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import { hostIdFromUrl } from "../../lib/hosts";
import { detectedHosts as computeDetectedHosts } from "../../lib/discovery";
import { dragWidth } from "../../lib/panelResize";
import { ipc, type DiscoveredHost } from "../../lib/ipc";
import { shortcutLabel } from "../../lib/platform";

const router = useRouter();
const appPrefs = useAppPrefsStore();
const generation = useGenerationStore();
const composer = useComposerStore();
const contextMenu = useContextMenuStore();
const hosts = useHostsStore();
const toasts = useToastStore();

// Live width while dragging the right-edge handle; null follows the
// persisted preference (appPrefs.navRailWidth). Persistence happens only on
// commit, never per pointermove.
const draftRailWidth = ref<number | null>(null);
const railWidth = computed(() => draftRailWidth.value ?? appPrefs.navRailWidth);

function onRailResize(dx: number) {
  draftRailWidth.value = dragWidth("navRail", appPrefs.navRailWidth, dx, "right");
}

async function onRailCommit() {
  const width = draftRailWidth.value;
  if (width === null) return;
  if (width !== appPrefs.navRailWidth) await appPrefs.update({ navRailWidth: width });
  draftRailWidth.value = null;
}

function onRailReset() {
  draftRailWidth.value = null;
  void appPrefs.update({ navRailWidth: null });
}

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

/** Detected on the network but not connected (and not this machine). Deduped
 *  by slug AND instance id, so a box already connected by hostname isn't
 *  re-offered when mDNS advertises it by IP. */
const detectedHosts = computed(() =>
  computeDetectedHosts(
    discovered.value,
    new Set(hosts.all.map((h) => h.id)),
    new Set(hosts.all.map((h) => h.instanceId).filter((id): id is string => !!id)),
  ),
);

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

/** Connect a detected host in place when its key is already stored — under
 *  its advertised URL slug, or under the slug of a remembered host with the
 *  same instance id (a box remembered by hostname is often advertised by IP
 *  under a different slug). */
async function connectDetected(host: DiscoveredHost) {
  const id = hostIdFromUrl(host.url);
  let key = await ipc.secretGet(`remote-api-key.${id}`);
  if (!key && host.instanceId) {
    try {
      const saved = (await ipc.appSettingsGet()).savedHosts.find(
        (s) => s.instanceId === host.instanceId,
      );
      if (saved && saved.id !== id) key = await ipc.secretGet(`remote-api-key.${saved.id}`);
    } catch {
      // Settings unreadable — fall through to the key prompt.
    }
  }
  if (host.authRequired && !key) {
    toasts.push(`${host.name} needs an API key — add it in Settings → Hosts.`);
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

const renameTarget = ref<HostView | null>(null);

/** Open the host's web UI in the default browser. */
async function openHostUrl(url: string) {
  try {
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    await openUrl(url);
  } catch {
    window.open(url, "_blank", "noopener");
  }
}

/** Drop the host AND its saved entry + stored API key (recoverable only by
 *  re-adding it). Disconnect alone keeps both for later reconnect. */
async function forgetHost(host: HostView) {
  await hosts.disconnect(host.id);
  await ipc.forgetRemoteHost(host.id);
  toasts.push(`Forgot ${host.label}`);
}

function onRenameSave(name: string) {
  const host = renameTarget.value;
  renameTarget.value = null;
  if (host) void hosts.rename(host.id, name);
}

function hostMenu(host: HostView): MenuEntry[] {
  const entries: MenuEntry[] = [];
  const isTarget = (appPrefs.settings?.generateTargetHost ?? null) === host.id;
  if (isTarget) {
    entries.push({
      label: "Route automatically",
      action: () => void appPrefs.update({ generateTargetHost: null }),
    });
  } else {
    entries.push({
      label: "Set as generation target",
      disabled: host.status !== "ready",
      action: () => void appPrefs.update({ generateTargetHost: host.id }),
    });
  }
  entries.push({
    label: "View gallery",
    action: () => {
      // Direct set: pushing an unchanged ?host= query is a no-op navigation,
      // so the view's watcher alone can't re-apply the filter.
      useGalleryStore().filter = host.id;
      void router.push({ path: "/gallery", query: { host: host.id } });
    },
  });
  entries.push({ separator: true });
  entries.push({
    label: "Open web UI",
    disabled: !host.baseUrl,
    action: () => void openHostUrl(host.baseUrl ?? ""),
  });
  entries.push({
    label: "Copy URL",
    disabled: !host.baseUrl,
    action: () =>
      void navigator.clipboard.writeText(host.baseUrl ?? "").then(() => toasts.push("Copied")),
  });
  if (host.kind === "remote") {
    entries.push({
      label: "Rename…",
      action: () => {
        renameTarget.value = host;
      },
    });
  }
  entries.push({ separator: true });
  if (host.primary) {
    entries.push({ label: "Manage in Settings", action: () => void router.push("/settings") });
    return entries;
  }
  if (host.status === "error") {
    entries.push({ label: "Reconnect", action: () => void hosts.reconnect(host.id) });
  }
  entries.push({
    label: "Disconnect",
    danger: true,
    action: () => void hosts.disconnect(host.id),
  });
  entries.push({
    label: "Forget",
    danger: true,
    action: () => void forgetHost(host),
  });
  return entries;
}

const destinations = [
  { route: "/generate", label: "Generate", key: shortcutLabel("1") },
  { route: "/gallery", label: "Gallery", key: shortcutLabel("2") },
  { route: "/chains", label: "Chains", key: shortcutLabel("3") },
  { route: "/models", label: "Catalog", key: shortcutLabel("4") },
  { route: "/history", label: "History", key: shortcutLabel("5") },
  { route: "/jobs", label: "Jobs", key: shortcutLabel("6") },
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
          guidance: job.guidance,
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
  <nav
    class="border-edge relative flex flex-col border-r bg-bench pt-2 pb-2"
    :style="{ width: `${railWidth}px` }"
    aria-label="Primary"
  >
    <PanelResizeHandle
      class="absolute inset-y-0 -right-0.5 z-10"
      label="Resize sidebar"
      @resize="onRailResize"
      @commit="onRailCommit"
      @reset="onRailReset"
    />
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
      <button
        v-for="host in hosts.all"
        :key="host.id"
        type="button"
        data-test="host-row"
        class="mx-2 flex h-7 cursor-pointer items-center gap-2 rounded-control px-2.5 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)]"
        :title="host.baseUrl ?? undefined"
        @click="router.push(`/hosts/${host.id}`)"
        @contextmenu.prevent="contextMenu.open($event, hostMenu(host))"
      >
        <span class="h-1.5 w-1.5 shrink-0 rounded-full" :class="hostDot(host)" />
        <span class="min-w-0 truncate text-caption text-ink-2">{{ host.label }}</span>
        <span v-if="host.queueDepth !== null" class="edge-code ml-auto shrink-0">
          {{ host.queueDepth }}
        </span>
      </button>
      <div
        v-for="d in detectedHosts"
        :key="d.url"
        data-test="detected-host-row"
        class="mx-2 flex h-7 items-center gap-2 rounded-control px-2.5"
        :title="d.url"
      >
        <span class="h-1.5 w-1.5 shrink-0 rounded-full border border-control-edge" />
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
      <kbd class="kbd-hint text-ink-3">{{ shortcutLabel(",") }}</kbd>
    </RouterLink>

    <RenameDialog
      :open="!!renameTarget"
      title="Rename host"
      :initial="renameTarget?.label ?? ''"
      @save="onRenameSave"
      @cancel="renameTarget = null"
    />
  </nav>
</template>
