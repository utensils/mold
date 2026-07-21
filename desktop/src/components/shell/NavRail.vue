<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import NavItem from "@ui/components/NavItem.vue";
import Keycap from "@ui/components/Keycap.vue";
import type { IconName } from "@ui/icons";
import logoUrl from "../../assets/logo.png";
import StatusPopover from "./StatusPopover.vue";
import RenameDialog from "./RenameDialog.vue";
import { useGenerationStore, jobStatusCode, railOrder, type Job } from "../../stores/generation";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useComposerStore } from "../../stores/composer";
import { useContextMenuStore, type MenuEntry } from "../../stores/contextMenu";
import { useGalleryStore } from "../../stores/gallery";
import { useHostsStore, type HostView } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import { hostIdFromUrl } from "../../lib/hosts";
import { detectedHosts as computeDetectedHosts } from "../../lib/discovery";
import { ipc, type DiscoveredHost, type SavedHost } from "../../lib/ipc";
import { shortcutLabel } from "../../lib/platform";

const route = useRoute();
const router = useRouter();
const appPrefs = useAppPrefsStore();
const generation = useGenerationStore();
const composer = useComposerStore();
const contextMenu = useContextMenuStore();
const hosts = useHostsStore();
const toasts = useToastStore();

// Fixed widths per the prototype (210 ↔ 62). The old drag-resize handle is
// gone; the persisted navRailWidth pref is left untouched but no longer read.
const collapsed = computed(() => appPrefs.sidebarCollapsed);

// Quiet background mDNS scan so nearby `mold serve` instances surface in the
// rail without a trip to Settings.
const discovered = ref<DiscoveredHost[]>([]);
// Remembered hosts snapshot so the detected-row menu can offer Forget for a
// box that's saved (under any slug) but not currently connected.
const remembered = ref<SavedHost[]>([]);
let scanTimer: ReturnType<typeof setInterval> | null = null;

async function scanNetwork() {
  try {
    discovered.value = await ipc.discoverServers();
    remembered.value = (await ipc.appSettingsGet()).savedHosts;
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

/** The remembered entry for a detected box — matched by advertised slug or,
 *  when the advertisement carries an instance id, by a saved twin under any
 *  slug (remembered by hostname, advertised by IP). */
function rememberedTwinOf(d: DiscoveredHost): SavedHost | null {
  const id = hostIdFromUrl(d.url);
  return (
    remembered.value.find((s) => s.id === id) ??
    (d.instanceId ? (remembered.value.find((s) => s.instanceId === d.instanceId) ?? null) : null)
  );
}

function detectedMenu(d: DiscoveredHost): MenuEntry[] {
  const entries: MenuEntry[] = [
    { label: "Connect", action: () => void connectDetected(d) },
    { separator: true },
    { label: "Open web UI", action: () => void openHostUrl(d.url) },
    {
      label: "Copy URL",
      action: () => void navigator.clipboard.writeText(d.url).then(() => toasts.push("Copied")),
    },
  ];
  const saved = rememberedTwinOf(d);
  if (saved) {
    entries.push({ separator: true });
    entries.push({
      label: "Forget",
      danger: true,
      action: () =>
        void ipc.forgetRemoteHost(saved.id).then(() => {
          remembered.value = remembered.value.filter((s) => s.id !== saved.id);
          toasts.push(`Forgot ${saved.name ?? d.name}`);
        }),
    });
  }
  return entries;
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
    label: "View in library",
    action: () => {
      // Direct set: pushing an unchanged ?host= query is a no-op navigation,
      // so the view's watcher alone can't re-apply the filter.
      useGalleryStore().filter = host.id;
      void router.push({ path: "/library", query: { host: host.id } });
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

interface Destination {
  route: string;
  label: string;
  icon: IconName;
}

// Five destinations (spec §04). Chains fold into Create, History into
// Library's drawer, Jobs/RunPod into Machines — all still reachable by
// deep-link and the command palette. Settings is pinned separately at the
// bottom. ⌘1–⌘4 track this order (see lib/shortcuts).
const destinations: Destination[] = [
  { route: "/create", label: "Create", icon: "create" },
  { route: "/library", label: "Library", icon: "library" },
  { route: "/models", label: "Models", icon: "models" },
  { route: "/machines", label: "Machines", icon: "machines" },
];

function isActive(path: string): boolean {
  return route.path === path;
}

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

function jobRunning(job: Job): boolean {
  return job.status === "denoising" || job.status === "finishing" || job.status === "loading";
}

/** Lowercase mono progress line for the developing strip. */
function developingLabel(job: Job): string {
  if (job.status === "denoising") return `developing ${job.step}/${job.total}`;
  return jobStatusCode(job).toLowerCase();
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
        void router.push("/create");
      },
    },
    {
      label: "Show in library",
      disabled: job.status !== "complete",
      action: () => void router.push("/library"),
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
    class="nav-rail relative flex shrink-0 flex-col border-r border-edge bg-bench pt-3.5 pb-3"
    :class="collapsed ? 'px-1.5' : 'px-2.5'"
    :style="{ width: collapsed ? '62px' : '210px' }"
    aria-label="Primary"
  >
    <!-- header: logo + gradient wordmark + STUDIO kicker -->
    <div class="mb-4 flex items-center gap-2.5 px-2" :class="collapsed ? 'justify-center' : ''">
      <img :src="logoUrl" alt="mold" class="h-6 w-6 shrink-0 object-contain" />
      <span v-if="!collapsed" class="ms-wordmark select-none">mold</span>
      <span v-if="!collapsed" class="rail-studio mt-[3px] select-none">STUDIO</span>
    </div>

    <!-- nav destinations -->
    <div class="flex flex-col gap-[3px]">
      <NavItem
        v-for="d in destinations"
        :key="d.route"
        :icon="d.icon"
        :label="d.label"
        :collapsed="collapsed"
        :active="isActive(d.route)"
        @select="router.push(d.route)"
      />
    </div>

    <!-- now developing -->
    <template v-if="!collapsed">
      <div class="mt-5 mb-1.5 flex items-center gap-2 px-3">
        <span class="rail-kicker">Now developing</span>
        <span v-if="generation.pending.length > 1" class="rail-kicker ml-auto">
          {{ generation.pending.length }}
        </span>
      </div>
      <div
        v-if="railJobs.length > 0"
        class="flex max-h-44 min-h-0 flex-col gap-2 overflow-y-auto px-2"
      >
        <a
          v-for="job in railJobs"
          :key="job.clientId"
          href="#"
          class="flex items-center gap-2.5 rounded-[8px] px-1 py-1 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)]"
          @click.prevent="router.push('/create')"
          @contextmenu.prevent="contextMenu.open($event, jobMenu(job))"
        >
          <span
            class="h-[30px] w-[30px] shrink-0 overflow-hidden rounded-[6px] border border-[color-mix(in_srgb,var(--rebate)_12%,transparent)] bg-print-surface"
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
            <span v-else-if="jobRunning(job)" class="ms-shimmer block h-full w-full" />
            <span v-else class="block h-full w-full bg-print-surface" />
          </span>
          <span class="min-w-0 flex-1">
            <span class="block truncate text-[11.5px] text-ink-2" :title="job.prompt">
              {{ job.model }}<template v-if="job.hostLabel"> · {{ job.hostLabel }}</template>
            </span>
            <span
              class="block font-utility text-[9.5px]"
              :class="job.status === 'error' ? 'text-stop' : 'text-safelight'"
            >
              {{ developingLabel(job) }}
            </span>
          </span>
        </a>
      </div>
      <p v-else class="px-3 text-caption text-ink-3">nothing developing</p>
    </template>

    <!-- machines: ambient host status + quick nav to /machines/:id -->
    <div v-if="!collapsed" class="mt-5 mb-1.5 flex items-center gap-2 px-3">
      <span class="rail-kicker">Machines</span>
      <div class="h-px flex-1 border-t border-edge" />
    </div>
    <div v-else class="mt-4 mb-1 h-px border-t border-edge" />
    <div data-test="hosts-section" class="flex flex-col gap-[2px]">
      <button
        v-for="host in hosts.all"
        :key="host.id"
        type="button"
        data-test="host-row"
        class="flex h-8 cursor-pointer items-center gap-2.5 rounded-[9px] hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)]"
        :class="collapsed ? 'justify-center px-0' : 'px-2.5'"
        :title="collapsed ? host.label : (host.baseUrl ?? undefined)"
        @click="router.push(`/machines/${host.id}`)"
        @contextmenu.prevent="contextMenu.open($event, hostMenu(host))"
      >
        <span class="h-[7px] w-[7px] shrink-0 rounded-full" :class="hostDot(host)" />
        <template v-if="!collapsed">
          <span class="min-w-0 flex-1 truncate text-caption text-ink-2">{{ host.label }}</span>
          <span v-if="host.queueDepth !== null" class="font-utility text-[9.5px] text-ink-3">
            {{ host.queueDepth }}
          </span>
        </template>
      </button>
      <div
        v-for="d in detectedHosts"
        :key="d.url"
        data-test="detected-host-row"
        class="flex h-8 items-center gap-2.5 rounded-[9px]"
        :class="collapsed ? 'justify-center px-0' : 'px-2.5'"
        :title="d.url"
        @contextmenu.prevent="contextMenu.open($event, detectedMenu(d))"
      >
        <span class="h-[7px] w-[7px] shrink-0 rounded-full border border-control-edge" />
        <template v-if="!collapsed">
          <span class="min-w-0 flex-1 truncate text-caption text-ink-3">{{ d.name }}</span>
          <button
            type="button"
            data-test="detected-host-connect"
            class="shrink-0 rounded-[6px] px-1 text-caption text-ink-3 hover:text-ink"
            :aria-label="`Connect to ${d.name}`"
            @click="connectDetected(d)"
          >
            +
          </button>
        </template>
      </div>
      <p
        v-if="!collapsed && hosts.all.length === 0 && detectedHosts.length === 0"
        class="px-3 text-caption text-ink-3"
      >
        No hosts
      </p>
    </div>

    <div class="flex-1" />

    <!-- status + settings -->
    <StatusPopover :collapsed="collapsed" />
    <div class="relative mt-1">
      <NavItem
        icon="settings"
        label="Settings"
        :collapsed="collapsed"
        :active="isActive('/settings')"
        @select="router.push('/settings')"
      />
      <Keycap
        v-if="!collapsed"
        class="pointer-events-none absolute top-1/2 right-3 -translate-y-1/2"
      >
        {{ shortcutLabel(",") }}
      </Keycap>
    </div>

    <RenameDialog
      :open="!!renameTarget"
      title="Rename host"
      :initial="renameTarget?.label ?? ''"
      @save="onRenameSave"
      @cancel="renameTarget = null"
    />
  </nav>
</template>

<style scoped>
.nav-rail {
  transition: width var(--dur-base) var(--ease);
}

.ms-wordmark {
  font-family: var(--f-display);
  font-weight: 800;
  font-size: 17px;
  letter-spacing: -0.01em;
  line-height: 1;
  background: var(--grad);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.rail-studio {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.16em;
  color: var(--ink-3);
}

.rail-kicker {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-3);
  white-space: nowrap;
}
</style>
