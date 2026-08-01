<script setup lang="ts">
/*
 * Machines overview (spec §06) — every machine as a card: This device first,
 * then connected remotes, then remembered (offline) hosts, then anything
 * discovered on the LAN, plus a RunPod offer card. A right-hand column mirrors
 * the live queue. Host CRUD reuses the hosts store verbatim (the same actions
 * the old Settings → Hosts panel drove), so instance-UUID dedupe, per-host key
 * storage, and boot reconnect are unchanged — this view only reframes them.
 */
import { computed, onMounted, onUnmounted, ref } from "vue";
import { useRouter } from "vue-router";
import CardSurface from "@ui/components/CardSurface.vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import Icon from "@ui/components/Icon.vue";
import ConnectMachineModal from "../components/machines/ConnectMachineModal.vue";
import QueueColumn from "../components/machines/QueueColumn.vue";
import PodCostMeter from "../components/machines/PodCostMeter.vue";
import ConfirmDialog from "../components/shell/ConfirmDialog.vue";
import { ipc, type DiscoveredHost, type SavedHost } from "../lib/ipc";
import { gpuFleetLabel, gpuSnapshotsFromWorkers } from "../lib/api/gpuStatus";
import { addressLabel, prepareHosts, versionLabel } from "../lib/discovery";
import { formatGB } from "../lib/format";
import { hostIdFromUrl, inferBackendFromGpuName } from "../lib/hosts";
import { podGpuName, type RunPodPod } from "../lib/runpod";
import { useHostsStore, type HostView } from "../stores/hosts";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useJobsStore } from "../stores/jobs";
import { useRunPodStore } from "../stores/runpod";
import { useToastStore } from "../stores/toasts";

const router = useRouter();
const hosts = useHostsStore();
const appPrefs = useAppPrefsStore();
const contextMenu = useContextMenuStore();
const jobs = useJobsStore();
const runpod = useRunPodStore();
const toasts = useToastStore();

onMounted(() => {
  jobs.startPolling();
  void refreshSaved();
  void scan();
  // Surface any billing pods on the overview so a paid host's running cost is
  // visible without opening the RunPod view (§08 G9). No-ops when unconfigured.
  void runpod.load();
});
onUnmounted(() => jobs.stopPolling());

async function stopPod(pod: RunPodPod) {
  try {
    await runpod.act("stop", pod.id);
    toasts.push(`Stopped ${pod.name ?? pod.id}`);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

const connectOpen = ref(false);

function openDetail(host: HostView) {
  void router.push(`/machines/${host.id}`);
}

async function copyAddress(address: string) {
  try {
    await navigator.clipboard.writeText(address);
    toasts.push("Address copied");
  } catch (err) {
    toasts.push(err instanceof Error ? err.message : String(err), "error");
  }
}

async function openHostUrl(url: string) {
  try {
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    await openUrl(url);
  } catch {
    window.open(url, "_blank", "noopener");
  }
}

async function disconnectHost(host: HostView) {
  await hosts.disconnect(host.id);
  await refreshSaved();
  toasts.push(`Disconnected from ${host.label}`);
}

const forgetCandidate = ref<{ id: string; label: string; connected: boolean } | null>(null);

async function forgetHost() {
  const candidate = forgetCandidate.value;
  forgetCandidate.value = null;
  if (!candidate) return;
  try {
    if (candidate.connected) await hosts.disconnect(candidate.id);
    await ipc.forgetRemoteHost(candidate.id);
    await refreshSaved();
    toasts.push(`Forgot ${candidate.label}`);
  } catch (err) {
    toasts.push(err instanceof Error ? err.message : String(err), "error");
  }
}

function connectedHostMenu(host: HostView): MenuEntry[] {
  const isTarget = (appPrefs.settings?.generateTargetHost ?? null) === host.id;
  return [
    { label: "Open details", action: () => openDetail(host) },
    {
      label: isTarget ? "Generation target" : "Set as generation target",
      disabled: isTarget || host.status !== "ready",
      action: () => void appPrefs.update({ generateTargetHost: host.id }),
    },
    {
      label: "Copy address",
      disabled: !host.baseUrl,
      action: () => void copyAddress(host.baseUrl ?? ""),
    },
    {
      label: "Open web UI",
      disabled: !host.baseUrl,
      action: () => void openHostUrl(host.baseUrl ?? ""),
    },
    ...(host.kind === "remote"
      ? [
          { separator: true } as const,
          ...(host.status === "error"
            ? [{ label: "Retry connection", action: () => void hosts.reconnect(host.id) }]
            : []),
          { label: "Disconnect", action: () => void disconnectHost(host) },
          {
            label: "Forget…",
            danger: true,
            action: () => {
              forgetCandidate.value = { id: host.id, label: host.label, connected: true };
            },
          },
        ]
      : []),
  ];
}

function rememberedHostMenu(host: SavedHost): MenuEntry[] {
  return [
    { label: "Connect", disabled: adding.value, action: () => void connectSaved(host) },
    { label: "Copy address", action: () => void copyAddress(host.url) },
    { separator: true },
    {
      label: "Forget…",
      danger: true,
      action: () => {
        forgetCandidate.value = { id: host.id, label: savedHostLabel(host), connected: false };
      },
    },
  ];
}

function discoveredHostMenu(host: DiscoveredHost): MenuEntry[] {
  return [
    {
      label: "Connect",
      disabled: adding.value || isThisMachine(host),
      action: () => void addDiscovered(host),
    },
    { label: "Copy address", action: () => void copyAddress(host.url) },
  ];
}

// ── Card telemetry (from the app-wide status poll) ────────────────────────
function hostGpus(id: string) {
  const telemetry = hosts.telemetry[id];
  return gpuSnapshotsFromWorkers(telemetry?.gpuInfo, telemetry?.gpuWorkers);
}

/** Right-aligned hardware line: "RTX 4090 · Metal" / "RTX 4090 · host:port". */
function hardwareLine(host: HostView): string {
  const gpus = hostGpus(host.id);
  const parts: string[] = [];
  const fleet = gpuFleetLabel(gpus);
  if (fleet) parts.push(fleet);
  if (host.kind === "local") {
    if (gpus[0]) {
      parts.push((gpus[0].backend ?? inferBackendFromGpuName(gpus[0].name)).toUpperCase());
    }
  } else if (host.baseUrl) {
    parts.push(host.baseUrl.replace(/^https?:\/\//, ""));
  }
  return parts.join(" · ") || (host.baseUrl?.replace(/^https?:\/\//, "") ?? "");
}

function memoryLabel(host: HostView): string | null {
  const gpus = hostGpus(host.id);
  if (!gpus.length) return null;
  const used = gpus.reduce((sum, gpu) => sum + gpu.vram_used, 0);
  const total = gpus.reduce((sum, gpu) => sum + gpu.vram_total, 0);
  return `Memory ${formatGB(used)} / ${formatGB(total)}`;
}

function memoryPct(host: HostView): number {
  const gpus = hostGpus(host.id);
  const used = gpus.reduce((sum, gpu) => sum + gpu.vram_used, 0);
  const total = gpus.reduce((sum, gpu) => sum + gpu.vram_total, 0);
  return total > 0 ? Math.round((used / total) * 100) : 0;
}

function statusDot(status: HostView["status"]): string {
  switch (status) {
    case "ready":
      return "bg-safelight";
    case "connecting":
      return "bg-halide animate-pulse";
    default:
      return "bg-stop";
  }
}

const connectedRemotes = computed(() => hosts.all.filter((h) => !h.primary));
const connectedRemoteIds = computed(() => new Set(connectedRemotes.value.map((h) => h.id)));
const connectedRemoteInstanceIds = computed(
  () => new Set(connectedRemotes.value.map((h) => h.instanceId).filter((id): id is string => !!id)),
);

// ── Remembered (saved but offline) ────────────────────────────────────────
const savedHosts = ref<SavedHost[]>([]);
const adding = ref(false);
const actionError = ref<string | null>(null);

async function refreshSaved() {
  savedHosts.value = (await ipc.appSettingsGet()).savedHosts ?? [];
}

const rememberedHosts = computed(() =>
  savedHosts.value.filter((h) => !connectedRemoteIds.value.has(h.id)),
);

function savedHostLabel(host: SavedHost): string {
  return host.name ?? host.url.replace(/^https?:\/\//, "");
}

async function connectSaved(host: SavedHost) {
  adding.value = true;
  actionError.value = null;
  try {
    const key = await ipc.secretGet(`remote-api-key.${host.id}`);
    await hosts.connect(host.url, key, host.name);
    await refreshSaved();
  } catch (err) {
    actionError.value = String(err);
  } finally {
    adding.value = false;
  }
}

// ── On your network (mDNS discovery) ──────────────────────────────────────
const discovered = ref<DiscoveredHost[]>([]);
const scanning = ref(false);

const undiscovered = computed(() =>
  discovered.value.filter(
    (d) =>
      !connectedRemoteIds.value.has(hostIdFromUrl(d.url)) &&
      !(d.instanceId && connectedRemoteInstanceIds.value.has(d.instanceId)),
  ),
);

async function scan() {
  scanning.value = true;
  try {
    discovered.value = prepareHosts(await ipc.discoverServers());
  } catch {
    discovered.value = [];
  } finally {
    scanning.value = false;
  }
}

function isThisMachine(host: DiscoveredHost): boolean {
  return (
    host.isThisMachine || (!!host.instanceId && host.instanceId === hosts.primaryHost?.instanceId)
  );
}

async function addDiscovered(host: DiscoveredHost) {
  adding.value = true;
  actionError.value = null;
  try {
    // Key lookup: the advertised URL slug first, then a remembered twin with
    // the same instance id — a box remembered by hostname is often advertised
    // by IP under a different slug, and its stored key must still apply.
    let key = await ipc.secretGet(`remote-api-key.${hostIdFromUrl(host.url)}`);
    if (!key && host.instanceId) {
      try {
        const saved = (await ipc.appSettingsGet()).savedHosts.find(
          (s) => s.instanceId === host.instanceId,
        );
        if (saved && saved.id !== hostIdFromUrl(host.url)) {
          key = await ipc.secretGet(`remote-api-key.${saved.id}`);
        }
      } catch {
        // Settings unreadable — connect proceeds without a key.
      }
    }
    const view = await hosts.connect(host.url, key, host.name);
    toasts.push(`Connected to ${view.label}`);
    await refreshSaved();
  } catch (err) {
    actionError.value = String(err);
  } finally {
    adding.value = false;
  }
}

async function onConnected() {
  await refreshSaved();
}
</script>

<template>
  <div class="flex h-full min-h-0 flex-col">
    <!-- Workspace header -->
    <header class="border-edge flex h-13 shrink-0 items-center gap-3 border-b px-6">
      <h1 class="font-display text-display-sm font-bold text-ink" style="font-stretch: 90%">
        Machines
      </h1>
      <div class="flex-1" />
      <button
        type="button"
        data-test="add-machine"
        class="border-ce flex items-center gap-1.5 rounded-chrome border px-3.5 py-2 text-body font-semibold text-ink-2 transition-colors hover:text-ink active:translate-y-px"
        @click="connectOpen = true"
      >
        <Icon name="plus" :size="14" :stroke-width="2" />
        Add machine
      </button>
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto px-6 py-5">
      <div class="flex flex-col gap-5 xl:flex-row xl:items-start">
        <!-- Machine cards -->
        <div class="flex min-w-0 flex-1 flex-col gap-3">
          <div class="edge-code uppercase">Connected</div>

          <!-- This device + connected remotes -->
          <button
            v-for="host in hosts.all"
            :key="host.id"
            type="button"
            :data-test="host.primary ? 'this-device-card' : 'host-card'"
            class="border-edge rounded-chrome border bg-bench p-4 text-left shadow-[inset_0_1px_0_var(--card-hi)] transition-colors hover:border-ink-3"
            @click="openDetail(host)"
            @contextmenu="contextMenu.open($event, connectedHostMenu(host))"
          >
            <div class="flex items-center gap-2.5">
              <span class="h-2 w-2 shrink-0 rounded-full" :class="statusDot(host.status)" />
              <span class="text-body-lg font-semibold text-ink">{{ host.label }}</span>
              <span class="data-mono ml-auto truncate text-caption text-ink-3">
                {{ hardwareLine(host) }}
              </span>
              <Icon name="chevron-right" :size="16" :stroke-width="2" class="shrink-0 text-ink-3" />
            </div>
            <div
              v-if="memoryLabel(host)"
              class="data-mono mt-3.5 mb-1.5 flex items-center justify-between text-caption text-ink-3"
            >
              <span>{{ memoryLabel(host) }}</span>
              <span>queue {{ host.queueDepth ?? 0 }}</span>
            </div>
            <div v-else class="data-mono mt-3.5 mb-1.5 flex justify-end text-caption text-ink-3">
              <span>queue {{ host.queueDepth ?? 0 }}</span>
            </div>
            <ProgressBar
              v-if="memoryLabel(host)"
              :value="memoryPct(host)"
              :tone="host.kind === 'local' ? 'accent' : 'info'"
              :height="6"
            />
          </button>

          <!-- RunPod offer -->
          <CardSurface dashed>
            <div class="flex items-center gap-2.5">
              <span class="h-2 w-2 shrink-0 rounded-full bg-ink-3" />
              <span class="text-body-lg font-semibold text-ink-2">RunPod cloud</span>
              <span class="data-mono ml-auto text-caption text-ink-3">pay per minute</span>
            </div>
            <div class="mt-3.5 flex items-center justify-between">
              <span class="text-caption text-ink-3">Off · pay only while running</span>
              <button
                type="button"
                data-test="start-pod"
                class="rounded-chrome bg-safelight px-3.5 py-1.5 text-caption font-bold text-on-accent hover:brightness-105 active:translate-y-px"
                @click="router.push('/machines/runpod')"
              >
                Start pod
              </button>
            </div>
          </CardSurface>

          <!-- Billing pods: live running-cost meter + Stop (§08 G9) -->
          <div
            v-for="pod in runpod.runningPods"
            :key="pod.id"
            data-test="runpod-running"
            class="border-edge flex items-center gap-3 rounded-chrome border bg-bench px-4 py-3"
          >
            <span class="h-2 w-2 shrink-0 rounded-full bg-halide" />
            <div class="min-w-0 flex-1">
              <div class="truncate text-body font-semibold text-ink">{{ pod.name ?? pod.id }}</div>
              <div class="data-mono truncate text-caption text-ink-3">
                {{ podGpuName(pod) }} · RunPod
              </div>
            </div>
            <PodCostMeter
              :cost-per-hr="pod.costPerHr"
              :uptime-seconds="pod.uptimeSeconds"
              :stoppable="!pod.networkVolume"
              :busy="runpod.mutating === `stop:${pod.id}`"
              @stop="stopPod(pod)"
            />
          </div>

          <!-- Remembered (offline) -->
          <template v-if="rememberedHosts.length">
            <div class="edge-code mt-2 uppercase">Remembered</div>
            <div
              v-for="saved in rememberedHosts"
              :key="saved.id"
              data-test="remembered-host"
              class="border-edge flex items-center gap-3 rounded-chrome border bg-bench px-4 py-3 opacity-70"
              @contextmenu="contextMenu.open($event, rememberedHostMenu(saved))"
            >
              <span class="h-2 w-2 shrink-0 rounded-full bg-ink-3" />
              <div class="min-w-0 flex-1">
                <div class="truncate text-body text-ink">{{ savedHostLabel(saved) }}</div>
                <div class="data-mono truncate text-caption text-ink-3">{{ saved.url }}</div>
              </div>
              <button
                type="button"
                data-test="remembered-connect"
                class="border-edge h-7 shrink-0 rounded-control border px-2.5 text-caption text-ink-2 hover:text-ink disabled:opacity-50"
                :disabled="adding"
                @click="connectSaved(saved)"
              >
                Connect
              </button>
            </div>
          </template>

          <!-- On your network -->
          <div class="mt-2 flex items-center gap-2">
            <span class="edge-code uppercase">On your network</span>
            <div class="flex-1" />
            <button
              type="button"
              class="border-edge h-7 rounded-control border px-2.5 text-caption text-ink-2 hover:text-ink disabled:opacity-50"
              :disabled="scanning"
              @click="scan"
            >
              {{ scanning ? "Scanning…" : "Scan again" }}
            </button>
          </div>
          <div
            v-for="host in undiscovered"
            :key="host.url"
            data-test="discovered-host"
            class="border-edge flex items-center gap-3 rounded-chrome border bg-bench px-4 py-3"
            @contextmenu="contextMenu.open($event, discoveredHostMenu(host))"
          >
            <span class="h-2 w-2 shrink-0 rounded-full bg-halide" />
            <div class="min-w-0 flex-1">
              <div class="flex items-center gap-2">
                <span class="truncate text-body text-ink">{{ host.name }}</span>
                <span v-if="isThisMachine(host)" class="edge-code">THIS DEVICE</span>
                <span v-if="host.authRequired" class="edge-code">KEY</span>
              </div>
              <div class="data-mono truncate text-caption text-ink-3">
                {{ addressLabel(host) }} · {{ versionLabel(host) }}
              </div>
            </div>
            <button
              v-if="!isThisMachine(host)"
              type="button"
              data-test="discovered-add"
              class="border-edge h-7 shrink-0 rounded-control border px-2.5 text-caption text-ink-2 hover:text-ink disabled:opacity-50"
              :disabled="adding"
              @click="addDiscovered(host)"
            >
              Connect
            </button>
          </div>
          <p v-if="!undiscovered.length && !scanning" class="text-caption text-ink-3">
            No other mold servers found on your network.
          </p>

          <p v-if="actionError" class="text-caption text-stop">{{ actionError }}</p>
        </div>

        <!-- Queue mirror -->
        <div class="shrink-0 xl:w-[300px]">
          <QueueColumn />
        </div>
      </div>
    </div>

    <ConnectMachineModal
      :open="connectOpen"
      @close="connectOpen = false"
      @connected="onConnected"
    />
    <ConfirmDialog
      :open="forgetCandidate !== null"
      title="Forget this machine?"
      :message="`${forgetCandidate?.label ?? 'This machine'} and its saved API key will be removed.`"
      confirm-label="Forget machine"
      danger
      @confirm="forgetHost"
      @cancel="forgetCandidate = null"
    />
  </div>
</template>
