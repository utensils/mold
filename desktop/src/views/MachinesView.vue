<script setup lang="ts">
/*
 * Machines — master/detail (README §03): a 326px list of every machine as a
 * card (This device first, then connected remotes, billing pods, remembered
 * hosts, and anything discovered on the LAN, with the Rent-a-GPU offer last)
 * beside the selected machine's detail pane, which the nested route renders.
 * Landing on /machines opens the machine the shell already talks about. Host
 * CRUD reuses the hosts store verbatim (instance-UUID dedupe, per-host key
 * storage, boot reconnect are unchanged) — this view only frames it.
 */
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import ConnectMachineModal from "../components/machines/ConnectMachineModal.vue";
import PodCostMeter from "../components/machines/PodCostMeter.vue";
import ConfirmDialog from "../components/shell/ConfirmDialog.vue";
import { countPhrase } from "../composables/useShellSubtitle";
import { HOST_RECONNECTING_LABEL } from "@studio/lib/hostConnectivity";
import { ipc, type DiscoveredHost, type SavedHost } from "../lib/ipc";
import { gpuFleetLabel, gpuSnapshotsFromWorkers } from "../lib/api/gpuStatus";
import { addressLabel, prepareHosts, versionLabel } from "../lib/discovery";
import { formatGB, percent } from "../lib/format";
import { hostIdFromUrl, inferBackendFromGpuName } from "../lib/hosts";
import { podGpuName, podProxyUrl, runPodForHostUrl, type RunPodPod } from "../lib/runpod";
import { useHostsStore, type HostView } from "../stores/hosts";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useHostStatusStore } from "../stores/hostStatus";
import { useJobsStore } from "../stores/jobs";
import { useRunPodStore } from "../stores/runpod";
import { useToastStore } from "../stores/toasts";

const route = useRoute();
const router = useRouter();
const hosts = useHostsStore();
const hostStatus = useHostStatusStore();
const appPrefs = useAppPrefsStore();
const contextMenu = useContextMenuStore();
const jobs = useJobsStore();
const runpod = useRunPodStore();
const toasts = useToastStore();

onMounted(() => {
  jobs.startPolling();
  void refreshSaved();
  void scan();
  // Surface any billing pods on the list so a paid host's running cost is
  // visible without opening the RunPod pane (§08 G9). No-ops when unconfigured.
  void runpod.load();
});
onUnmounted(() => jobs.stopPolling());

/** The machine whose detail the pane shows; null on the RunPod pane. */
const selectedId = computed(() =>
  route.name === "host-detail" ? String(route.params.id ?? "") : null,
);

// Landing on the bare workspace opens the machine the shell already talks
// about — the pinned target, else the one making images, else this device.
watch(
  () => [route.path, hosts.all.length] as const,
  () => {
    if (route.path !== "/machines") return;
    const id = hostStatus.displayHost?.id ?? hosts.primaryHost?.id;
    if (id) void router.replace(`/machines/${id}`);
  },
  { immediate: true },
);

async function stopPod(pod: RunPodPod) {
  try {
    await runpod.act("stop", pod.id);
    toasts.push(`Stopped ${pod.name ?? pod.id}`);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

function hostForPod(pod: RunPodPod): HostView | null {
  return hosts.all.find((host) => runPodForHostUrl([pod], host.baseUrl) !== null) ?? null;
}

async function openPodDetail(pod: RunPodPod) {
  try {
    const host = hostForPod(pod) ?? (await hosts.connect(podProxyUrl(pod.id), null, pod.name));
    openDetail(host);
  } catch (err) {
    toasts.push(`The instance is still starting: ${String(err)}`, "error");
  }
}

async function targetPod(pod: RunPodPod) {
  try {
    const host = hostForPod(pod) ?? (await hosts.connect(podProxyUrl(pod.id), null, pod.name));
    await appPrefs.update({ generateTargetHost: host.id });
    toasts.push(`${host.label} is making images from now on`);
  } catch (err) {
    toasts.push(`The instance is still starting: ${String(err)}`, "error");
  }
}

const connectOpen = ref(false);
const connectPromptHost = ref<DiscoveredHost | null>(null);

function openConnectModal(host: DiscoveredHost | null = null) {
  connectPromptHost.value = host;
  connectOpen.value = true;
}

function closeConnectModal() {
  connectOpen.value = false;
  connectPromptHost.value = null;
}

// `?connect=1` (the palette's Connect a machine…) opens the dialog once and
// leaves the address clean.
watch(
  () => route.query.connect,
  (connect) => {
    if (connect === undefined) return;
    openConnectModal();
    const { connect: _connect, ...rest } = route.query;
    void router.replace({ path: route.path, query: rest });
  },
  { immediate: true },
);

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

function isTarget(host: HostView): boolean {
  return (appPrefs.settings?.generateTargetHost ?? null) === host.id;
}

function connectedHostMenu(host: HostView): MenuEntry[] {
  const target = isTarget(host);
  return [
    { label: "Open details", action: () => openDetail(host) },
    {
      label: target ? "Making images here" : "Make images here",
      disabled: target || host.status !== "ready",
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

function podMenu(pod: RunPodPod): MenuEntry[] {
  const host = hostForPod(pod);
  const proxyUrl = podProxyUrl(pod.id);
  const entries: MenuEntry[] = host
    ? connectedHostMenu(host)
    : [
        { label: "Open details", action: () => void openPodDetail(pod) },
        { label: "Make images here", action: () => void targetPod(pod) },
        { label: "Copy address", action: () => void copyAddress(proxyUrl) },
        { label: "Open web UI", action: () => void openHostUrl(proxyUrl) },
      ];
  entries.push(
    { separator: true },
    { label: "Manage RunPod", action: () => void router.push("/machines/runpod") },
  );
  if (!pod.networkVolume) {
    entries.push({
      label: "Stop it",
      danger: true,
      disabled: runpod.mutating === `stop:${pod.id}`,
      action: () => void stopPod(pod),
    });
  }
  return entries;
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

/** "RTX 4090 · CUDA" — the card's hardware, or nothing before telemetry. */
function hardwareLabel(host: HostView): string {
  const gpus = hostGpus(host.id);
  const fleet = gpuFleetLabel(gpus);
  if (!fleet) return "";
  const first = gpus[0];
  return host.kind === "local" && first
    ? `${fleet} · ${(first.backend ?? inferBackendFromGpuName(first.name)).toUpperCase()}`
    : fleet;
}

/** The plain sentence under the name: what it is and where it lives. */
function hostSentence(host: HostView): string {
  const address = host.baseUrl?.replace(/^https?:\/\//, "") ?? "";
  const where =
    host.kind === "local"
      ? "This machine — works without a network."
      : /\.runpod\.net/.test(address)
        ? "Rented cloud GPU — stop it to stop paying."
        : `On your network at ${address}.`;
  const hardware = hardwareLabel(host);
  return hardware ? `${hardware} — ${where.charAt(0).toLowerCase()}${where.slice(1)}` : where;
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
  return percent(used, total);
}

function queueLabel(host: HostView): string {
  const depth = host.queueDepth ?? 0;
  return depth === 0 ? "nothing waiting" : countPhrase(depth, "waiting", "waiting");
}

function reconnecting(host: HostView): boolean {
  return host.status === "error" || host.status === "connecting" || host.stale === true;
}

function statusDot(host: HostView): string {
  if (host.stale || host.status === "connecting") return "bg-sapphire ms-pulse";
  return host.status === "ready" ? "bg-success" : "bg-error";
}

const readyCount = computed(() => hosts.all.filter((h) => h.status === "ready").length);
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

const undiscovered = computed(() => {
  const seenInstanceIds = new Set<string>();
  return discovered.value.filter((d) => {
    const instanceId = d.instanceId?.trim() || null;
    const visible =
      !connectedRemoteIds.value.has(hostIdFromUrl(d.url)) &&
      !(instanceId && connectedRemoteInstanceIds.value.has(instanceId)) &&
      // The app's own embedded server is already the This-device card; its
      // mDNS advertisement is noise here. Instance UUID identifies the
      // primary (so a standalone `mold serve` on this machine stays listed),
      // and isThisMachine guards against a copied MOLD_HOME sharing that
      // UUID from another box.
      !(d.isThisMachine && instanceId && instanceId === hosts.primaryHost?.instanceId) &&
      !(instanceId && seenInstanceIds.has(instanceId));
    if (visible && instanceId) seenInstanceIds.add(instanceId);
    return visible;
  });
});

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
    if (host.authRequired && !key) {
      openConnectModal(host);
      return;
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
  <div class="flex h-full min-h-0 bg-bg">
    <!-- the machine list -->
    <aside
      class="flex w-[var(--mold-shell-secondary-w)] shrink-0 flex-col border-r border-border bg-chrome"
      data-test="machines-list"
    >
      <div
        class="flex h-[var(--mold-shell-viewbar-h)] shrink-0 items-center gap-2.5 border-b border-border px-3.5"
      >
        <span class="ms-group-label uppercase">Connected · {{ readyCount }}</span>
        <span class="flex-1" />
        <button
          type="button"
          data-test="add-machine"
          class="ms-toolbar-button"
          @click="openConnectModal()"
        >
          <Icon name="plus" :size="13" :stroke-width="2" />
          Connect a machine
        </button>
      </div>

      <div class="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-2.5">
        <!-- This device + connected remotes -->
        <button
          v-for="host in hosts.all"
          :key="host.id"
          type="button"
          :data-test="host.primary ? 'this-device-card' : 'host-card'"
          class="machine-card"
          :class="{
            'machine-card--target': isTarget(host),
            'machine-card--selected': selectedId === host.id,
          }"
          :aria-current="selectedId === host.id ? 'true' : undefined"
          @click="openDetail(host)"
          @contextmenu="contextMenu.open($event, connectedHostMenu(host))"
        >
          <span class="flex items-center gap-2.5">
            <span class="h-2 w-2 shrink-0 rounded-full" :class="statusDot(host)" />
            <span class="truncate font-mono text-xs font-bold text-fg">{{ host.label }}</span>
            <span class="flex-1" />
            <span
              v-if="isTarget(host)"
              data-test="target-badge"
              class="shrink-0 rounded-inner bg-accent px-1.5 py-0.5 font-mono text-micro text-on-accent"
              >making images here</span
            >
          </span>
          <span class="text-xs leading-snug text-fg-2">{{ hostSentence(host) }}</span>
          <!-- The 10 s status poll keeps probing an unreachable machine, so
               it comes back on its own; say so rather than leaving a bare
               red dot that reads as "gone". -->
          <span
            v-if="reconnecting(host)"
            class="text-micro text-warning"
            data-test="host-reconnecting"
          >
            {{ HOST_RECONNECTING_LABEL }}
          </span>
          <span class="block h-[5px] overflow-hidden bg-bg-crust" aria-hidden="true">
            <span
              class="block h-full"
              :class="host.kind === 'local' ? 'bg-sapphire' : 'bg-accent'"
              :style="{ width: `${memoryPct(host)}%` }"
            />
          </span>
          <span class="flex justify-between font-mono text-micro text-fg-dim">
            <span>{{ memoryLabel(host) ?? "no telemetry yet" }}</span>
            <span>{{ queueLabel(host) }}</span>
          </span>
        </button>

        <!-- Billing pods not yet connected as machines: cost meter + Stop (§08 G9) -->
        <div
          v-for="pod in runpod.runningPods"
          :key="pod.id"
          data-test="runpod-running"
          class="machine-card machine-card--pod relative cursor-pointer"
          @click="openPodDetail(pod)"
          @contextmenu="contextMenu.open($event, podMenu(pod))"
        >
          <button
            type="button"
            data-test="runpod-open"
            :aria-label="`Open ${pod.name ?? pod.id} machine details`"
            class="absolute inset-0 rounded-control focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            @click.stop="openPodDetail(pod)"
          />
          <span class="pointer-events-none relative z-10 flex items-center gap-2.5">
            <span class="h-2 w-2 shrink-0 rounded-full bg-teal" />
            <span class="truncate font-mono text-xs font-bold text-fg">{{
              pod.name ?? pod.id
            }}</span>
            <span class="flex-1" />
            <Icon
              name="chevron-right"
              data-test="machine-chevron"
              :size="14"
              :stroke-width="2"
              class="shrink-0 text-fg-dim"
            />
          </span>
          <span class="pointer-events-none relative z-10 text-xs leading-snug text-fg-2">
            {{ podGpuName(pod) }} — rented cloud GPU. Stop it to stop paying.
          </span>
          <PodCostMeter
            :cost-per-hr="pod.costPerHr"
            :uptime-seconds="pod.uptimeSeconds"
            :stoppable="!pod.networkVolume"
            :busy="runpod.mutating === `stop:${pod.id}`"
            class="relative z-10"
            @stop="stopPod(pod)"
          />
        </div>

        <!-- Remembered (offline) -->
        <template v-if="rememberedHosts.length">
          <span class="ms-group-label mt-2 px-1 uppercase">Remembered</span>
          <div
            v-for="saved in rememberedHosts"
            :key="saved.id"
            data-test="remembered-host"
            class="machine-card machine-card--dim"
            @contextmenu="contextMenu.open($event, rememberedHostMenu(saved))"
          >
            <span class="flex items-center gap-2.5">
              <span class="h-2 w-2 shrink-0 rounded-full bg-fg-dim" />
              <span class="min-w-0 flex-1 truncate font-mono text-xs font-bold text-fg">
                {{ savedHostLabel(saved) }}
              </span>
              <button
                type="button"
                data-test="remembered-connect"
                class="ms-toolbar-button"
                :disabled="adding"
                @click="connectSaved(saved)"
              >
                Connect
              </button>
            </span>
            <span class="truncate font-mono text-micro text-fg-dim">{{ saved.url }}</span>
          </div>
        </template>

        <!-- On your network -->
        <span class="mt-2 flex items-center gap-2 px-1">
          <span class="ms-group-label uppercase">On your network</span>
          <span class="flex-1" />
          <button
            type="button"
            class="text-micro text-fg-dim hover:text-fg disabled:text-fg-faint"
            :disabled="scanning"
            @click="scan"
          >
            {{ scanning ? "Scanning…" : "Scan again" }}
          </button>
        </span>
        <div
          v-for="host in undiscovered"
          :key="host.url"
          data-test="discovered-host"
          class="machine-card"
          @contextmenu="contextMenu.open($event, discoveredHostMenu(host))"
        >
          <span class="flex items-center gap-2.5">
            <span class="h-2 w-2 shrink-0 rounded-full bg-sapphire" />
            <span class="min-w-0 flex-1 truncate font-mono text-xs font-bold text-fg">
              {{ host.name }}
            </span>
            <span v-if="isThisMachine(host)" class="font-mono text-micro text-fg-dim"
              >THIS DEVICE</span
            >
            <span v-if="host.authRequired" class="font-mono text-micro text-fg-dim">KEY</span>
            <button
              v-if="!isThisMachine(host)"
              type="button"
              data-test="discovered-add"
              class="ms-toolbar-button"
              :disabled="adding"
              @click="addDiscovered(host)"
            >
              Connect
            </button>
          </span>
          <span class="truncate font-mono text-micro text-fg-dim">
            {{ addressLabel(host) }} · {{ versionLabel(host) }}
          </span>
        </div>
        <p v-if="!undiscovered.length && !scanning" class="px-1 text-micro text-fg-dim">
          No other mold servers found on your network.
        </p>
        <p v-if="actionError" class="px-1 text-micro text-error">{{ actionError }}</p>

        <!-- Rent a GPU -->
        <div
          class="mt-2 flex flex-col gap-2.5 rounded-control border border-dashed border-surface-3 p-3.5"
        >
          <span class="flex items-center gap-2.5">
            <Icon name="cloud" :size="15" class="text-fg-dim" />
            <span class="font-mono text-xs font-bold text-fg-2">runpod cloud</span>
          </span>
          <span class="text-xs leading-snug text-fg-dim">
            Rent a fast GPU by the minute when your own machine is busy. You pay only while it runs.
          </span>
          <button
            type="button"
            data-test="start-pod"
            class="ms-toolbar-button ms-toolbar-button--on self-start font-semibold"
            @click="router.push('/machines/runpod')"
          >
            Rent a GPU
          </button>
        </div>
      </div>
    </aside>

    <!-- the detail pane: a machine, or the RunPod console -->
    <div class="flex min-w-0 flex-1 flex-col">
      <RouterView />
    </div>

    <ConnectMachineModal
      :open="connectOpen"
      :initial-host="connectPromptHost"
      @close="closeConnectModal"
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

<style scoped>
/* Machine card (README §04): dot · mono name · sentence · meter · two mono
   readouts. The target machine gets a 1px accent border; the selected one the
   row-selected fill with an inset ring. */
.machine-card {
  display: flex;
  width: 100%;
  flex-direction: column;
  gap: 9px;
  padding: 13px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-surface);
  text-align: left;
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    background var(--mold-dur-quick) var(--mold-ease-out);
}
.machine-card:hover {
  border-color: var(--mold-border-focus);
}
.machine-card--target {
  border-color: var(--mold-blue);
}
.machine-card--selected,
.machine-card--selected:hover {
  border-color: var(--mold-blue);
  background: var(--mold-row-selected);
  box-shadow: inset 0 0 0 1px var(--mold-blue);
}
.machine-card--pod {
  border-color: color-mix(in srgb, var(--mold-teal) 50%, var(--mold-border));
}
.machine-card--dim {
  opacity: 0.7;
}
</style>
