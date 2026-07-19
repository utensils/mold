<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import { outputFormatsForFamily } from "../lib/capabilities";
import type { GalleryImage, ModelEntry, ServerStatus } from "../lib/api/types";
import {
  applyModelDefaults,
  buildRequest,
  newGenerateForm,
  type GenerateForm,
} from "../lib/generateForm";
import { galleryMediaPath, isVideoItem } from "../lib/gallery/media";
import {
  isCancelledError,
  jobStatusCode,
  railOrder,
  useGenerationStore,
  type Job,
} from "../stores/generation";
import { mobileHostTarget, normalizeRemoteAddress, remoteHostId, type MobileHost } from "./hosts";
import { applyMobileGalleryMetadata } from "./reuse";
import MobileCatalogView from "./MobileCatalogView.vue";
import MobileGalleryViewer from "./MobileGalleryViewer.vue";
import MobileHostDetail from "./MobileHostDetail.vue";
import MobileResolutionPicker from "./MobileResolutionPicker.vue";

type Tab = "generate" | "gallery" | "catalog" | "hosts";

interface DiscoveredHost {
  name: string;
  host: string;
  port: number;
}

interface GalleryPrint extends GalleryImage {
  hostId: string;
  hostName: string;
  target: ApiTarget;
  thumbnailUrl: string;
}

interface PendingGalleryPrint extends GalleryImage {
  hostId: string;
  hostName: string;
  target: ApiTarget;
}

const STORAGE_KEY = "mold.mobile.hosts.v1";
const SELECTED_KEY = "mold.mobile.selected-host.v1";
const HOST_PROBE_TIMEOUT_MS = 9_000;
const tab = ref<Tab>("generate");
const hosts = ref<MobileHost[]>(loadHosts());
const selectedHostId = ref(localStorage.getItem(SELECTED_KEY) ?? hosts.value[0]?.id ?? "");
const catalogHostId = ref(selectedHostId.value || hosts.value[0]?.id || "");
const hostDetailId = ref("");
const hostInput = reactive({ name: "", address: "", apiKey: "" });
const discovered = ref<DiscoveredHost[]>([]);
const discovering = ref(false);
const hostError = ref("");
const models = ref<ModelEntry[]>([]);
const modelsHostId = ref("");
const loadingModels = ref(false);
const form = reactive<GenerateForm>(newGenerateForm());
const progress = ref("Ready");
const generationAnnouncement = ref("");
const gallery = ref<GalleryPrint[]>([]);
const galleryLoading = ref(false);
const galleryLoadingMore = ref(false);
const galleryError = ref("");
const galleryRemaining = ref(0);
const selectedPrint = ref<GalleryPrint | null>(null);
const reusingPrint = ref(false);
const reusePrintError = ref("");
const latestResultClientId = ref<number | null>(null);
const resultMediaLoadKey = ref(0);
const objectUrls = new Set<string>();
const handledGenerationClientIds = new Set<number>();
let pendingGallery: PendingGalleryPrint[] = [];
let modelLoadEpoch = 0;
let galleryRefreshRequested = false;
let galleryRefreshDeferred = false;
let galleryRefreshTask: Promise<void> | null = null;
let galleryOperationTail: Promise<void> = Promise.resolve();
let resultMediaRecoveryClientId: number | null = null;
let resultMediaRecoveryAttempts = 0;
let hostProbeTimer: ReturnType<typeof setInterval> | null = null;
let hostProbeEpoch = 0;
const hostProbes = new Map<
  string,
  { epoch: number; controller: AbortController; timeout: ReturnType<typeof setTimeout> }
>();
const generation = useGenerationStore();

const selectedHost = computed(() => hosts.value.find((host) => host.id === selectedHostId.value));
const hostDetail = computed(() => hosts.value.find((host) => host.id === hostDetailId.value));
const selectedPrintIndex = computed(() => {
  const selected = selectedPrint.value;
  if (!selected) return -1;
  return gallery.value.findIndex(
    (print) => print.hostId === selected.hostId && print.filename === selected.filename,
  );
});
const selectedTarget = computed<ApiTarget | null>(() => {
  const host = selectedHost.value;
  return host ? mobileHostTarget(host) : null;
});
const outputFormats = computed(() => outputFormatsForFamily(form.family));
const selectedModelAvailable = computed(
  () =>
    modelsHostId.value === selectedHostId.value &&
    models.value.some((model) => model.name === form.model),
);
const queuedJobs = computed(() => railOrder(generation.pending));
const activeGeneration = computed(() => {
  const active = generation.active;
  return active && active.status !== "complete" && active.status !== "error" ? active : null;
});
const latestResultJob = computed(() => {
  const latest = generation.jobs.find((job) => job.clientId === latestResultClientId.value);
  // Once a completion is promoted, never put an older print underneath its
  // new seed/status while a saved-file URL is loading or has failed.
  if (latestResultClientId.value !== null) {
    return latest?.status === "complete" ? latest : null;
  }
  for (let index = generation.jobs.length - 1; index >= 0; index -= 1) {
    const job = generation.jobs[index];
    if (job?.status === "complete") return job;
  }
  return null;
});
const resultUrl = computed(() => latestResultJob.value?.resultUrl ?? "");
const resultIsVideo = computed(() => latestResultJob.value?.result?.format === "mp4");
const resultPreviewError = computed(() => latestResultJob.value?.resultError ?? "");
const developButtonLabel = computed(() =>
  queuedJobs.value.length > 0
    ? `Develop print (+${queuedJobs.value.length} queued)`
    : "Develop print",
);
const queueAnnouncement = computed(() => {
  const count = queuedJobs.value.length;
  return count === 0
    ? "No active generations."
    : `${count} active generation${count === 1 ? "" : "s"}.`;
});
const generationStatus = computed(() => {
  const active = activeGeneration.value;
  if (!active) return progress.value;
  switch (active.status) {
    case "queued":
      return active.queuePosition && active.queuePosition > 0
        ? `Queued #${active.queuePosition}`
        : "Queued";
    case "loading":
      return active.stage ?? "Loading model";
    case "denoising":
      return `Developing ${active.step} / ${active.total}`;
    case "finishing":
      return active.stage ?? "Finalizing";
    default:
      return jobStatusCode(active);
  }
});

function loadHosts(): MobileHost[] {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as MobileHost[];
    return raw.map((host) => ({ ...host, apiKey: "", online: false }));
  } catch {
    return [];
  }
}

function persistHosts(): void {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(hosts.value.map(({ apiKey: _apiKey, ...host }) => host)),
  );
}

async function hydrateApiKeys(): Promise<void> {
  await Promise.all(
    hosts.value.map(async (host) => {
      host.apiKey =
        (await invoke<string | null>("keychain_get_api_key", { hostId: host.id })) ?? "";
    }),
  );
}

async function connectHost(address?: string, discoveredName?: string): Promise<void> {
  hostError.value = "";
  try {
    const baseUrl = normalizeRemoteAddress(address ?? hostInput.address);
    const target = { baseUrl, apiKey: hostInput.apiKey.trim() || null };
    const status = await apiJsonTo<ServerStatus>(target, "/api/status");
    const instanceId = status.instance_id ?? undefined;
    const existing = hosts.value.find(
      (host) =>
        host.baseUrl === baseUrl ||
        (instanceId &&
          (host.instanceId === instanceId || host.id === instanceId) &&
          (!host.hostname || !status.hostname || host.hostname === status.hostname)),
    );
    // URL identity keeps two machines that copied the same MOLD_HOME distinct;
    // a compatible saved alias keeps its existing keychain id.
    const id = existing?.id ?? remoteHostId(baseUrl);
    const saved: MobileHost = {
      id,
      name: hostInput.name.trim() || discoveredName || status.hostname || new URL(baseUrl).hostname,
      baseUrl,
      apiKey: hostInput.apiKey.trim(),
      hostname: status.hostname ?? undefined,
      version: status.version,
      instanceId,
      online: true,
    };
    if (existing) Object.assign(existing, saved);
    else hosts.value.push(saved);
    if (saved.apiKey) {
      await invoke("keychain_set_api_key", { hostId: saved.id, apiKey: saved.apiKey });
    } else {
      await invoke("keychain_delete_api_key", { hostId: saved.id });
    }
    persistHosts();
    selectedHostId.value = saved.id;
    catalogHostId.value = saved.id;
    tab.value = "generate";
    hostInput.name = "";
    hostInput.address = "";
    hostInput.apiKey = "";
    await refreshModels();
  } catch (error) {
    hostError.value = error instanceof Error ? error.message : String(error);
  }
}

async function discoverHosts(): Promise<void> {
  discovering.value = true;
  hostError.value = "";
  try {
    discovered.value = await invoke<DiscoveredHost[]>("discover_mold_hosts", { timeoutMs: 2500 });
  } catch (error) {
    hostError.value = error instanceof Error ? error.message : String(error);
  } finally {
    discovering.value = false;
  }
}

async function selectHost(id: string): Promise<void> {
  selectedHostId.value = id;
  await refreshModels();
}

function showHostDetail(id: string): void {
  hostDetailId.value = id;
}

function renameHost(payload: { id: string; name: string }): void {
  const host = hosts.value.find((candidate) => candidate.id === payload.id);
  if (!host) return;
  host.name = payload.name;
  persistHosts();
}

function updateHostStatus(payload: { id: string; status: ServerStatus | null }): void {
  const host = hosts.value.find((candidate) => candidate.id === payload.id);
  if (!host) return;
  host.online = payload.status !== null;
  if (payload.status) {
    host.version = payload.status.version;
    host.hostname = payload.status.hostname ?? undefined;
    host.instanceId = payload.status.instance_id ?? host.instanceId;
  }
}

function cancelHostProbe(id: string): void {
  const probe = hostProbes.get(id);
  if (!probe) return;
  probe.controller.abort();
  clearTimeout(probe.timeout);
  hostProbes.delete(id);
}

async function probeHost(host: MobileHost): Promise<void> {
  cancelHostProbe(host.id);
  const controller = new AbortController();
  const epoch = ++hostProbeEpoch;
  const timeout = setTimeout(() => controller.abort(), HOST_PROBE_TIMEOUT_MS);
  const probe = { epoch, controller, timeout };
  hostProbes.set(host.id, probe);
  try {
    const status = await apiJsonTo<ServerStatus>(mobileHostTarget(host), "/api/status", {
      signal: controller.signal,
    });
    if (hostProbes.get(host.id)?.epoch !== epoch) return;
    updateHostStatus({ id: host.id, status });
  } catch {
    if (hostProbes.get(host.id)?.epoch !== epoch) return;
    updateHostStatus({ id: host.id, status: null });
  } finally {
    if (hostProbes.get(host.id)?.epoch === epoch) hostProbes.delete(host.id);
    clearTimeout(timeout);
  }
}

function probeHosts(): void {
  for (const host of hosts.value) void probeHost(host);
}

function removeHost(id: string): void {
  cancelHostProbe(id);
  const removedSelectedHost = selectedHostId.value === id;
  const removedCatalogHost = catalogHostId.value === id;
  if (hostDetailId.value === id) hostDetailId.value = "";
  hosts.value = hosts.value.filter((host) => host.id !== id);
  if (removedSelectedHost) {
    selectedHostId.value = hosts.value[0]?.id ?? "";
    models.value = [];
    modelsHostId.value = "";
    void refreshModels();
  }
  if (removedCatalogHost) catalogHostId.value = hosts.value[0]?.id ?? "";
  persistHosts();
  void invoke("keychain_delete_api_key", { hostId: id });
}

function selectCatalogHost(id: string): void {
  if (hosts.value.some((host) => host.id === id)) catalogHostId.value = id;
}

function openCatalog(id?: string): void {
  if (id && hosts.value.some((host) => host.id === id)) catalogHostId.value = id;
  else if (!hosts.value.some((host) => host.id === catalogHostId.value)) {
    catalogHostId.value = selectedHostId.value || hosts.value[0]?.id || "";
  }
  hostDetailId.value = "";
  tab.value = "catalog";
}

function catalogModelsChanged(hostId: string): void {
  if (hostId === selectedHostId.value) void refreshModels();
}

async function refreshModels(): Promise<boolean> {
  const epoch = ++modelLoadEpoch;
  const host = selectedHost.value;
  if (!host) {
    models.value = [];
    modelsHostId.value = "";
    loadingModels.value = false;
    return false;
  }
  const hostId = host.id;
  const target = { baseUrl: host.baseUrl, apiKey: host.apiKey || null };
  loadingModels.value = true;
  models.value = [];
  modelsHostId.value = "";
  try {
    const [status, entries] = await Promise.all([
      apiJsonTo<ServerStatus>(target, "/api/status"),
      apiJsonTo<ModelEntry[]>(target, "/api/models"),
    ]);
    if (epoch !== modelLoadEpoch || selectedHostId.value !== hostId) return false;
    host.online = true;
    host.version = status.version;
    host.hostname = status.hostname ?? undefined;
    models.value = entries.filter((model) => model.downloaded);
    modelsHostId.value = hostId;
    if (!models.value.some((model) => model.name === form.model) && models.value[0]) {
      applyModelDefaults(form, models.value[0]);
    }
    return true;
  } catch (error) {
    if (epoch !== modelLoadEpoch || selectedHostId.value !== hostId) return false;
    host.online = false;
    progress.value = error instanceof Error ? error.message : String(error);
    return false;
  } finally {
    if (epoch === modelLoadEpoch && selectedHostId.value === hostId) loadingModels.value = false;
  }
}

function changeModel(): void {
  const model = models.value.find((entry) => entry.name === form.model);
  if (model) applyModelDefaults(form, model);
}

function revokeObjectUrl(url: string): void {
  URL.revokeObjectURL(url);
  objectUrls.delete(url);
}

function generate(): void {
  const host = selectedHost.value;
  const target = selectedTarget.value;
  if (!host || !target || !form.prompt.trim() || !selectedModelAvailable.value) return;

  // Exactly like desktop: snapshot the request and host for this tap, open a
  // separate SSE stream immediately, and let the remote engine schedule it.
  const { settled } = generation.submitBatch(buildRequest(form), 1, {
    hostId: host.id,
    label: host.name,
    kind: "remote",
    target: { ...target },
    mirrorRemoteOutput: false,
    retainEncodedResult: false,
    metadataOnlyCompletion: true,
  });
  progress.value = "Queued";
  generationAnnouncement.value = "";
  void settled.then((jobs) => {
    const job = jobs[0];
    if (!job) return;
    handledGenerationClientIds.add(job.clientId);
    if (job.status === "complete" && job.result) {
      latestResultClientId.value = job.clientId;
      if (job.resultError) {
        progress.value = job.resultError;
        generationAnnouncement.value = `Generation completed, but its preview is unavailable. ${job.resultError}`;
      } else {
        progress.value = `${(job.result.generation_time_ms / 1000).toFixed(1)}s · seed ${job.result.seed_used}`;
        generationAnnouncement.value = "Generation completed.";
      }
      if (tab.value === "gallery") void refreshGallery();
    } else if (job.error?.includes("remote cancellation was not confirmed")) {
      progress.value = job.error;
      generationAnnouncement.value = `Cancellation failed. ${job.error}`;
    } else if (job.error && !isCancelledError(job.error)) {
      progress.value = job.error;
      generationAnnouncement.value = `Generation failed. ${job.error}`;
    } else if (isCancelledError(job.error)) {
      progress.value = "Cancelled";
      generationAnnouncement.value = "Generation cancelled.";
    }
    // Only terminal jobs whose callbacks have run are eligible: multiple
    // completion microtasks cannot prune one another before they promote the
    // correct latest result. The UI renders one result, so retain one Blob.
    generation.prune(1, latestResultClientId.value, handledGenerationClientIds);
    for (const clientId of handledGenerationClientIds) {
      if (!generation.jobs.some((candidate) => candidate.clientId === clientId)) {
        handledGenerationClientIds.delete(clientId);
      }
    }
  });
}

async function cancelGeneration(job: Job): Promise<void> {
  try {
    await generation.cancel(job.clientId);
    if (job.status === "complete" && job.result) {
      latestResultClientId.value = job.clientId;
      progress.value = `${(job.result.generation_time_ms / 1000).toFixed(1)}s · seed ${job.result.seed_used}`;
      generationAnnouncement.value = "Generation completed.";
      if (tab.value === "gallery") void refreshGallery();
    } else if (job.error && !isCancelledError(job.error)) {
      progress.value = job.error;
      generationAnnouncement.value = `Generation failed. ${job.error}`;
    } else {
      progress.value = "Cancelled";
      generationAnnouncement.value = "Generation cancelled.";
    }
  } catch (error) {
    progress.value = error instanceof Error ? error.message : String(error);
    generationAnnouncement.value = `Cancellation failed. ${progress.value}`;
  }
}

function renewGeneratedResult(force: boolean): void {
  const job = latestResultJob.value;
  if (!job?.metadataOnlyCompletion || !job.result || job.resultUrlLoading) return;
  const previousUrl = job.resultUrl;
  void generation
    .refreshRemoteResultUrl(job.clientId, force)
    .then(() => {
      if (latestResultClientId.value !== job.clientId || job.resultError || !job.resultUrl) return;
      if (force) resultMediaLoadKey.value += 1;
      progress.value = `${(job.result!.generation_time_ms / 1000).toFixed(1)}s · seed ${job.result!.seed_used}`;
      if (force || job.resultUrl !== previousUrl) {
        generationAnnouncement.value = "Result preview refreshed.";
      }
    })
    .catch(() => {
      // The store exposes the directed failure through resultError.
    });
}

function generatedMediaReady(): void {
  resultMediaRecoveryClientId = latestResultClientId.value;
  resultMediaRecoveryAttempts = 0;
}

function recoverGeneratedMedia(): void {
  const job = latestResultJob.value;
  if (!job || job.resultUrlLoading) return;
  if (resultMediaRecoveryClientId !== job.clientId) {
    resultMediaRecoveryClientId = job.clientId;
    resultMediaRecoveryAttempts = 0;
  }
  if (resultMediaRecoveryAttempts === 0) {
    resultMediaRecoveryAttempts = 1;
    renewGeneratedResult(true);
    return;
  }

  if (job.resultUrl && job.resultUrlIsObjectUrl) URL.revokeObjectURL(job.resultUrl);
  job.resultUrl = null;
  job.resultUrlIsObjectUrl = false;
  job.resultUrlExpiresAt = null;
  job.resultError = "Couldn’t load this generated print from the host.";
}

function retryGeneratedPreview(): void {
  resultMediaRecoveryClientId = latestResultClientId.value;
  resultMediaRecoveryAttempts = 0;
  renewGeneratedResult(true);
}

async function thumbnailUrl(target: ApiTarget, filename: string): Promise<string> {
  const response = await apiFetchTo(target, galleryMediaPath(filename, "host", true));
  const url = URL.createObjectURL(await response.blob());
  objectUrls.add(url);
  return url;
}

function refreshGallery(): Promise<void> {
  if (selectedPrint.value) {
    // The viewer uses the grid thumbnail as its placeholder/poster and
    // returns focus to that tile. Keep both alive until the viewer closes.
    galleryRefreshDeferred = true;
    return Promise.resolve();
  }
  galleryRefreshRequested = true;
  if (!galleryRefreshTask) {
    const operation = enqueueGalleryOperation(async () => {
      while (galleryRefreshRequested) {
        if (selectedPrint.value) {
          galleryRefreshRequested = false;
          galleryRefreshDeferred = true;
          break;
        }
        galleryRefreshRequested = false;
        await performGalleryRefresh();
      }
    });
    galleryRefreshTask = operation.then(
      async () => {
        galleryRefreshTask = null;
        // A request can arrive after the loop's final condition but before
        // this continuation. Adopt the re-armed task so every caller waits
        // for the refresh it requested.
        if (galleryRefreshRequested) await refreshGallery();
      },
      (error: unknown) => {
        galleryRefreshTask = null;
        throw error;
      },
    );
  }
  return galleryRefreshTask;
}

function enqueueGalleryOperation(operation: () => Promise<void>): Promise<void> {
  const task = galleryOperationTail.then(operation, operation);
  galleryOperationTail = task.catch(() => {});
  return task;
}

async function performGalleryRefresh(): Promise<void> {
  galleryLoading.value = true;
  galleryError.value = "";
  const prior = gallery.value;
  gallery.value = [];
  for (const item of prior) revokeObjectUrl(item.thumbnailUrl);
  const results = await Promise.allSettled(
    hosts.value.map(async (host) => {
      const target = { baseUrl: host.baseUrl, apiKey: host.apiKey || null };
      const prints = await apiJsonTo<GalleryImage[]>(target, "/api/gallery");
      return prints.map((print) => ({
        ...print,
        hostId: host.id,
        hostName: host.name,
        target,
      }));
    }),
  );
  pendingGallery = results
    .flatMap((result) => (result.status === "fulfilled" ? result.value : []))
    .sort((a, b) => b.timestamp - a.timestamp);
  const failed = results.filter((result) => result.status === "rejected").length;
  if (failed) galleryError.value = `${failed} host${failed === 1 ? "" : "s"} unavailable`;
  await loadMoreGalleryPage();
  galleryLoading.value = false;
}

function loadMoreGallery(): Promise<void> {
  return enqueueGalleryOperation(loadMoreGalleryPage);
}

async function loadMoreGalleryPage(): Promise<void> {
  galleryLoadingMore.value = true;
  const page = pendingGallery.splice(0, 40);
  for (let offset = 0; offset < page.length; offset += 4) {
    const batch = await Promise.allSettled(
      page.slice(offset, offset + 4).map(async ({ target, ...print }) => ({
        ...print,
        target,
        thumbnailUrl: await thumbnailUrl(target, print.filename),
      })),
    );
    gallery.value.push(
      ...batch.flatMap((result) => (result.status === "fulfilled" ? [result.value] : [])),
    );
  }
  galleryRemaining.value = pendingGallery.length;
  galleryLoadingMore.value = false;
}

async function reusePrint(print: GalleryPrint): Promise<void> {
  if (reusingPrint.value || print.metadata_synthetic || !print.metadata.prompt?.trim()) return;
  reusingPrint.value = true;
  reusePrintError.value = "";
  try {
    if (selectedHostId.value !== print.hostId) {
      selectedHostId.value = print.hostId;
    }
    if (modelsHostId.value !== print.hostId) {
      if (!(await refreshModels())) {
        reusePrintError.value = `Couldn’t load models from ${print.hostName}. Check the host and try again.`;
        return;
      }
    }
    if (models.value.length === 0) {
      reusePrintError.value = `${print.hostName} has no downloaded models available.`;
      return;
    }
    const reuse = applyMobileGalleryMetadata(form, print.metadata, models.value);
    progress.value = reuse.substitutedModel
      ? `The original model isn’t installed on ${print.hostName}; using ${reuse.modelName}.`
      : "Prompt settings restored";
    selectedPrint.value = null;
    // The next Gallery visit performs its normal refresh; do not refetch the
    // grid while navigating directly to the restored prompt.
    galleryRefreshDeferred = false;
    tab.value = "generate";
    void nextTick(() => document.querySelector<HTMLTextAreaElement>("#mobile-prompt")?.focus());
  } finally {
    reusingPrint.value = false;
  }
}

function openPrint(print: GalleryPrint): void {
  reusePrintError.value = "";
  selectedPrint.value = print;
}

function navigateSelectedPrint(delta: -1 | 1): void {
  const next = gallery.value[selectedPrintIndex.value + delta];
  if (!next || reusingPrint.value) return;
  reusePrintError.value = "";
  selectedPrint.value = next;
}

function closePrint(): void {
  if (reusingPrint.value) return;
  reusePrintError.value = "";
  selectedPrint.value = null;
  if (galleryRefreshDeferred || galleryRefreshRequested) {
    galleryRefreshDeferred = false;
    // The viewer normally restores focus to its tile. A deferred refresh — or
    // one still queued behind Load older — will replace that tile, so move
    // focus to the stable Gallery tab first.
    void nextTick(() => {
      document.querySelector<HTMLButtonElement>("[data-test='mobile-tab-gallery']")?.focus();
      void refreshGallery();
    });
  }
}

function reuseSelectedPrint(): void {
  const print = selectedPrint.value;
  if (print) void reusePrint(print);
}

watch(selectedHostId, (id) => {
  if (id) localStorage.setItem(SELECTED_KEY, id);
  else localStorage.removeItem(SELECTED_KEY);
});

watch(tab, (next) => {
  if (next === "gallery") void refreshGallery();
  if (next !== "hosts") hostDetailId.value = "";
});

watch(resultPreviewError, (error) => {
  if (!error) return;
  progress.value = error;
  generationAnnouncement.value = `Generation completed, but its preview is unavailable. ${error}`;
});

onMounted(async () => {
  await hydrateApiKeys();
  // Start the cadence before awaiting individual tailnet hosts. One slow host
  // must not prevent every other saved host from being probed on schedule.
  hostProbeTimer = setInterval(probeHosts, 10_000);
  if (selectedHost.value) {
    await Promise.all([
      refreshModels(),
      ...hosts.value
        .filter((host) => host.id !== selectedHostId.value)
        .map((host) => probeHost(host)),
    ]);
  } else {
    tab.value = "hosts";
  }
});

onBeforeUnmount(() => {
  if (hostProbeTimer) clearInterval(hostProbeTimer);
  hostProbeTimer = null;
  for (const id of [...hostProbes.keys()]) cancelHostProbe(id);
  generation.resetJobs();
  for (const url of objectUrls) URL.revokeObjectURL(url);
});
</script>

<template>
  <main class="mobile-shell">
    <header class="mobile-header">
      <div class="mobile-wordmark">Mold</div>
      <div class="host-chip">{{ selectedHost?.name ?? "Remote only" }}</div>
    </header>

    <p class="sr-only" aria-live="polite" aria-atomic="true">
      {{ queueAnnouncement }}
    </p>
    <p class="sr-only" aria-live="polite" aria-atomic="true">
      {{ generationAnnouncement }}
    </p>

    <section class="mobile-content">
      <template v-if="tab === 'generate'">
        <div v-if="!selectedHost" class="empty-state">
          <div>
            <h1 class="section-title">Connect a host</h1>
            <p>Generation runs on a remote Mold engine.</p>
            <button class="primary-button" type="button" @click="tab = 'hosts'">Add host</button>
          </div>
        </div>
        <template v-else>
          <h1 class="section-title">Generate</h1>
          <p class="section-note">Develop on {{ selectedHost.name }}</p>
          <label class="field">
            <span>Model</span>
            <select
              v-model="form.model"
              class="control"
              :disabled="loadingModels"
              @change="changeModel"
            >
              <option v-for="model in models" :key="model.name" :value="model.name">
                {{ model.name }}
              </option>
            </select>
          </label>
          <label class="field">
            <span>Prompt</span>
            <textarea
              id="mobile-prompt"
              v-model="form.prompt"
              class="control"
              placeholder="Describe the print…"
            />
          </label>
          <label class="field">
            <span>Negative prompt</span>
            <input v-model="form.negativePrompt" class="control" placeholder="Optional" />
          </label>
          <MobileResolutionPicker
            v-model:width="form.width"
            v-model:height="form.height"
            :family="form.family"
            :disabled="loadingModels"
          />
          <div class="field-grid">
            <label class="field"
              ><span>Steps</span
              ><input v-model.number="form.steps" class="control" type="number" inputmode="numeric"
            /></label>
            <label class="field"
              ><span>Guidance</span
              ><input
                v-model.number="form.guidance"
                class="control"
                type="number"
                inputmode="decimal"
                step="0.1"
            /></label>
            <label class="field"
              ><span>Seed</span
              ><input v-model="form.seed" class="control" inputmode="numeric" placeholder="Random"
            /></label>
            <label class="field"
              ><span>Format</span
              ><select v-model="form.outputFormat" class="control">
                <option v-for="format in outputFormats" :key="format" :value="format">
                  {{ format.toUpperCase() }}
                </option>
              </select></label
            >
          </div>
          <template v-if="form.family.includes('video') || form.family.includes('ltx2')">
            <div class="field-grid">
              <label class="field"
                ><span>Frames</span
                ><input
                  v-model.number="form.frames"
                  class="control"
                  type="number"
                  inputmode="numeric"
              /></label>
              <label class="field"
                ><span>FPS</span
                ><input v-model.number="form.fps" class="control" type="number" inputmode="numeric"
              /></label>
            </div>
          </template>
          <button
            class="primary-button"
            type="button"
            :disabled="!form.prompt.trim() || !selectedModelAvailable"
            data-test="mobile-develop-button"
            @click="generate"
          >
            {{ developButtonLabel }}
          </button>
          <section
            v-if="queuedJobs.length"
            class="mobile-generation-queue"
            aria-label="Generation queue"
            data-test="mobile-generation-queue"
          >
            <div class="mobile-generation-queue-head">
              <h2>Queue</h2>
              <span>{{ queuedJobs.length }} active</span>
            </div>
            <ol>
              <li
                v-for="job in queuedJobs"
                :key="job.clientId"
                class="mobile-generation-job"
                data-test="mobile-generation-job"
              >
                <div class="mobile-generation-job-copy">
                  <p>{{ job.prompt }}</p>
                  <span>{{ job.model }} · {{ job.hostLabel }}</span>
                </div>
                <div class="mobile-generation-job-action">
                  <span data-test="mobile-generation-status">{{ jobStatusCode(job) }}</span>
                  <button
                    class="mobile-generation-cancel"
                    type="button"
                    :aria-label="`Cancel ${job.prompt}`"
                    data-test="mobile-generation-cancel"
                    @click="cancelGeneration(job)"
                  >
                    Cancel
                  </button>
                </div>
              </li>
            </ol>
          </section>
          <div
            class="status-line"
            :class="{ 'error-text': generationStatus.toLowerCase().includes('error') }"
            data-test="mobile-generation-summary"
          >
            {{ generationStatus }}
          </div>
          <div v-if="resultPreviewError" class="result-preview-error" role="alert">
            <p class="status-line error-text">{{ resultPreviewError }}</p>
            <button class="secondary-button" type="button" @click="retryGeneratedPreview">
              Try preview again
            </button>
          </div>
          <video
            v-if="resultUrl && resultIsVideo"
            :key="`${latestResultJob?.clientId}:${resultMediaLoadKey}`"
            class="result-media"
            :src="resultUrl"
            controls
            playsinline
            @play="renewGeneratedResult(false)"
            @loadedmetadata="generatedMediaReady"
            @error="recoverGeneratedMedia"
          />
          <img
            v-else-if="resultUrl"
            :key="`${latestResultJob?.clientId}:${resultMediaLoadKey}`"
            class="result-media"
            :src="resultUrl"
            alt="Generated print"
            @load="generatedMediaReady"
            @error="recoverGeneratedMedia"
          />
        </template>
      </template>

      <template v-else-if="tab === 'gallery'">
        <h1 class="section-title">Gallery</h1>
        <p class="section-note">Prints from every saved host</p>
        <p v-if="galleryError" class="status-line error-text">{{ galleryError }}</p>
        <div v-if="galleryLoading" class="empty-state">Loading prints…</div>
        <div v-else-if="gallery.length" class="gallery-grid">
          <button
            v-for="print in gallery"
            :key="`${print.hostId}:${print.filename}`"
            class="gallery-item"
            type="button"
            :aria-label="`Open ${print.filename} from ${print.hostName}`"
            data-test="gallery-item"
            @click="openPrint(print)"
          >
            <img
              :src="print.thumbnailUrl"
              :alt="print.metadata.prompt || print.filename"
              loading="lazy"
            />
            <span v-if="isVideoItem(print)" class="gallery-video-badge" aria-hidden="true">▶</span>
          </button>
        </div>
        <div v-else class="empty-state">No prints found.</div>
        <button
          v-if="!galleryLoading && galleryRemaining"
          class="secondary-button gallery-more"
          type="button"
          :disabled="galleryLoading || galleryLoadingMore"
          @click="loadMoreGallery"
        >
          {{ galleryLoadingMore ? "Loading…" : `Load older prints (${galleryRemaining})` }}
        </button>
      </template>

      <template v-else-if="tab === 'hosts'">
        <MobileHostDetail
          v-if="hostDetail"
          :host="hostDetail"
          :active="hostDetail.id === selectedHostId"
          @back="hostDetailId = ''"
          @select="selectHost"
          @rename="renameHost"
          @forget="removeHost"
          @catalog="openCatalog"
          @status="updateHostStatus"
        />
        <template v-else>
          <h1 class="section-title">Hosts</h1>
          <p class="section-note">LAN discovery, Tailscale MagicDNS, or an address</p>
          <button
            class="secondary-button"
            type="button"
            :disabled="discovering"
            @click="discoverHosts"
          >
            {{ discovering ? "Scanning…" : "Discover nearby" }}
          </button>
          <div v-for="host in discovered" :key="`${host.host}:${host.port}`" class="host-row">
            <div class="host-row-head">
              <div>
                <div class="host-name">{{ host.name }}</div>
                <div class="host-url">{{ host.host }}:{{ host.port }}</div>
              </div>
              <button
                class="secondary-button"
                type="button"
                @click="connectHost(`${host.host}:${host.port}`, host.name)"
              >
                Connect
              </button>
            </div>
          </div>
          <form style="margin-top: 20px" @submit.prevent="connectHost()">
            <label class="field"
              ><span>Name</span
              ><input
                v-model="hostInput.name"
                class="control"
                placeholder="Studio Mac (optional)"
                autocomplete="off"
            /></label>
            <label class="field"
              ><span>Address or MagicDNS name</span
              ><input
                v-model="hostInput.address"
                class="control"
                placeholder="studio.tailnet.ts.net or 192.168.1.20"
                autocapitalize="none"
                autocomplete="url"
                required
            /></label>
            <label class="field"
              ><span>API key</span
              ><input
                v-model="hostInput.apiKey"
                class="control"
                type="password"
                placeholder="If required"
                autocomplete="off"
            /></label>
            <button class="primary-button" type="submit">Test and save</button>
          </form>
          <p v-if="hostError" class="status-line error-text">{{ hostError }}</p>
          <div v-for="host in hosts" :key="host.id" class="host-row">
            <button
              class="host-row-button"
              type="button"
              :aria-label="`View ${host.name}`"
              data-test="mobile-host-row"
              @click="showHostDetail(host.id)"
            >
              <span class="host-row-head">
                <span>
                  <span class="host-name">{{ host.name }}</span>
                  <span class="host-url">{{ host.baseUrl }}</span>
                </span>
                <span class="host-row-state">
                  <span class="status-dot" :class="host.online ? 'is-ready' : 'is-error'" />
                  <span class="host-chip">{{
                    host.online ? `v${host.version ?? ""}` : "offline"
                  }}</span>
                  <span aria-hidden="true">›</span>
                </span>
              </span>
            </button>
            <div class="row-actions">
              <button
                class="secondary-button"
                type="button"
                :disabled="host.id === selectedHostId"
                @click="selectHost(host.id)"
              >
                {{ host.id === selectedHostId ? "Active" : "Use host" }}
              </button>
            </div>
          </div>
        </template>
      </template>

      <KeepAlive>
        <MobileCatalogView
          v-if="tab === 'catalog'"
          :hosts="hosts"
          :selected-host-id="catalogHostId"
          @select-host="selectCatalogHost"
          @models-changed="catalogModelsChanged"
        />
      </KeepAlive>
    </section>

    <MobileGalleryViewer
      v-if="selectedPrint"
      :item="selectedPrint"
      :target="selectedPrint.target"
      :cache-key="selectedPrint.hostId"
      :host-name="selectedPrint.hostName"
      :thumbnail-url="selectedPrint.thumbnailUrl"
      :reusing="reusingPrint"
      :reuse-error="reusePrintError"
      :generation-announcement="generationAnnouncement"
      :position="selectedPrintIndex + 1"
      :total="gallery.length"
      :has-previous="selectedPrintIndex > 0"
      :has-next="selectedPrintIndex >= 0 && selectedPrintIndex < gallery.length - 1"
      @close="closePrint"
      @reuse="reuseSelectedPrint"
      @previous="navigateSelectedPrint(-1)"
      @next="navigateSelectedPrint(1)"
    />

    <nav class="mobile-tabs" aria-label="Primary">
      <button
        class="mobile-tab"
        type="button"
        :aria-current="tab === 'generate' ? 'page' : undefined"
        data-test="mobile-tab-generate"
        @click="tab = 'generate'"
      >
        Generate
      </button>
      <button
        class="mobile-tab"
        type="button"
        :aria-current="tab === 'gallery' ? 'page' : undefined"
        data-test="mobile-tab-gallery"
        @click="tab = 'gallery'"
      >
        Gallery
      </button>
      <button
        class="mobile-tab"
        type="button"
        :aria-current="tab === 'catalog' ? 'page' : undefined"
        data-test="mobile-tab-catalog"
        @click="openCatalog()"
      >
        Catalog
      </button>
      <button
        class="mobile-tab"
        type="button"
        :aria-current="tab === 'hosts' ? 'page' : undefined"
        data-test="mobile-tab-hosts"
        @click="tab = 'hosts'"
      >
        Hosts
      </button>
    </nav>
  </main>
</template>
