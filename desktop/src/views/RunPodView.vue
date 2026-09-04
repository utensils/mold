<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import {
  isNetworkVolumeDatacenter,
  podGpuName,
  podHardwareSummary,
  podProxyUrl,
  runPodRegionLabel,
  type RunPodCreateInput,
  type RunPodNetworkVolume,
  type RunPodPod,
} from "../lib/runpod";
import { inTauri } from "../lib/ipc";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useHostsStore } from "../stores/hosts";
import { useRunPodStore } from "../stores/runpod";
import { useToastStore } from "../stores/toasts";
import ConfirmDialog from "../components/shell/ConfirmDialog.vue";
import PodCostMeter from "../components/machines/PodCostMeter.vue";
import ErrorNotice from "@ui/components/ErrorNotice.vue";
import Icon from "@ui/components/Icon.vue";

const runpod = useRunPodStore();
const prefs = useAppPrefsStore();
const hosts = useHostsStore();
const toasts = useToastStore();
const apiKey = ref("");
const savingKey = ref(false);
const confirmingDelete = ref<string | null>(null);
// Confirm before spend: nothing provisions or resumes billing without
// an explicit acknowledgement.
const confirmProvision = ref(false);
const confirmStartPod = ref<RunPodPod | null>(null);
const showVolumeCreate = ref(false);
const editingVolumeId = ref<string | null>(null);
const confirmingVolumeDelete = ref<string | null>(null);
const restoredVolumePreference = ref(false);
const volumeCreate = reactive({ name: "mold-models", sizeGb: 100, datacenterId: "" });
const volumeEdit = reactive({ name: "", sizeGb: 0 });
let poll: ReturnType<typeof setInterval> | null = null;

const form = reactive<RunPodCreateInput>({
  name: null,
  gpuTypeId: "",
  gpuDisplayName: "",
  cloudType: "SECURE",
  datacenterId: null,
  containerDiskGb: 30,
  volumeGb: 80,
  networkVolumeId: null,
  model: null,
  includeHfToken: false,
});

const selectedGpu = computed(() =>
  runpod.gpus.find((gpu) => (gpu.id ?? gpu.gpuId) === form.gpuTypeId),
);

const selectedNetworkVolume = computed(() =>
  runpod.overview.networkVolumes.find((volume) => volume.id === form.networkVolumeId),
);

const datacenters = computed(() => {
  if (selectedNetworkVolume.value) {
    return runpod.overview.datacenters.filter(
      (dc) => dc.id === selectedNetworkVolume.value?.dataCenterId,
    );
  }
  if (!selectedGpu.value) return runpod.overview.datacenters;
  return runpod.overview.datacenters.filter((dc) =>
    dc.gpuAvailability.some(
      (gpu) => gpu.displayName === selectedGpu.value?.displayName && gpu.stockStatus !== "None",
    ),
  );
});

const networkVolumeDatacenters = computed(() =>
  runpod.overview.datacenters.filter((dc) => isNetworkVolumeDatacenter(dc.id)),
);

watch(
  [() => prefs.settings?.runpodNetworkVolumeId, () => runpod.overview.networkVolumes],
  ([saved, volumes]) => {
    if (restoredVolumePreference.value || volumes.length === 0) return;
    form.networkVolumeId = volumes.some((volume) => volume.id === saved) ? (saved ?? null) : null;
    restoredVolumePreference.value = true;
  },
  { immediate: true },
);

watch(selectedNetworkVolume, (volume, previous) => {
  if (volume) {
    form.cloudType = "SECURE";
    form.datacenterId = volume.dataCenterId;
    form.volumeGb = 0;
  } else if (previous && form.volumeGb === 0) {
    form.volumeGb = 80;
  }
});

watch(
  () => form.networkVolumeId,
  (id) => {
    if (restoredVolumePreference.value && prefs.settings && id !== prefs.runpodNetworkVolumeId) {
      void prefs.update({ runpodNetworkVolumeId: id });
    }
  },
);

watch(
  () => runpod.gpus,
  (gpus) => {
    if (form.gpuTypeId || gpus.length === 0) return;
    const first = gpus.find((gpu) => gpu.available) ?? gpus[0]!;
    form.gpuTypeId = first.id ?? first.gpuId ?? "";
    form.gpuDisplayName = first.displayName;
  },
  { immediate: true },
);

watch(selectedGpu, (gpu) => {
  if (gpu) form.gpuDisplayName = gpu.displayName;
  if (form.datacenterId && !datacenters.value.some((dc) => dc.id === form.datacenterId)) {
    form.datacenterId = null;
  }
});

watch(
  () => prefs.settings?.runpodIncludeHfToken,
  (include) => {
    if (include !== undefined) form.includeHfToken = include;
  },
  { immediate: true },
);

watch(
  () => form.includeHfToken,
  (include) => {
    if (prefs.settings && include !== prefs.runpodIncludeHfToken) {
      void prefs.update({ runpodIncludeHfToken: include });
    }
  },
);

const money = (value: number) =>
  new Intl.NumberFormat(undefined, { style: "currency", currency: "USD" }).format(value);

function uptime(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function connectKey() {
  savingKey.value = true;
  try {
    await runpod.saveApiKey(apiKey.value);
    apiKey.value = "";
    toasts.push("RunPod connected");
  } catch (error) {
    toasts.push(errorMessage(error), "error");
  } finally {
    savingKey.value = false;
  }
}

// Provisioning always begins billing, so the button opens a blocking confirm;
// only the confirm actually creates the instance.
function launch() {
  confirmProvision.value = true;
}

async function confirmLaunch() {
  confirmProvision.value = false;
  try {
    await runpod.create({ ...form });
    toasts.push("GPU instance requested");
  } catch {}
}

// Resuming a stopped pod resumes billing — confirm it too, showing the pod's
// known hourly rate.
function requestStart(pod: RunPodPod) {
  confirmStartPod.value = pod;
}

async function confirmStart() {
  const pod = confirmStartPod.value;
  confirmStartPod.value = null;
  if (pod) await act("start", pod);
}

async function createNetworkVolume() {
  try {
    const volume = await runpod.createNetworkVolume({ ...volumeCreate });
    form.networkVolumeId = volume.id;
    showVolumeCreate.value = false;
    toasts.push(`Created ${volume.name}`);
  } catch {}
}

function startVolumeEdit(volume: RunPodNetworkVolume) {
  editingVolumeId.value = volume.id;
  volumeEdit.name = volume.name;
  volumeEdit.sizeGb = volume.size;
}

async function updateNetworkVolume(volume: RunPodNetworkVolume) {
  try {
    await runpod.updateNetworkVolume({
      id: volume.id,
      name: volumeEdit.name === volume.name ? null : volumeEdit.name,
      sizeGb: volumeEdit.sizeGb === volume.size ? null : volumeEdit.sizeGb,
    });
    editingVolumeId.value = null;
    toasts.push(`Updated ${volumeEdit.name}`);
  } catch {}
}

async function deleteNetworkVolume(volume: RunPodNetworkVolume) {
  if (confirmingVolumeDelete.value !== volume.id) {
    confirmingVolumeDelete.value = volume.id;
    return;
  }
  confirmingVolumeDelete.value = null;
  try {
    await runpod.deleteNetworkVolume(volume.id);
    if (form.networkVolumeId === volume.id) form.networkVolumeId = null;
    toasts.push(`Deleted ${volume.name}`);
  } catch {}
}

async function act(action: "start" | "stop" | "delete", pod: RunPodPod) {
  if (action === "delete" && confirmingDelete.value !== pod.id) {
    confirmingDelete.value = pod.id;
    return;
  }
  confirmingDelete.value = null;
  try {
    await runpod.act(action, pod.id);
    toasts.push(
      `${action === "delete" ? "Deleted" : action === "start" ? "Started" : "Stopped"} ${pod.name ?? pod.id}`,
    );
  } catch {}
}

async function useInMold(pod: RunPodPod) {
  try {
    const host = await hosts.connect(podProxyUrl(pod.id), null, pod.name);
    // Send generations to the pod by default; the user can flip back to Auto.
    await prefs.update({ generateTargetHost: host.id });
    toasts.push(`Connected to ${pod.name ?? pod.id}`);
  } catch (error) {
    toasts.push(`The instance is still starting: ${errorMessage(error)}`, "error");
  }
}

async function openConsole() {
  const url = "https://www.runpod.io/console/pods";
  if (inTauri()) {
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    await openUrl(url);
  } else {
    window.open(url, "_blank", "noopener");
  }
}

onMounted(() => {
  void runpod.load();
  poll = setInterval(() => {
    if (runpod.overview.configured && !runpod.mutating) void runpod.load();
  }, 10_000);
});
onBeforeUnmount(() => {
  if (poll) clearInterval(poll);
});
</script>

<template>
  <div class="flex h-full min-h-0 flex-col">
    <div
      class="flex h-[var(--mold-shell-viewbar-h)] shrink-0 items-center gap-2.5 border-b border-border bg-chrome px-3.5"
    >
      <Icon name="runpod" :size="15" class="shrink-0 text-fg-dim" />
      <span class="font-mono text-base font-bold text-fg">runpod cloud</span>
      <span v-if="runpod.overview.account" class="min-w-0 truncate text-xs text-fg-dim">
        {{ runpod.overview.account.email }} ·
        <span class="font-mono">{{ money(runpod.overview.account.spendPerHour) }}/hr</span> active ·
        <span class="font-mono">{{ money(runpod.overview.account.balance) }}</span> balance
      </span>
      <span class="flex-1" />
      <button
        type="button"
        aria-label="Refresh RunPod status"
        class="ms-toolbar-button"
        @click="runpod.load()"
      >
        Refresh
      </button>
      <button type="button" class="ms-toolbar-button" @click="openConsole">RunPod console ↗</button>
    </div>

    <div
      v-if="runpod.loaded && !runpod.overview.configured"
      class="flex flex-1 items-center justify-center p-8"
    >
      <section class="border-border w-full max-w-lg rounded-window border bg-bg p-6">
        <h1 class="font-sans text-lg font-bold text-fg">Add cloud GPUs to your bench</h1>
        <p class="mt-2 max-w-md text-sm text-fg-2">
          Paste a RunPod API key to launch, stop, and connect to mold instances from this app. The
          key is stored only on this device.
        </p>
        <label class="mt-5 block text-micro font-medium text-fg" for="runpod-api-key">
          RunPod API key
        </label>
        <div class="mt-1 flex gap-2">
          <input
            id="runpod-api-key"
            v-model="apiKey"
            data-selectable
            type="password"
            autocomplete="off"
            placeholder="rpa_…"
            class="border-border font-mono text-xs h-9 min-w-0 flex-1 rounded-control border bg-bg-deep px-2.5 text-fg placeholder:text-fg-dim"
            @keydown.enter="connectKey"
          />
          <button
            type="button"
            class="h-9 rounded-control bg-accent px-4 text-sm font-semibold text-on-accent hover:brightness-105 disabled:opacity-50"
            :disabled="savingKey || !apiKey.trim()"
            @click="connectKey"
          >
            {{ savingKey ? "Checking…" : "Connect RunPod" }}
          </button>
        </div>
        <p v-if="runpod.error" class="mt-2 text-micro text-error">{{ runpod.error }}</p>
      </section>
    </div>

    <!-- The house loading treatment: the shape of what is coming, shimmering. -->
    <div
      v-else-if="!runpod.loaded"
      class="grid min-h-0 flex-1 grid-cols-[340px_1fr] overflow-hidden"
      data-test="runpod-loading"
      aria-busy="true"
    >
      <div class="flex flex-col gap-2.5 border-r border-border bg-chrome p-3.5">
        <span v-for="row in 5" :key="row" class="ms-shimmer h-7 rounded-control" />
      </div>
      <div class="flex flex-col gap-2.5 p-3.5">
        <span v-for="row in 3" :key="row" class="ms-shimmer h-16 rounded-control" />
      </div>
    </div>

    <div v-else class="grid min-h-0 flex-1 grid-cols-[340px_1fr] overflow-hidden">
      <aside class="min-h-0 overflow-y-auto border-r border-border bg-chrome p-3.5">
        <div class="flex items-center justify-between">
          <h2 class="text-base font-semibold text-fg">Rent a GPU</h2>
          <span class="font-mono text-micro text-fg-dim">MOLD SERVE</span>
        </div>
        <p class="mt-1 text-xs text-fg-dim">
          Billing starts the moment it boots and stops when you stop it.
        </p>

        <label class="mt-4 block text-micro text-fg-dim" for="runpod-gpu">Graphics card</label>
        <select
          id="runpod-gpu"
          v-model="form.gpuTypeId"
          class="border-border mt-1 h-9 w-full rounded-control border bg-bg-deep px-2 text-sm text-fg"
        >
          <option
            v-for="gpu in runpod.gpus"
            :key="gpu.id ?? gpu.displayName"
            :value="gpu.id ?? gpu.gpuId"
            :disabled="!gpu.available"
          >
            {{ gpu.displayName }} ·
            {{ gpu.memoryInGb == null ? "VRAM unknown" : `${gpu.memoryInGb} GB` }} ·
            {{ gpu.stockStatus ?? "No stock" }}
          </option>
        </select>

        <span class="mt-4 block text-micro text-fg-dim">Kind</span>
        <div
          class="mt-1 flex rounded-control border border-border-control bg-bg-deep p-0.5"
          role="group"
          aria-label="RunPod cloud type"
        >
          <button
            v-for="cloud in ['SECURE', 'COMMUNITY'] as const"
            :key="cloud"
            type="button"
            :disabled="Boolean(selectedNetworkVolume) && cloud === 'COMMUNITY'"
            class="flex-1 rounded-control px-2 py-1.5 text-sm transition-colors"
            :class="
              form.cloudType === cloud
                ? 'bg-accent font-semibold text-on-accent'
                : 'text-fg-2 hover:text-fg disabled:cursor-not-allowed disabled:opacity-35'
            "
            @click="form.cloudType = cloud"
          >
            {{ cloud === "SECURE" ? "Secure" : "Community" }}
          </button>
        </div>

        <label class="mt-4 block text-micro text-fg-dim" for="runpod-dc">Where</label>
        <select
          id="runpod-dc"
          v-model="form.datacenterId"
          :disabled="Boolean(selectedNetworkVolume)"
          class="border-border mt-1 h-9 w-full rounded-control border bg-bg-deep px-2 text-sm text-fg"
        >
          <option :value="null">Automatic (best stock)</option>
          <option v-for="dc in datacenters" :key="dc.id" :value="dc.id">
            {{ runPodRegionLabel(dc.id, dc.location) }}
          </option>
        </select>

        <label class="mt-4 block text-micro text-fg-dim" for="runpod-model"
          >Style to preload (optional)</label
        >
        <input
          id="runpod-model"
          v-model="form.model"
          data-selectable
          type="text"
          placeholder="flux-dev:q8"
          class="border-border font-mono text-xs mt-1 h-9 w-full rounded-control border bg-bg-deep px-2 text-fg placeholder:text-fg-dim"
        />

        <div class="mt-4 grid grid-cols-2 gap-2">
          <label class="text-micro text-fg-dim">
            Container disk
            <span
              class="mt-1 flex h-9 items-center rounded-control border border-border-control bg-bg-deep px-2"
            >
              <input
                v-model.number="form.containerDiskGb"
                data-selectable
                type="number"
                min="10"
                max="1000"
                class="font-mono text-xs min-w-0 flex-1 bg-transparent text-fg outline-none"
              />
              <span>GB</span>
            </span>
          </label>
          <label class="text-micro text-fg-dim">
            Workspace
            <span
              class="mt-1 flex h-9 items-center rounded-control border border-border-control bg-bg-deep px-2"
            >
              <input
                v-model.number="form.volumeGb"
                data-selectable
                type="number"
                min="0"
                max="10000"
                :disabled="Boolean(selectedNetworkVolume)"
                class="font-mono text-xs min-w-0 flex-1 bg-transparent text-fg outline-none"
              />
              <span>GB</span>
            </span>
          </label>
        </div>

        <label class="mt-4 block text-micro text-fg-dim" for="runpod-volume">Keep files on</label>
        <select
          id="runpod-volume"
          v-model="form.networkVolumeId"
          class="border-border mt-1 h-9 w-full rounded-control border bg-bg-deep px-2 text-sm text-fg"
        >
          <option :value="null">None</option>
          <option
            v-for="volume in runpod.overview.networkVolumes"
            :key="volume.id"
            :value="volume.id"
          >
            {{ volume.name }} · {{ volume.size }} GB · {{ volume.dataCenterId }}
          </option>
        </select>
        <p v-if="selectedNetworkVolume" class="mt-1 text-micro text-fg-dim">
          Secure Cloud · pinned to
          {{ runPodRegionLabel(selectedNetworkVolume.dataCenterId) }} · mounted at /workspace
        </p>
        <p
          v-else-if="runpod.overview.networkVolumes.length === 0"
          class="mt-1 text-micro text-fg-dim"
        >
          Create persistent storage in the Network volumes panel.
        </p>

        <label class="mt-4 flex items-start gap-2 text-micro text-fg-2">
          <input v-model="form.includeHfToken" type="checkbox" class="mt-0.5 accent-accent" />
          Pass the saved Hugging Face token for gated model downloads
        </label>

        <button
          type="button"
          class="mt-5 h-9 w-full rounded-control bg-accent text-sm font-semibold text-on-accent hover:brightness-105 active:translate-y-px disabled:opacity-50"
          :disabled="!form.gpuTypeId || runpod.mutating === 'create'"
          @click="launch"
        >
          {{ runpod.mutating === "create" ? "Requesting GPU…" : "Rent a GPU" }}
        </button>
        <p class="mt-2 text-micro text-fg-dim">
          Uses the mold CUDA image matched to the selected GPU and exposes the server on port 7680.
        </p>
        <button
          v-if="runpod.overview.credentialSource === 'app'"
          type="button"
          class="mt-5 text-micro text-fg-dim hover:text-error"
          @click="runpod.disconnect()"
        >
          Remove desktop RunPod key
        </button>
        <p v-else class="mt-5 text-micro text-fg-dim">
          Credential provided by
          {{
            runpod.overview.credentialSource === "environment"
              ? "RUNPOD_API_KEY"
              : "runpod.api_key"
          }}.
        </p>
      </aside>

      <main class="min-h-0 overflow-y-auto p-4">
        <ErrorNotice
          v-if="runpod.operationError"
          compact
          dismissible
          class="mb-4"
          :message="runpod.operationError"
          @dismiss="runpod.clearOperationError()"
        />
        <section class="border-border mb-5 rounded-window border bg-bg-deep">
          <div class="flex items-center justify-between px-3 py-2.5">
            <div>
              <h2 class="text-base font-semibold text-fg">Network volumes</h2>
              <p class="text-micro text-fg-dim">
                Persistent storage that survives instance deletion and remains billable until
                deleted.
              </p>
            </div>
            <button
              type="button"
              class="border-border h-7 rounded-control border px-2.5 text-sm text-fg-2 hover:text-fg"
              @click="showVolumeCreate = !showVolumeCreate"
            >
              {{ showVolumeCreate ? "Cancel" : "New volume" }}
            </button>
          </div>

          <form
            v-if="showVolumeCreate"
            class="border-border grid grid-cols-[1fr_110px_1fr_auto] items-end gap-2 border-t p-3"
            @submit.prevent="createNetworkVolume"
          >
            <label class="text-micro text-fg-dim">
              Name
              <input
                v-model="volumeCreate.name"
                required
                class="border-border mt-1 h-8 w-full rounded-control border bg-bg px-2 text-sm text-fg"
              />
            </label>
            <label class="text-micro text-fg-dim">
              Size (GB)
              <input
                v-model.number="volumeCreate.sizeGb"
                type="number"
                min="10"
                max="3999"
                required
                class="border-border font-mono text-xs mt-1 h-8 w-full rounded-control border bg-bg px-2 text-fg"
              />
            </label>
            <label class="text-micro text-fg-dim">
              Datacenter
              <select
                v-model="volumeCreate.datacenterId"
                required
                class="border-border mt-1 h-8 w-full rounded-control border bg-bg px-2 text-sm text-fg"
              >
                <option value="" disabled>Choose a datacenter</option>
                <option v-for="dc in networkVolumeDatacenters" :key="dc.id" :value="dc.id">
                  {{ runPodRegionLabel(dc.id, dc.location) }}
                </option>
              </select>
            </label>
            <button
              type="submit"
              :disabled="runpod.mutating === 'volume:create'"
              class="h-8 rounded-control bg-accent px-3 text-sm font-semibold text-on-accent disabled:opacity-50"
            >
              {{ runpod.mutating === "volume:create" ? "Creating…" : "Create" }}
            </button>
          </form>

          <p
            v-if="runpod.overview.networkVolumes.length === 0 && !showVolumeCreate"
            class="border-border border-t px-3 py-4 text-micro text-fg-dim"
          >
            No network volumes yet. Create one to keep models and outputs between instances.
          </p>
          <div v-else-if="runpod.overview.networkVolumes.length" class="border-border border-t">
            <article
              v-for="(volume, index) in runpod.overview.networkVolumes"
              :key="volume.id"
              class="px-3 py-2.5"
              :class="index ? 'border-border border-t' : ''"
            >
              <div v-if="editingVolumeId !== volume.id" class="flex items-center gap-3">
                <div class="min-w-0 flex-1">
                  <p class="truncate text-sm font-semibold text-fg">{{ volume.name }}</p>
                  <p class="font-mono text-micro text-fg-dim">
                    {{ volume.size }} GB · {{ runPodRegionLabel(volume.dataCenterId) }} ·
                    {{ volume.id }}
                  </p>
                </div>
                <span
                  v-if="form.networkVolumeId === volume.id"
                  class="font-mono text-micro text-fg-dim whitespace-nowrap text-sapphire"
                >
                  SELECTED
                </span>
                <button
                  type="button"
                  class="border-border h-7 rounded-control border px-2.5 text-sm text-fg-2 hover:text-fg"
                  @click="startVolumeEdit(volume)"
                >
                  Manage
                </button>
              </div>
              <form
                v-else
                class="grid grid-cols-[1fr_110px_auto_auto_auto] items-end gap-2"
                @submit.prevent="updateNetworkVolume(volume)"
              >
                <label class="text-micro text-fg-dim">
                  Name
                  <input
                    v-model="volumeEdit.name"
                    class="border-border mt-1 h-8 w-full rounded-control border bg-bg px-2 text-sm text-fg"
                  />
                </label>
                <label class="text-micro text-fg-dim">
                  Grow to (GB)
                  <input
                    v-model.number="volumeEdit.sizeGb"
                    type="number"
                    :min="volume.size"
                    max="3999"
                    class="border-border font-mono text-xs mt-1 h-8 w-full rounded-control border bg-bg px-2 text-fg"
                  />
                </label>
                <button
                  type="submit"
                  :disabled="
                    (volumeEdit.name === volume.name && volumeEdit.sizeGb === volume.size) ||
                    runpod.mutating === `volume:update:${volume.id}`
                  "
                  class="h-8 rounded-control bg-accent px-3 text-sm font-semibold text-on-accent disabled:opacity-50"
                >
                  Save
                </button>
                <button
                  type="button"
                  class="h-8 px-2 text-sm text-fg-dim hover:text-fg"
                  @click="editingVolumeId = null"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  class="h-8 rounded-control px-2 text-sm"
                  :class="
                    confirmingVolumeDelete === volume.id
                      ? 'bg-error font-semibold text-on-accent'
                      : 'text-error'
                  "
                  @blur="confirmingVolumeDelete = null"
                  @click="deleteNetworkVolume(volume)"
                >
                  {{ confirmingVolumeDelete === volume.id ? "Delete all data?" : "Delete" }}
                </button>
              </form>
            </article>
          </div>
        </section>
        <div class="flex items-baseline justify-between">
          <div>
            <h2 class="text-base font-semibold text-fg">Your instances</h2>
            <p class="text-micro text-fg-dim">Status refreshes every 10 seconds.</p>
          </div>
          <span class="font-mono text-micro text-fg-2"
            >{{ runpod.runningPods.length }} running · {{ runpod.overview.pods.length }} total</span
          >
        </div>

        <div
          v-if="runpod.overview.pods.length === 0"
          class="border-border mt-4 rounded-window border p-8 text-center"
        >
          <p class="text-sm font-medium text-fg">No rented GPUs</p>
          <p class="mt-1 text-micro text-fg-dim">Choose a card and rent your first one.</p>
        </div>

        <div v-else class="border-border mt-4 overflow-hidden rounded-window border">
          <article
            v-for="(pod, index) in runpod.overview.pods"
            :key="pod.id"
            :class="index ? 'border-border border-t' : ''"
            class="bg-bg-deep p-3"
          >
            <div class="flex items-center gap-3">
              <span
                class="h-2 w-2 rounded-full"
                :class="pod.desiredStatus === 'RUNNING' ? 'bg-sapphire' : 'bg-fg-dim'"
                aria-hidden="true"
              />
              <div class="min-w-0 flex-1">
                <div class="flex items-baseline gap-2">
                  <span class="truncate text-sm font-semibold text-fg">{{
                    pod.name ?? pod.id
                  }}</span>
                  <span
                    class="font-mono text-micro text-fg-dim whitespace-nowrap"
                    :class="pod.desiredStatus === 'RUNNING' ? 'text-sapphire' : ''"
                    >{{ pod.desiredStatus || "UNKNOWN" }}</span
                  >
                </div>
                <p class="font-mono truncate text-micro text-fg-dim">
                  {{ podGpuName(pod) }} · {{ pod.machine?.location ?? "Placement pending" }} ·
                  <span class="text-state-cost">{{ money(pod.costPerHr) }}/hr</span
                  ><span v-if="pod.uptimeSeconds"> · {{ uptime(pod.uptimeSeconds) }}</span>
                </p>
                <p v-if="podHardwareSummary(pod)" class="font-mono truncate text-micro text-fg-dim">
                  {{ podHardwareSummary(pod) }}
                </p>
                <p v-if="pod.networkVolume" class="font-mono truncate text-micro text-fg-dim">
                  {{ pod.networkVolume.name }} · {{ pod.networkVolume.size }} GB network volume
                </p>
                <p v-if="pod.networkVolume" class="mt-1 text-micro text-fg-dim">
                  Delete the instance to stop compute; /workspace remains on the volume.
                </p>
              </div>
              <PodCostMeter
                v-if="pod.desiredStatus === 'RUNNING'"
                :cost-per-hr="pod.costPerHr"
                :uptime-seconds="pod.uptimeSeconds"
              />
              <button
                v-if="pod.desiredStatus === 'RUNNING'"
                type="button"
                class="h-7 rounded-control bg-accent px-3 text-sm font-semibold text-on-accent hover:brightness-105"
                @click="useInMold(pod)"
              >
                Use in mold
              </button>
              <button
                v-if="!pod.networkVolume && pod.desiredStatus === 'RUNNING'"
                type="button"
                class="border-border h-7 rounded-control border px-2.5 text-sm text-fg-2 hover:text-fg disabled:opacity-50"
                :disabled="runpod.mutating === `stop:${pod.id}`"
                @click="act('stop', pod)"
              >
                Stop
              </button>
              <button
                v-else-if="!pod.networkVolume"
                type="button"
                class="border-border h-7 rounded-control border px-2.5 text-sm text-fg-2 hover:text-fg disabled:opacity-50"
                :disabled="runpod.mutating === `start:${pod.id}`"
                @click="requestStart(pod)"
              >
                Start
              </button>
              <button
                type="button"
                class="border-border h-7 rounded-control border px-2.5 text-sm text-fg-2 hover:text-fg"
                @click="openConsole"
              >
                RunPod console ↗
              </button>
              <button
                type="button"
                class="h-7 rounded-control px-2 text-sm"
                :class="
                  confirmingDelete === pod.id
                    ? 'bg-error font-semibold text-on-accent'
                    : 'text-fg-dim hover:text-error'
                "
                @blur="confirmingDelete = null"
                @click="act('delete', pod)"
              >
                {{ confirmingDelete === pod.id ? "Delete instance?" : "Delete" }}
              </button>
            </div>
          </article>
        </div>
      </main>
    </div>

    <ConfirmDialog
      :open="confirmProvision"
      title="Rent this GPU?"
      confirm-label="Start it — billing begins now"
      danger
      :busy="runpod.mutating === 'create'"
      @confirm="confirmLaunch"
      @cancel="confirmProvision = false"
    >
      <div class="font-mono space-y-1 text-micro text-fg-2">
        <div>
          {{ form.gpuDisplayName || "GPU" }} ·
          {{ form.cloudType === "SECURE" ? "Secure" : "Community" }} cloud
        </div>
        <div>
          Container disk {{ form.containerDiskGb }} GB<span v-if="selectedNetworkVolume">
            · network volume {{ selectedNetworkVolume.name }} ({{
              selectedNetworkVolume.size
            }}
            GB)</span
          ><span v-else-if="form.volumeGb"> · workspace {{ form.volumeGb }} GB</span>
        </div>
        <div class="text-fg-dim">RunPod bills this GPU by the minute while it runs.</div>
      </div>
    </ConfirmDialog>

    <ConfirmDialog
      :open="!!confirmStartPod"
      title="Start it again?"
      :confirm-label="
        confirmStartPod ? `Resume · ${money(confirmStartPod.costPerHr)}/hr` : 'Resume'
      "
      message="Billing resumes immediately."
      danger
      :busy="!!confirmStartPod && runpod.mutating === `start:${confirmStartPod.id}`"
      @confirm="confirmStart"
      @cancel="confirmStartPod = null"
    />
  </div>
</template>
