<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import { sseStream } from "../lib/api/sse";
import { outputFormatsForFamily } from "../lib/capabilities";
import type {
  CompleteEvent,
  GalleryImage,
  ModelEntry,
  ProgressEvent,
  ServerStatus,
} from "../lib/api/types";
import {
  applyModelDefaults,
  buildRequest,
  newGenerateForm,
  type GenerateForm,
} from "../lib/generateForm";
import { normalizeRemoteAddress, remoteHostId } from "./hosts";

type Tab = "generate" | "gallery" | "hosts";

interface SavedHost {
  id: string;
  name: string;
  baseUrl: string;
  apiKey: string;
  hostname: string | undefined;
  version: string | undefined;
  online: boolean;
}

interface DiscoveredHost {
  name: string;
  host: string;
  port: number;
}

interface GalleryPrint extends GalleryImage {
  hostId: string;
  hostName: string;
  mediaUrl: string;
}

interface PendingGalleryPrint extends GalleryImage {
  hostId: string;
  hostName: string;
  target: ApiTarget;
}

const STORAGE_KEY = "mold.mobile.hosts.v1";
const SELECTED_KEY = "mold.mobile.selected-host.v1";
const tab = ref<Tab>("generate");
const hosts = ref<SavedHost[]>(loadHosts());
const selectedHostId = ref(localStorage.getItem(SELECTED_KEY) ?? hosts.value[0]?.id ?? "");
const hostInput = reactive({ name: "", address: "", apiKey: "" });
const discovered = ref<DiscoveredHost[]>([]);
const discovering = ref(false);
const hostError = ref("");
const models = ref<ModelEntry[]>([]);
const loadingModels = ref(false);
const form = reactive<GenerateForm>(newGenerateForm());
const generating = ref(false);
const progress = ref("Ready");
const resultUrl = ref("");
const resultFormat = ref("");
const gallery = ref<GalleryPrint[]>([]);
const galleryLoading = ref(false);
const galleryLoadingMore = ref(false);
const galleryError = ref("");
const galleryRemaining = ref(0);
let generationAbort: AbortController | null = null;
const objectUrls = new Set<string>();
let pendingGallery: PendingGalleryPrint[] = [];

const selectedHost = computed(() => hosts.value.find((host) => host.id === selectedHostId.value));
const selectedTarget = computed<ApiTarget | null>(() => {
  const host = selectedHost.value;
  return host ? { baseUrl: host.baseUrl, apiKey: host.apiKey || null } : null;
});
const resultIsVideo = computed(() => resultFormat.value === "mp4");
const outputFormats = computed(() => outputFormatsForFamily(form.family));

function loadHosts(): SavedHost[] {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as SavedHost[];
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
    const id = status.instance_id || remoteHostId(baseUrl);
    const existing = hosts.value.find((host) => host.id === id || host.baseUrl === baseUrl);
    const saved: SavedHost = {
      id,
      name: hostInput.name.trim() || discoveredName || status.hostname || new URL(baseUrl).hostname,
      baseUrl,
      apiKey: hostInput.apiKey.trim(),
      hostname: status.hostname ?? undefined,
      version: status.version,
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

function removeHost(id: string): void {
  hosts.value = hosts.value.filter((host) => host.id !== id);
  if (selectedHostId.value === id) selectedHostId.value = hosts.value[0]?.id ?? "";
  persistHosts();
  void invoke("keychain_delete_api_key", { hostId: id });
}

async function refreshModels(): Promise<void> {
  const target = selectedTarget.value;
  if (!target) return;
  loadingModels.value = true;
  try {
    const [status, entries] = await Promise.all([
      apiJsonTo<ServerStatus>(target, "/api/status"),
      apiJsonTo<ModelEntry[]>(target, "/api/models"),
    ]);
    const host = selectedHost.value;
    if (host) {
      host.online = true;
      host.version = status.version;
      host.hostname = status.hostname ?? undefined;
    }
    models.value = entries.filter((model) => model.downloaded);
    if (!models.value.some((model) => model.name === form.model) && models.value[0]) {
      applyModelDefaults(form, models.value[0]);
    }
  } catch (error) {
    if (selectedHost.value) selectedHost.value.online = false;
    progress.value = error instanceof Error ? error.message : String(error);
  } finally {
    loadingModels.value = false;
  }
}

function changeModel(): void {
  const model = models.value.find((entry) => entry.name === form.model);
  if (model) applyModelDefaults(form, model);
}

function mimeFor(format: string): string {
  return format === "mp4" ? "video/mp4" : format === "jpeg" ? "image/jpeg" : `image/${format}`;
}

function base64Url(data: string, format: string): string {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  const url = URL.createObjectURL(new Blob([bytes], { type: mimeFor(format) }));
  objectUrls.add(url);
  return url;
}

function revokeObjectUrl(url: string): void {
  URL.revokeObjectURL(url);
  objectUrls.delete(url);
}

async function generate(): Promise<void> {
  const target = selectedTarget.value;
  if (!target || !form.prompt.trim() || !form.model) return;
  generationAbort?.abort();
  generationAbort = new AbortController();
  generating.value = true;
  progress.value = "Submitting";
  if (resultUrl.value) revokeObjectUrl(resultUrl.value);
  resultUrl.value = "";
  await sseStream("/api/generate/stream", {
    target,
    method: "POST",
    body: buildRequest(form),
    signal: generationAbort.signal,
    retry: false,
    onOpen: () => (progress.value = "Queued"),
    onEvent: (event, data) => {
      try {
        if (event === "progress") {
          const update = JSON.parse(data) as ProgressEvent;
          if (update.type === "denoise_step")
            progress.value = `Developing ${update.step} / ${update.total}`;
          else if (update.type === "stage_start") progress.value = update.name;
          else if (update.type === "queued") progress.value = `Queued ${update.position + 1}`;
          else if (update.type === "info") progress.value = update.message;
        } else if (event === "complete") {
          const complete = JSON.parse(data) as CompleteEvent;
          resultUrl.value = base64Url(complete.image, complete.format);
          resultFormat.value = complete.format;
          progress.value = `${(complete.generation_time_ms / 1000).toFixed(1)}s · seed ${complete.seed_used}`;
          generating.value = false;
        } else if (event === "error") {
          progress.value = data;
          generating.value = false;
        }
      } catch {
        generationAbort?.abort();
        progress.value = "The host returned an invalid generation update.";
        generating.value = false;
      }
    },
    onClose: (error) => {
      if (error) progress.value = error.message;
      generating.value = false;
    },
  });
}

function stopGeneration(): void {
  generationAbort?.abort();
  generating.value = false;
  progress.value = "Cancelled";
}

async function mediaUrl(target: ApiTarget, filename: string): Promise<string> {
  const response = await apiFetchTo(
    target,
    `/api/gallery/thumbnail/${encodeURIComponent(filename)}`,
  );
  const url = URL.createObjectURL(await response.blob());
  objectUrls.add(url);
  return url;
}

async function refreshGallery(): Promise<void> {
  galleryLoading.value = true;
  galleryError.value = "";
  const prior = gallery.value;
  gallery.value = [];
  for (const item of prior) revokeObjectUrl(item.mediaUrl);
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
  await loadMoreGallery();
  galleryLoading.value = false;
}

async function loadMoreGallery(): Promise<void> {
  galleryLoadingMore.value = true;
  const page = pendingGallery.splice(0, 40);
  for (let offset = 0; offset < page.length; offset += 4) {
    const batch = await Promise.allSettled(
      page.slice(offset, offset + 4).map(async ({ target, ...print }) => ({
        ...print,
        mediaUrl: await mediaUrl(target, print.filename),
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
  if (selectedHostId.value !== print.hostId) {
    selectedHostId.value = print.hostId;
    await refreshModels();
  }
  const meta = print.metadata;
  const model = models.value.find((entry) => entry.name === meta.model);
  if (model) applyModelDefaults(form, model);
  form.prompt = meta.prompt;
  form.negativePrompt = meta.negative_prompt ?? "";
  form.width = meta.generation_width ?? meta.width;
  form.height = meta.generation_height ?? meta.height;
  form.steps = meta.steps;
  form.guidance = meta.guidance;
  form.seed = String(meta.seed);
  tab.value = "generate";
  void nextTick(() => document.querySelector<HTMLTextAreaElement>("#mobile-prompt")?.focus());
}

watch(selectedHostId, (id) => {
  if (id) localStorage.setItem(SELECTED_KEY, id);
  else localStorage.removeItem(SELECTED_KEY);
});

watch(tab, (next) => {
  if (next === "gallery") void refreshGallery();
});

onMounted(async () => {
  await hydrateApiKeys();
  if (selectedHost.value) await refreshModels();
  else tab.value = "hosts";
});

onBeforeUnmount(() => {
  generationAbort?.abort();
  for (const url of objectUrls) URL.revokeObjectURL(url);
});
</script>

<template>
  <main class="mobile-shell">
    <header class="mobile-header">
      <div class="mobile-wordmark">Mold</div>
      <div class="host-chip">{{ selectedHost?.name ?? "Remote only" }}</div>
    </header>

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
          <div class="field-grid">
            <label class="field"
              ><span>Width</span
              ><input v-model.number="form.width" class="control" type="number" inputmode="numeric"
            /></label>
            <label class="field"
              ><span>Height</span
              ><input
                v-model.number="form.height"
                class="control"
                type="number"
                inputmode="numeric"
            /></label>
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
            v-if="!generating"
            class="primary-button"
            type="button"
            :disabled="!form.prompt.trim() || !form.model"
            @click="generate"
          >
            Develop print
          </button>
          <button v-else class="danger-button" type="button" @click="stopGeneration">
            Cancel generation
          </button>
          <div
            class="status-line"
            :class="{ 'error-text': progress.toLowerCase().includes('error') }"
          >
            {{ progress }}
          </div>
          <video
            v-if="resultUrl && resultIsVideo"
            class="result-media"
            :src="resultUrl"
            controls
            playsinline
          />
          <img v-else-if="resultUrl" class="result-media" :src="resultUrl" alt="Generated print" />
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
            :aria-label="`Reuse settings from ${print.filename} on ${print.hostName}`"
            @click="reusePrint(print)"
          >
            <img
              :src="print.mediaUrl"
              :alt="print.metadata.prompt || print.filename"
              loading="lazy"
            />
          </button>
        </div>
        <div v-else class="empty-state">No prints found.</div>
        <button
          v-if="galleryRemaining"
          class="secondary-button gallery-more"
          type="button"
          :disabled="galleryLoadingMore"
          @click="loadMoreGallery"
        >
          {{ galleryLoadingMore ? "Loading…" : `Load older prints (${galleryRemaining})` }}
        </button>
      </template>

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
          <div class="host-row-head">
            <div>
              <div class="host-name">{{ host.name }}</div>
              <div class="host-url">{{ host.baseUrl }}</div>
            </div>
            <span class="host-chip">{{ host.online ? `v${host.version ?? ""}` : "offline" }}</span>
          </div>
          <div class="row-actions">
            <button
              class="secondary-button"
              type="button"
              :disabled="host.id === selectedHostId"
              @click="selectHost(host.id)"
            >
              {{ host.id === selectedHostId ? "Active" : "Use host" }}</button
            ><button class="danger-button" type="button" @click="removeHost(host.id)">
              Remove
            </button>
          </div>
        </div>
      </template>
    </section>

    <nav class="mobile-tabs" aria-label="Primary">
      <button
        class="mobile-tab"
        type="button"
        :aria-selected="tab === 'generate'"
        @click="tab = 'generate'"
      >
        Generate
      </button>
      <button
        class="mobile-tab"
        type="button"
        :aria-selected="tab === 'gallery'"
        @click="tab = 'gallery'"
      >
        Gallery
      </button>
      <button
        class="mobile-tab"
        type="button"
        :aria-selected="tab === 'hosts'"
        @click="tab = 'hosts'"
      >
        Hosts
      </button>
    </nav>
  </main>
</template>
