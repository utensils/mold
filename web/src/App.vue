<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";
import DownloadsDrawer from "./components/DownloadsDrawer.vue";
import ResourceTray from "./components/ResourceTray.vue";
import {
  computeEtaSeconds,
  onDownloadComplete,
  useDownloads,
} from "./composables/useDownloads";
import { fetchModels } from "./api";

// Singleton — mounted once, survives navigation.
const downloads = useDownloads();
const drawerOpen = ref(false);

function openDownloads() {
  drawerOpen.value = true;
}
function closeDownloads() {
  drawerOpen.value = false;
}

// Exposed to child components via provide/inject-free singleton from useDownloads.
// We additionally listen for the page-level event "mold:open-downloads" so
// TopBar can open the drawer without a prop drill when it lives inside a page.
function onOpenEvent() {
  openDownloads();
}
window.addEventListener("mold:open-downloads", onOpenEvent);

const off = onDownloadComplete(() => {
  // Best-effort: if the Generate page listens, it can refresh its own models
  // list too. We refresh anyway for the picker.
  void fetchModels().catch(() => undefined);
});

onBeforeUnmount(() => {
  window.removeEventListener("mold:open-downloads", onOpenEvent);
  off();
});

const etaSeconds = computed(() => {
  const a = downloads.active.value;
  if (!a) return null;
  const samples = downloads.ratesByJob.value[a.id] ?? [];
  return computeEtaSeconds(samples, a.bytes_total);
});

async function handleCancel(id: string) {
  await downloads.cancel(id);
}
async function handleRetry(model: string) {
  await downloads.enqueue(model);
}
// ── Agent B (resource telemetry) ────────────────────────────────────────────
// `useResources` is mounted once at the App root and injected into pages
// that need it (/generate, and Agent C's PlacementPanel). This ensures a
// single shared EventSource instead of one per page navigation.
import { provide } from "vue";
import {
  useResources,
  RESOURCES_INJECTION_KEY,
} from "./composables/useResources";

const resources = useResources();
provide(RESOURCES_INJECTION_KEY, resources);
</script>

<template>
  <router-view />
  <ResourceTray />
  <DownloadsDrawer
    :open="drawerOpen"
    :active="downloads.active.value"
    :queued="downloads.queued.value"
    :history="downloads.history.value"
    :eta-seconds="etaSeconds"
    @close="closeDownloads"
    @cancel="handleCancel"
    @retry="handleRetry"
  />
</template>
