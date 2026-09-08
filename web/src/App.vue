<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, provide, ref } from "vue";
import { useRoute } from "vue-router";
import ToastShelf from "@ui/components/ToastShelf.vue";
import DownloadsPopover from "./components/shell/DownloadsPopover.vue";
import AppNav from "./components/shell/AppNav.vue";
import CommandK from "./components/shell/CommandK.vue";
import ConfirmDialog from "./components/shell/ConfirmDialog.vue";
import LicenseAcceptanceDialog from "@studio/components/LicenseAcceptanceDialog.vue";
import { dismissToast, runToastAction, useNotifications } from "./lib/toasts";
import {
  computeEtaSeconds,
  computeRateBytesPerSec,
  onDownloadComplete,
  useDownloads,
} from "./composables/useDownloads";
import { useCatalog } from "./composables/useCatalog";
import { useGenerateStream } from "./composables/useGenerateStream";
import { startGenerateQueueReconciler } from "./composables/useQueueReconciler";
import { useHostRouting } from "./composables/useHostRouting";
import { useActivityRows } from "./composables/useActivityRows";
import { useQueueSections } from "./composables/useQueueSections";
import { useLiveActivity } from "./composables/useLiveActivity";
import { installNotifications } from "./lib/notifications";
import {
  useResources,
  RESOURCES_INJECTION_KEY,
} from "./composables/useResources";

const route = useRoute();

// Singleton — mounted once, survives navigation.
const downloads = useDownloads();

// Queue reconciliation (L3): poll `/api/queue` and dead-letter any
// running card whose server-side registry entry is gone. Catches the
// edge cases the per-job SSE error path can't see (server restarted
// mid-generation, browser tab suspended past keepalive, etc.). Mounted
// once at the App root so the loop runs regardless of which page the
// user is on.
const stream = useGenerateStream();
const reconciler = startGenerateQueueReconciler(stream);

// One fleet activity loop belongs to the shell. Page and navigation consumers
// read the same singleton, including while Create is unmounted.
const routing = useHostRouting();
const liveActivity = useLiveActivity(routing);
const activityRows = useActivityRows(stream.jobs, liveActivity.rows);
const queueSections = useQueueSections(
  activityRows.localActivityJobs,
  activityRows.sharedActivityRows,
  routing.queueStatus,
  routing.hosts,
);
const liveSummary = computed(() =>
  queueSections.value
    .filter((section) => section.count > 0)
    .map((section) => `${section.count} ${section.label.toLowerCase()}`)
    .join(" · "),
);

onMounted(() => liveActivity.start());

// Downloads popover (spec §06). Opened by the AppNav button and ⌘K palette,
// both of which dispatch the shared `mold:open-downloads` window event.
const downloadsOpen = ref(false);
function onOpenEvent() {
  downloadsOpen.value = true;
}
window.addEventListener("mold:open-downloads", onOpenEvent);

// ⌘K / Ctrl+K toggles the command palette from anywhere (spec §06). Esc close
// is handled inside PalettePanel.
const paletteOpen = ref(false);
function onKeydown(event: KeyboardEvent) {
  if (
    (event.metaKey || event.ctrlKey) &&
    (event.key === "k" || event.key === "K")
  ) {
    event.preventDefault();
    paletteOpen.value = !paletteOpen.value;
  }
}
window.addEventListener("keydown", onKeydown);

const off = onDownloadComplete(() => {
  // Refresh the shared Installed shelf once a pull lands. Merely fetching and
  // discarding `/api/models` leaves the singleton's rendered rows stale.
  void useCatalog().refreshInstalled();
});

// Cross-workspace notifications (spec §08 G11): generation-done / pull /
// host-offline toasts plus the nav badge signals.
const teardownNotifications = installNotifications({
  jobs: stream.jobs,
  downloads,
  currentRouteName: () => String(route.name ?? ""),
});

onBeforeUnmount(() => {
  window.removeEventListener("mold:open-downloads", onOpenEvent);
  window.removeEventListener("keydown", onKeydown);
  off();
  reconciler.stop();
  liveActivity.stop();
  teardownNotifications();
});

const etaByJob = computed(() =>
  Object.fromEntries(
    downloads.activeJobs.value.map((job) => [
      job.id,
      computeEtaSeconds(
        downloads.ratesByJob.value[job.id] ?? [],
        job.bytes_total,
      ),
    ]),
  ),
);

const rateByJob = computed(() =>
  Object.fromEntries(
    downloads.activeJobs.value.map((job) => [
      job.id,
      computeRateBytesPerSec(downloads.ratesByJob.value[job.id] ?? []),
    ]),
  ),
);

async function handleCancel(id: string) {
  await downloads.cancel(id);
}
async function handleRetry(model: string) {
  await downloads.enqueue(model);
}

// `useResources` is mounted once at the App root and provided so pages that
// need GPU telemetry (Advanced → PlacementPanel) share a single EventSource
// instead of opening one per navigation.
const resources = useResources();
provide(RESOURCES_INJECTION_KEY, resources);

// App-frame notifications (spec §08 G11/G12): the shelf and confirm modal
// render inside this frame — never a page-level fixed layer.
const notifications = useNotifications();
</script>

<template>
  <div class="app-frame">
    <AppNav />
    <div
      v-if="liveSummary && route.path !== '/queue'"
      class="global-live-work"
      data-test="global-live-work"
    >
      <router-link to="/queue"
        ><span aria-live="polite">{{ liveSummary }}</span
        ><span>View Queue →</span></router-link
      >
    </div>
    <router-view />
    <DownloadsPopover
      :open="downloadsOpen"
      :active="downloads.activeJobs.value"
      :queued="downloads.queued.value"
      :history="downloads.history.value"
      :eta-by-job="etaByJob"
      :rate-by-job="rateByJob"
      @close="downloadsOpen = false"
      @cancel="handleCancel"
      @retry="handleRetry"
    />
    <ToastShelf
      :toasts="notifications.toasts"
      @dismiss="dismissToast"
      @action="runToastAction"
    />
    <ConfirmDialog />
    <LicenseAcceptanceDialog />
    <CommandK :open="paletteOpen" @close="paletteOpen = false" />
  </div>
</template>

<style scoped>
/* The app frame is the positioning context every overlay renders inside
 * (spec §05 — contained, not global). */
.app-frame {
  position: relative;
  min-height: 100svh;
  display: flex;
  flex-direction: column;
}
</style>

<style scoped>
.global-live-work {
  padding: 8px 24px;
  background: var(--mold-bg-deep);
  border-bottom: 1px solid var(--mold-border);
}
.global-live-work a {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 8px 24px;
  max-width: 1072px;
  margin: auto;
  min-height: 32px;
  font-size: 0.875rem;
  color: var(--mold-text);
  text-decoration: none;
}
.global-live-work a > span:last-child {
  color: var(--mold-blue);
}
@media (max-width: 639px) {
  .global-live-work {
    padding-inline: 16px;
  }
  .global-live-work a {
    min-height: 44px;
  }
}
</style>
