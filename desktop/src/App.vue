<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import TitleBar from "./components/shell/TitleBar.vue";
import NavRail from "./components/shell/NavRail.vue";
import BenchRail from "./components/shell/BenchRail.vue";
import Toasts from "./components/shell/Toasts.vue";
import CommandPalette from "./components/shell/CommandPalette.vue";
import ContextMenu from "./components/shell/ContextMenu.vue";
import UpdateBanner from "./components/shell/UpdateBanner.vue";
import { dockBadgeValue } from "./lib/dockBadge";
import { ipc } from "./lib/ipc";
import { allowsNativeSelectAll, isSelectAllChord, resolveShellShortcut } from "./lib/shortcuts";
import { useAppPrefsStore } from "./stores/appPrefs";
import { useConnectionStore } from "./stores/connection";
import { useContextMenuStore } from "./stores/contextMenu";
import { useEventsStore } from "./stores/events";
import { useHostsStore } from "./stores/hosts";
import { useGenerationStore } from "./stores/generation";
import { useToastStore } from "./stores/toasts";
import { useUiStore } from "./stores/ui";
import { useUpdaterStore } from "./stores/updater";

const router = useRouter();
const sidebarOpen = ref(true);
const appPrefs = useAppPrefsStore();
const connection = useConnectionStore();
const contextMenu = useContextMenuStore();
const events = useEventsStore();
const hostsStore = useHostsStore();

// App-wide server-event subscription (live gallery). Re-probe whenever the
// engine target changes — a different host may not support /api/events.
watch(
  () => [connection.ready, connection.baseUrl] as const,
  ([ready]) => {
    if (ready) void events.resubscribe();
    else events.unsubscribe();
  },
);
const generation = useGenerationStore();
const toasts = useToastStore();
const ui = useUiStore();
const updater = useUpdaterStore();

// Dock badge mirrors THIS app's active jobs, event-driven (no poll lag) and
// cleared the moment the last job settles.
watch(
  () => [generation.pending.length, appPrefs.dockBadge] as const,
  ([pending, enabled]) => void ipc.setDockBadge(dockBadgeValue(pending, enabled)),
);

function onKeydown(e: KeyboardEvent) {
  // WebKit honors ⌘A even under `user-select: none`, painting the whole app
  // chrome as selected. Editable fields keep their native in-field Select All.
  if (isSelectAllChord(e) && !allowsNativeSelectAll(document.activeElement)) {
    e.preventDefault();
    return;
  }
  const action = resolveShellShortcut(e);
  if (!action) return;
  e.preventDefault();
  const route = router.currentRoute.value.path;
  switch (action.kind) {
    case "navigate":
      void router.push(action.route);
      break;
    case "toggle-sidebar":
      sidebarOpen.value = !sidebarOpen.value;
      break;
    case "command-palette":
      ui.togglePalette();
      break;
    case "cancel-job": {
      const job = generation.active;
      if (job && job.status !== "complete" && job.status !== "error") {
        void generation.cancel(job.clientId).then(() => toasts.push("Cancelled"));
      }
      break;
    }
    case "new-generation":
      ui.newGeneration();
      void router.push("/generate");
      break;
    case "randomize-seed":
      if (route === "/generate") ui.randomizeSeed();
      break;
    case "copy-seed":
      ui.copySeed();
      break;
    case "ui-scale":
      contextMenu.close();
      void appPrefs.scaleUi(action.direction);
      break;
  }
}

/** Native menu items reuse the same actions as the keyboard map. */
async function listenForMenu() {
  if (!("__TAURI_INTERNALS__" in window)) return;
  const { listen } = await import("@tauri-apps/api/event");
  await listen<string>("menu", ({ payload: id }) => {
    if (id.startsWith("nav:")) return void router.push(id.slice(4));
    switch (id) {
      case "settings":
        return void router.push("/settings");
      case "check-for-updates":
        void router.push({ path: "/settings", query: { section: "updates" } });
        return void updater.check();
      case "new-generation":
        ui.newGeneration();
        return void router.push("/generate");
      case "new-chain":
        return void router.push("/chains");
      case "generate":
        return ui.generate();
      case "expand-prompt":
        return ui.expandPrompt();
      case "randomize-seed":
        return ui.randomizeSeed();
      case "cancel-job":
        if (generation.active) void generation.cancel().then(() => toasts.push("Cancelled"));
        return;
      case "toggle-sidebar":
        sidebarOpen.value = !sidebarOpen.value;
        return;
      case "zoom-in":
        contextMenu.close();
        return void appPrefs.scaleUi("in");
      case "zoom-out":
        contextMenu.close();
        return void appPrefs.scaleUi("out");
      case "actual-size":
        contextMenu.close();
        return void appPrefs.scaleUi("reset");
      case "help:api":
        if (connection.baseUrl) {
          void import("@tauri-apps/plugin-opener").then(({ openUrl }) =>
            openUrl(`${connection.baseUrl}/api/docs`),
          );
        }
        return;
    }
  });
}

/**
 * Replace WebKit's default context menu app-wide. Editable fields keep the
 * native menu (spellcheck, paste); components open the custom menu by
 * calling useContextMenuStore().open() in their own contextmenu handlers,
 * which stops propagation before this suppressor runs.
 */
function suppressNativeContextMenu(e: Event) {
  const target = e.target as HTMLElement | null;
  if (target?.closest("input, textarea, [contenteditable='true']")) return;
  e.preventDefault();
}

/**
 * Belt-and-braces companion to `user-select: none`: WebKit can still start a
 * selection on chrome (e.g. a drag that began in a selectable region), so
 * refuse selection at the source unless the target opted in.
 */
function suppressChromeSelection(e: Event) {
  if (!allowsNativeSelectAll(e.target as Element | null)) e.preventDefault();
}

onMounted(async () => {
  window.addEventListener("keydown", onKeydown);
  window.addEventListener("contextmenu", suppressNativeContextMenu);
  window.addEventListener("selectstart", suppressChromeSelection);
  // Prefs first: theme lands before the window is shown, and restore-last-view
  // navigates before the default route paints.
  const prefs = await appPrefs.init().catch(() => null);
  if (prefs?.restoreLastRoute && prefs.lastRoute && prefs.lastRoute !== "/") {
    await router.replace(prefs.lastRoute).catch(() => {});
  }
  router.afterEach((to) => void appPrefs.rememberRoute(to.path));
  // Check in the background after preferences select the correct channel.
  void updater.init();
  const connectionStartup = connection.init();
  void listenForMenu();
  // The window starts hidden (tauri.conf.json visible:false) to avoid a
  // white flash; reveal it once the shell has mounted. No-op in a browser.
  if ("__TAURI_INTERNALS__" in window) {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    const appWindow = getCurrentWindow();
    await appWindow.maximize();
    await appWindow.show();
  }
  await connectionStartup.catch(() => {});
  // Extra hosts reconnect after the primary connection settles; failures
  // surface as sidebar rows + a toast, never as a blocked launch.
  void hostsStore.init();
});
onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown);
  window.removeEventListener("contextmenu", suppressNativeContextMenu);
  window.removeEventListener("selectstart", suppressChromeSelection);
});
</script>

<template>
  <div class="flex h-full flex-col">
    <TitleBar class="h-11 shrink-0" />
    <UpdateBanner />
    <div class="grid min-h-0 flex-1 grid-cols-[auto_1fr]">
      <NavRail v-if="sidebarOpen" />
      <main class="min-h-0 min-w-0 overflow-hidden">
        <router-view />
      </main>
    </div>
    <BenchRail class="h-7 shrink-0" />
    <Toasts />
    <CommandPalette />
    <ContextMenu />
  </div>
</template>
