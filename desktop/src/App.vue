<script setup lang="ts">
import { onMounted, onUnmounted, watch } from "vue";
import { useRouter } from "vue-router";
import TitleBar from "./components/shell/TitleBar.vue";
import NavRail from "./components/shell/NavRail.vue";
import Toasts from "./components/shell/Toasts.vue";
import CommandPalette from "./components/shell/CommandPalette.vue";
import ContextMenu from "./components/shell/ContextMenu.vue";
import UpdateBanner from "./components/shell/UpdateBanner.vue";
import { dockBadgeValue } from "./lib/dockBadge";
import { ipc } from "./lib/ipc";
import { appIsBackground } from "./lib/notify";
import {
  detectOfflineTransitions,
  newlyCompletedJobs,
  shouldToastGenerationComplete,
  snapshotHostStatuses,
} from "./lib/notifications";
import {
  allowsNativeContextMenu,
  allowsNativeSelectAll,
  isSelectAllChord,
  resolveShellShortcut,
} from "./lib/shortcuts";
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

// Cross-surface notifications (§08 G11). A generation finishing while the user
// is somewhere other than Create raises a toast that jumps to Library; the
// native notification (dispatched by the generation store) covers the
// backgrounded case, so the foreground toast bows out then.
const notifiedComplete = new Set<number>();
watch(
  () => generation.jobs.map((j) => `${j.clientId}:${j.status}`).join("|"),
  () => {
    const done = newlyCompletedJobs(generation.jobs, notifiedComplete);
    for (const job of done) notifiedComplete.add(job.clientId);
    // Bound the seen-set to live jobs so it can't grow across a long session.
    for (const id of [...notifiedComplete]) {
      if (!generation.jobs.some((j) => j.clientId === id)) notifiedComplete.delete(id);
    }
    if (done.length === 0) return;
    if (!shouldToastGenerationComplete(router.currentRoute.value.path)) return;
    if (appIsBackground()) return;
    toasts.push("Generated — saved to Library", "info", {
      onClick: () => void router.push("/library"),
    });
  },
);

// A connected host dropping offline (ready → error) raises a sticky error toast
// once per transition, regardless of the active workspace.
let hostStatusSnapshot: Record<string, string> = {};
watch(
  () => hostsStore.all.map((h) => `${h.id}:${h.status}`).join("|"),
  () => {
    const current = hostsStore.all.map((h) => ({ id: h.id, label: h.label, status: h.status }));
    for (const host of detectOfflineTransitions(hostStatusSnapshot, current)) {
      toasts.push(`Can't reach ${host.label} — check Machines.`, "error", { sticky: true });
    }
    hostStatusSnapshot = snapshotHostStatuses(current);
  },
  { immediate: true },
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
      void appPrefs.update({ sidebarCollapsed: !appPrefs.sidebarCollapsed });
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
      void router.push("/create");
      break;
    case "randomize-seed":
      if (route === "/create") ui.randomizeSeed();
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
      case "new-sequence":
        return void router.push({ path: "/create", query: { output: "sequence" } });
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
        void appPrefs.update({ sidebarCollapsed: !appPrefs.sidebarCollapsed });
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
 * Replace WebKit's default context menu app-wide. Only text-editing fields
 * keep the native menu (spellcheck, paste) — `allowsNativeContextMenu`
 * excludes non-text inputs like range sliders, which previously leaked the
 * webview's Back / Reload / Inspect Element menu. Components open the custom
 * menu by calling useContextMenuStore().open() in their own contextmenu
 * handlers; devtools stay reachable via View → Developer Tools.
 */
function suppressNativeContextMenu(e: Event) {
  if (!allowsNativeContextMenu(e.target as Element | null)) e.preventDefault();
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
  // Remembered remotes are independent of This Mac. Start both boot paths
  // together so a slow local engine can never postpone host reconnects.
  const hostStartup = hostsStore.init();
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
  // Neither failure blocks launch: host errors remain visible in the sidebar,
  // while the local connection store owns its own error presentation.
  await Promise.allSettled([connectionStartup, hostStartup]);
});
onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown);
  window.removeEventListener("contextmenu", suppressNativeContextMenu);
  window.removeEventListener("selectstart", suppressChromeSelection);
});
</script>

<template>
  <div class="relative flex h-full flex-col overflow-hidden">
    <TitleBar class="h-11 shrink-0" />
    <UpdateBanner />
    <div class="grid min-h-0 flex-1 grid-cols-[auto_1fr]">
      <NavRail />
      <main class="min-h-0 min-w-0 overflow-hidden">
        <router-view />
      </main>
    </div>
    <Toasts />
    <CommandPalette />
    <ContextMenu />
  </div>
</template>
