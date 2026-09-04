<script setup lang="ts">
import { onMounted, onUnmounted, watch } from "vue";
import { useRouter } from "vue-router";
import TitleBar from "./components/shell/TitleBar.vue";
import Sidebar from "./components/shell/Sidebar.vue";
import StatusBar from "./components/shell/StatusBar.vue";
import Toasts from "./components/shell/Toasts.vue";
import CommandPalette from "./components/shell/CommandPalette.vue";
import ContextMenu from "./components/shell/ContextMenu.vue";
import UpdateBanner from "./components/shell/UpdateBanner.vue";
import LicenseAcceptanceDialog from "@studio/components/LicenseAcceptanceDialog.vue";
import { openExternal } from "./lib/openExternal";
import { dockBadgeValue } from "./lib/dockBadge";
import { ipc } from "./lib/ipc";
import { notificationRoute, type NotificationAction } from "./lib/notificationAction";
import { appIsBackground } from "./lib/notify";
import {
  HOST_OFFLINE_DESCRIPTION,
  applyHostConnectivity,
  hostOfflineTitle,
  hostReconnectedTitle,
  newlyCompletedJobs,
  shouldToastGenerationComplete,
} from "./lib/notifications";
import {
  allowsNativeContextMenu,
  allowsNativeSelectAll,
  isSelectAllChord,
  overlayOwnsKeyboard,
  resolveBareShellShortcut,
  resolveShellShortcut,
} from "./lib/shortcuts";
import { useQueueCommands } from "./composables/useQueueCommands";
import { useAppPrefsStore } from "./stores/appPrefs";
import { useConnectionStore } from "./stores/connection";
import { useContextMenuStore } from "./stores/contextMenu";
import { useEventsStore } from "./stores/events";
import { useHostsStore } from "./stores/hosts";
import { useHostStatusStore } from "./stores/hostStatus";
import { useGenerationStore } from "./stores/generation";
import { useLibraryPrefsStore } from "./stores/libraryPrefs";
import { useToastStore } from "./stores/toasts";
import { useUiStore } from "./stores/ui";
import { useUpdaterStore } from "./stores/updater";

const router = useRouter();
const appPrefs = useAppPrefsStore();
const connection = useConnectionStore();
const contextMenu = useContextMenuStore();
const events = useEventsStore();
const hostsStore = useHostsStore();
const hostStatus = useHostStatusStore();
const libraryPrefs = useLibraryPrefsStore();

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
const queueCommands = useQueueCommands();

// The machine card and the status bar read one telemetry authority. Its
// status poll follows the primary connection (the embedded-engine recovery
// invariant) while the resources stream follows the display host.
watch(
  () => connection.ready,
  (ready) => (ready ? hostStatus.start() : hostStatus.stop()),
);
watch(
  () => `${hostStatus.displayHost?.id ?? "none"}:${hostStatus.connection}`,
  () => {
    if (connection.ready) hostStatus.startResourceStream();
  },
);

function openNotificationAction(action: NotificationAction | null) {
  if (action) void router.push(notificationRoute(action));
}

let unlistenNotificationAction: (() => void) | null = null;

async function listenForNotificationActions() {
  if (!("__TAURI_INTERNALS__" in window)) return;
  const { listen } = await import("@tauri-apps/api/event");
  unlistenNotificationAction = await listen<NotificationAction>(
    "notification-action",
    ({ payload }) => {
      // The native side retains the action in case activation races startup.
      // Once the live listener receives it, consume that fallback so a later
      // launch cannot replay a notification the user already opened.
      void ipc.takeNotificationAction().catch(() => null);
      openNotificationAction(payload);
    },
  );
  openNotificationAction(await ipc.takeNotificationAction().catch(() => null));
}

// Dock badge mirrors THIS app's active jobs, event-driven (no poll lag) and
// cleared the moment the last job settles.
watch(
  () => [generation.pending.length, appPrefs.dockBadge] as const,
  ([pending, enabled]) => void ipc.setDockBadge(dockBadgeValue(pending, enabled)),
);

// Cross-surface notifications. A generation finishing while the user
// is somewhere other than Create raises a toast that jumps to My images; the
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
    toasts.push("Generated — saved to My images", "info", {
      onClick: () => void router.push("/library"),
    });
  },
);

// Host reachability, once per edge, regardless of the active workspace. The
// status poll keeps probing every listed host, so an unreachable machine
// reconnects on its own: dropping offline is a WARNING (sticky, because the
// condition persists), and coming back withdraws it and confirms in green.
// The policy and the id bookkeeping live in applyHostConnectivity; this only
// supplies the shelf effects.
let hostStatusSnapshot: Record<string, string> = {};
/** hostId → the sticky offline toast, withdrawn the moment the host answers. */
const offlineToastIds = new Map<string, number>();
watch(
  () => hostsStore.all.map((h) => `${h.id}:${h.status}`).join("|"),
  () => {
    const current = hostsStore.all.map((h) => ({ id: h.id, label: h.label, status: h.status }));
    for (const host of hostsStore.all) {
      if (host.status === "ready") generation.ensureDurableHostStream(host.id);
    }
    void generation.reconcileDurableAll();
    hostStatusSnapshot = applyHostConnectivity(hostStatusSnapshot, current, offlineToastIds, {
      warn: (host) =>
        toasts.push(hostOfflineTitle(host.label), "warning", {
          description: HOST_OFFLINE_DESCRIPTION,
          action: {
            label: "Open Machines",
            run: () => void router.push("/machines"),
          },
          sticky: true,
        }),
      announceRecovery: (host) => toasts.push(hostReconnectedTitle(host.label), "success"),
      dismiss: (toastId) => toasts.dismiss(toastId),
    });
  },
  { immediate: true },
);

function onKeydown(e: KeyboardEvent) {
  // WebKit honors ⌘A even under `user-select: none`, painting the whole app
  // chrome as selected. Editable fields keep their native in-field Select All.
  if (isSelectAllChord(e) && !allowsNativeSelectAll(document.activeElement)) {
    e.preventDefault();
    // Library owns a real Select All operation. Dispatch from the shell so it
    // remains reliable even when WebKit delivers the native command to this
    // long-lived listener before the route view's listener.
    if (router.currentRoute.value.path === "/library") {
      window.dispatchEvent(new CustomEvent("mold:library-select-all"));
    }
    return;
  }
  const route = router.currentRoute.value.path;
  const action =
    resolveShellShortcut(e) ??
    resolveBareShellShortcut(e, {
      target: document.activeElement,
      overlayOpen: ui.paletteOpen || contextMenu.visible || overlayOwnsKeyboard(),
      route,
    });
  if (!action) return;
  e.preventDefault();
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
        void generation
          .cancel(job.clientId)
          .then((cancelled) => {
            if (cancelled) toasts.push("Cancelled");
          })
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
          );
      }
      break;
    }
    case "new-generation":
      ui.newGeneration();
      void router.push("/create");
      break;
    case "make-variations":
      ui.makeVariations();
      if (route !== "/create") void router.push("/create");
      break;
    case "randomize-seed":
      if (route === "/create") ui.randomizeSeed();
      break;
    case "copy-seed":
      ui.copySeed();
      break;
    case "toggle-queue-pause":
      void queueCommands.togglePause();
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
        return void router.push("/create");
      case "new-sequence":
        return void router.push({ path: "/create", query: { output: "sequence" } });
      case "generate":
        return ui.generate();
      case "expand-prompt":
        return ui.expandPrompt();
      case "randomize-seed":
        return ui.randomizeSeed();
      case "cancel-job":
        if (generation.active)
          void generation
            .cancel()
            .then((cancelled) => {
              if (cancelled) toasts.push("Cancelled");
            })
            .catch((error) =>
              toasts.push(error instanceof Error ? error.message : String(error), "error"),
            );
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

function reconcileDurableOnWake() {
  if (document.visibilityState === "visible") void generation.reconcileDurableAll();
}

onMounted(async () => {
  // Synchronous and first: the Create form's auto-tag mirror has to be right
  // before any request can be built from it, whichever workspace opens.
  libraryPrefs.init();
  window.addEventListener("keydown", onKeydown);
  window.addEventListener("contextmenu", suppressNativeContextMenu);
  window.addEventListener("selectstart", suppressChromeSelection);
  window.addEventListener("focus", reconcileDurableOnWake);
  document.addEventListener("visibilitychange", reconcileDurableOnWake);
  // Prefs first: theme lands before the window is shown, and restore-last-view
  // navigates before the default route paints.
  const prefs = await appPrefs.init().catch(() => null);
  if (prefs?.restoreLastRoute && prefs.lastRoute && prefs.lastRoute !== "/") {
    await router.replace(prefs.lastRoute).catch(() => {});
  }
  router.afterEach((to) => void appPrefs.rememberRoute(to.path));
  // Check in the background after preferences select the correct channel.
  void updater.init();
  // Connected remotes are independent of This Mac. Start both boot paths
  // together so a slow local engine can never postpone host reconnects.
  const hostStartup = hostsStore.init();
  const connectionStartup = connection.init();
  void listenForMenu();
  void listenForNotificationActions();
  // The window starts hidden (tauri.conf.json visible:false) to avoid a
  // white flash; reveal it once the shell has mounted. No-op in a browser.
  if ("__TAURI_INTERNALS__" in window) {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    const appWindow = getCurrentWindow();
    await appWindow.maximize();
    await appWindow.show();
  }
  // Neither failure blocks launch: host errors remain visible in Machines,
  // while the local connection store owns its own error presentation.
  await Promise.allSettled([connectionStartup, hostStartup]);
  generation.resumeDurableGenerations();
});
onUnmounted(() => {
  window.removeEventListener("keydown", onKeydown);
  window.removeEventListener("contextmenu", suppressNativeContextMenu);
  window.removeEventListener("selectstart", suppressChromeSelection);
  window.removeEventListener("focus", reconcileDurableOnWake);
  document.removeEventListener("visibilitychange", reconcileDurableOnWake);
  unlistenNotificationAction?.();
  hostStatus.stop();
});
</script>

<template>
  <div class="relative flex h-full flex-col overflow-hidden bg-bg">
    <TitleBar />
    <UpdateBanner />
    <div class="flex min-h-0 flex-1 overflow-hidden">
      <Sidebar />
      <main class="min-h-0 min-w-0 flex-1 overflow-hidden">
        <router-view />
      </main>
    </div>
    <StatusBar />
    <Toasts />
    <CommandPalette />
    <ContextMenu />
    <LicenseAcceptanceDialog :open-external="openExternal" />
  </div>
</template>
