/*
 * Cross-workspace notifications (spec §08 G11). Three signals fire regardless
 * of which workspace is active:
 *   (a) a generation finishing → a success toast (unless you're already on
 *       Create, where the canvas already shows it) plus a "fresh prints" count
 *       that lights an accent dot on the Gallery nav pill until you visit it;
 *   (b) a model pull finishing or failing → a toast (the Downloads button keeps
 *       its own count badge);
 *   (c) a registered remote host going unreachable → a sticky error toast, once
 *       per offline transition, plus a stop-tinted dot on the Machines pill
 *       while any registered host is offline.
 *
 * The nav (AppNav) reads the two badge signals through `useNotificationSignals`
 * and clears the fresh-prints count with `markGalleryVisited` on entering the
 * Gallery route. `installNotifications` is mounted once from App.vue.
 */
import { computed, ref, watch, type Ref } from "vue";
import { toast } from "./toasts";
import { listStoredHosts } from "./hostRegistry";
import { hostStatus } from "../components/machines/hostClient";
import type { Job } from "../composables/useGenerateStream";
import type { UseDownloads } from "../composables/useDownloads";

const freshPrints = ref(0);
const offlineHostIds = ref<Set<string>>(new Set());

/** Reactive badge signals consumed by the nav. */
export function useNotificationSignals() {
  return {
    freshPrintCount: computed(() => freshPrints.value),
    hasOfflineHost: computed(() => offlineHostIds.value.size > 0),
  };
}

/** Clear the fresh-prints count — called when the Gallery route is entered. */
export function markGalleryVisited(): void {
  freshPrints.value = 0;
}

export interface NotificationDeps {
  /** Live generation jobs (from the App-level useGenerateStream singleton). */
  jobs: Ref<Job[]>;
  /** The App-level downloads singleton. */
  downloads: UseDownloads;
  /** Current route name, read fresh each time a signal fires. */
  currentRouteName: () => string;
  /** Host reachability poll interval (ms). */
  hostPollMs?: number;
}

/**
 * Wire the three notification signals. Returns a teardown that stops every
 * watcher and the host poll.
 */
export function installNotifications(deps: NotificationDeps): () => void {
  const pollMs = deps.hostPollMs ?? 30_000;

  // Pre-seed "already seen" so pre-existing jobs/history (a reconnect snapshot,
  // a prior session) never toast on the first tick — only new terminals do.
  const seenDone = new Set<string>();
  for (const j of deps.jobs.value) if (j.state === "done") seenDone.add(j.id);
  const seenHistory = new Set<string>();
  for (const job of deps.downloads.history.value) seenHistory.add(job.id);

  // (a) A generation finished.
  const stopJobs = watch(
    () =>
      deps.jobs.value
        .filter((j) => j.state === "done")
        .map((j) => j.id)
        .join(","),
    () => {
      const route = deps.currentRouteName();
      for (const j of deps.jobs.value) {
        if (j.state !== "done" || seenDone.has(j.id)) continue;
        seenDone.add(j.id);
        // The Gallery pill nudges you toward the new print unless you're
        // already looking at the gallery.
        if (route !== "gallery") freshPrints.value += 1;
        // Create's own canvas already shows the result — no toast there.
        if (route !== "create")
          toast("success", "generated — saved to gallery");
      }
    },
  );

  // (b) A model pull finished or failed.
  const stopDownloads = watch(
    () =>
      deps.downloads.history.value.map((j) => `${j.id}:${j.status}`).join(","),
    () => {
      for (const job of deps.downloads.history.value) {
        if (seenHistory.has(job.id)) continue;
        seenHistory.add(job.id);
        if (job.status === "completed") {
          toast("success", `installed ${job.model}`);
        } else if (job.status === "failed") {
          toast(
            "error",
            `${job.model} failed to download — retry from downloads`,
          );
        }
        // cancelled → silent; the user asked for it.
      }
    },
  );

  // (c) Remote host reachability.
  const wasOffline = new Set<string>();
  let hostTimer: ReturnType<typeof setInterval> | null = null;

  async function pollHosts(): Promise<void> {
    const hosts = listStoredHosts();
    if (hosts.length === 0) {
      wasOffline.clear();
      if (offlineHostIds.value.size) offlineHostIds.value = new Set();
      return;
    }
    await Promise.all(
      hosts.map(async (host) => {
        try {
          await hostStatus(host);
          wasOffline.delete(host.id);
        } catch {
          if (!wasOffline.has(host.id)) {
            wasOffline.add(host.id);
            toast("error", `can't reach ${host.name} — check machines`, {
              timeout: 0,
            });
          }
        }
      }),
    );
    // Drop ids for hosts that were removed while offline so a stale dot clears.
    const known = new Set(hosts.map((h) => h.id));
    for (const id of [...wasOffline]) if (!known.has(id)) wasOffline.delete(id);
    offlineHostIds.value = new Set(wasOffline);
  }

  void pollHosts();
  hostTimer = setInterval(() => void pollHosts(), pollMs);

  return () => {
    stopJobs();
    stopDownloads();
    if (hostTimer) clearInterval(hostTimer);
    hostTimer = null;
  };
}

/** Test hook — reset the module-level badge signals. */
export function __resetNotificationsForTest(): void {
  freshPrints.value = 0;
  offlineHostIds.value = new Set();
}
