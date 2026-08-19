/*
 * Cross-workspace notifications (spec §08 G11). Three signals fire regardless
 * of which workspace is active:
 *   (a) a generation finishing → a success toast (unless you're already on
 *       Create, where the canvas already shows it) plus a "fresh prints" count
 *       that lights an accent dot on the Gallery nav pill until you visit it;
 *   (b) a model pull finishing or failing → a toast (the Downloads button keeps
 *       its own count badge);
 *   (c) a registered remote host going unreachable → a sticky WARNING toast,
 *       once per offline transition, plus a stop-tinted dot on the Machines
 *       pill while any registered host is offline. The poll keeps retrying, so
 *       the machine reconnects on its own; when it answers again the warning is
 *       withdrawn and a green "Reconnected to …" toast confirms it.
 *
 * The nav (AppNav) reads the two badge signals through `useNotificationSignals`
 * and clears the fresh-prints count with `markGalleryVisited` on entering the
 * Gallery route. `installNotifications` is mounted once from App.vue.
 */
import { computed, ref, watch, type Ref } from "vue";
import { dismissToast, toast } from "./toasts";
import {
  HOST_OFFLINE_DESCRIPTION,
  hostOfflineTitle,
  hostReconnectedTitle,
  reconcileHostConnectivity,
} from "@studio/lib/hostConnectivity";
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

  // (c) Remote host reachability. Every listed host is re-probed on the timer,
  // so reconnection is automatic — these toasts only narrate the two edges, and
  // the edge policy itself is the shared reconciler desktop reads too.
  /** hostId → the sticky offline toast, withdrawn once the host answers. */
  const offlineToastIds = new Map<string, string>();
  /** Last settled reachability per host, carried between polls. */
  let reachability: Record<string, string> = {};
  /** Rising counter so a slow probe from an earlier tick cannot land late. */
  let pollEpoch = 0;
  let hostTimer: ReturnType<typeof setInterval> | null = null;

  function retireOfflineToast(id: string): void {
    const toastId = offlineToastIds.get(id);
    if (toastId) dismissToast(toastId);
    offlineToastIds.delete(id);
  }

  /** Abort a probe that outlives its own poll interval rather than letting it
   *  settle after a later tick already reported the opposite. */
  function pollSignal(): AbortSignal | undefined {
    if (typeof AbortSignal?.timeout === "function") {
      return AbortSignal.timeout(pollMs);
    }
    const controller = new AbortController();
    setTimeout(() => controller.abort(), pollMs);
    return controller.signal;
  }

  async function pollHosts(): Promise<void> {
    const epoch = ++pollEpoch;
    const hosts = listStoredHosts();
    if (hosts.length === 0) {
      for (const id of [...offlineToastIds.keys()]) retireOfflineToast(id);
      reachability = {};
      if (offlineHostIds.value.size) offlineHostIds.value = new Set();
      return;
    }
    const signal = pollSignal();
    const current = await Promise.all(
      hosts.map(async (host) => {
        try {
          await hostStatus(host, signal);
          return { id: host.id, label: host.name, status: "ready" };
        } catch {
          return { id: host.id, label: host.name, status: "error" };
        }
      }),
    );
    // A tick that started later has already reported; discard this one whole
    // rather than flapping the user between "reconnected" and "can't reach".
    if (epoch !== pollEpoch) return;
    const changes = reconcileHostConnectivity({
      previous: reachability,
      current,
      warned: offlineToastIds.keys(),
      // Unlike desktop, a registered machine that does not answer on the very
      // first probe is news here: the browser has no Machines row already
      // showing it as errored at that moment.
      warnOnFirstContact: true,
    });
    for (const host of changes.offline) {
      offlineToastIds.set(
        host.id,
        toast("warning", hostOfflineTitle(host.label), {
          description: HOST_OFFLINE_DESCRIPTION,
          timeout: 0,
        }),
      );
    }
    for (const host of changes.reconnected) {
      retireOfflineToast(host.id);
      toast("success", hostReconnectedTitle(host.label));
    }
    // Drop notices for hosts removed while offline so a stale dot clears.
    for (const id of changes.retired) retireOfflineToast(id);
    reachability = changes.next;
    offlineHostIds.value = new Set(offlineToastIds.keys());
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
