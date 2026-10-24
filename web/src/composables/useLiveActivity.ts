import { computed, ref, watch, type WatchStopHandle } from "vue";
import {
  listActiveWork,
  mergeFleetActivity,
  reconcileActivityHost,
  type ActivityHostSnapshot,
} from "@studio/api/activity";
import type { HostRouting } from "./useHostRouting";

const STORAGE_KEY = "mold.web.live-activity.v1";
const POLL_MS = 5_000;

function load(): Record<string, ActivityHostSnapshot> {
  try {
    const saved = JSON.parse(
      localStorage.getItem(STORAGE_KEY) ?? "{}",
    ) as Record<string, ActivityHostSnapshot>;
    for (const snapshot of Object.values(saved)) {
      snapshot.routeUrl ??= snapshot.target?.baseUrl ?? "";
      snapshot.instanceId ??= null;
      snapshot.unavailableKinds ??= [];
      snapshot.stale = true;
      snapshot.error ??= "Waiting to reconnect";
    }
    return saved;
  } catch {
    return {};
  }
}

const snapshots = ref(load());
const epochs: Record<string, number> = {};
let timer: ReturnType<typeof setTimeout> | null = null;
let consumers = 0;
let refreshing = false;
let refreshAgain = false;
let pollEpoch = 0;
let stopReadiness: WatchStopHandle | null = null;

export function useLiveActivity(routing: HostRouting) {
  let started = false;
  async function refresh(afterCurrent = false) {
    if (refreshing) {
      // A host became ready mid-read: read once more, after this one.
      if (afterCurrent) refreshAgain = true;
      return;
    }
    refreshing = true;
    const routes = routing.hosts.value;
    try {
      await Promise.all(
        routes.map(async (host) => {
          const route = {
            hostId: host.id,
            hostLabel: host.label,
            target: {
              baseUrl: host.url,
              apiKey: host.apiKey ?? null,
              connectionMode: host.connectionMode,
            },
          };
          const epoch = (epochs[host.id] ?? 0) + 1;
          epochs[host.id] = epoch;
          const previous = snapshots.value[host.id];
          let result;
          try {
            if (host.status !== "ready") throw new Error("Host is offline");
            result = await listActiveWork(route.target);
          } catch (error) {
            result = error instanceof Error ? error : new Error(String(error));
          }
          if (epochs[host.id] !== epoch) return;
          snapshots.value = {
            ...snapshots.value,
            [host.id]: reconcileActivityHost(route, previous, result),
          };
        }),
      );
      const configured = new Set(routing.hosts.value.map((host) => host.id));
      snapshots.value = Object.fromEntries(
        Object.entries(snapshots.value).filter(([id]) => configured.has(id)),
      );
      const safe = Object.fromEntries(
        Object.entries(snapshots.value).map(
          ([id, { target: _target, ...snapshot }]) => [id, snapshot],
        ),
      );
      localStorage.setItem(STORAGE_KEY, JSON.stringify(safe));
    } finally {
      refreshing = false;
    }
    if (refreshAgain) {
      refreshAgain = false;
      await refresh();
    }
  }

  return {
    rows: computed(() => mergeFleetActivity(Object.values(snapshots.value))),
    refresh,
    start() {
      if (started) return;
      started = true;
      consumers += 1;
      if (consumers > 1) return;
      const epoch = ++pollEpoch;
      const poll = async () => {
        await refresh();
        if (consumers > 0 && epoch === pollEpoch)
          timer = setTimeout(poll, POLL_MS);
      };
      timer = setTimeout(poll, 0);
      // The shell's first poll runs before any readiness probe has answered,
      // so every host reads "offline" and the Queue showed nothing for a whole
      // poll while the saved-video list (which waits for readiness) did.
      // Read a host the moment it becomes ready.
      const readyKeys = () =>
        routing.hosts.value
          .filter((host) => host.status === "ready")
          .map((host) => `${host.id}|${host.url}`);
      // Already-ready hosts are covered by the first poll above.
      let ready = new Set(readyKeys());
      stopReadiness = watch(readyKeys, (now) => {
        const fresh = now.some((key) => !ready.has(key));
        ready = new Set(now);
        if (fresh && consumers > 0 && epoch === pollEpoch) void refresh(true);
      });
    },
    stop() {
      if (!started) return;
      started = false;
      consumers = Math.max(0, consumers - 1);
      if (consumers) return;
      pollEpoch += 1;
      stopReadiness?.();
      stopReadiness = null;
      if (timer !== null) clearTimeout(timer);
      timer = null;
    },
  };
}
