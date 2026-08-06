import { computed, ref } from "vue";
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
let timer: ReturnType<typeof setInterval> | null = null;
let consumers = 0;
let refreshing = false;

export function useLiveActivity(routing: HostRouting) {
  async function refresh() {
    if (refreshing) return;
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
  }

  return {
    rows: computed(() => mergeFleetActivity(Object.values(snapshots.value))),
    refresh,
    start() {
      consumers += 1;
      if (timer) return;
      const poll = async () => {
        await refresh();
        if (consumers > 0) timer = setTimeout(poll, POLL_MS);
      };
      timer = setTimeout(poll, 0);
    },
    stop() {
      consumers = Math.max(0, consumers - 1);
      if (consumers || !timer) return;
      clearInterval(timer);
      timer = null;
    },
  };
}
