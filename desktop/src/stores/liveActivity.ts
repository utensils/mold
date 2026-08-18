import { defineStore } from "pinia";
import {
  listActiveWork,
  mergeFleetActivity,
  reconcileActivityHost,
  type ActivityHostSnapshot,
} from "@studio/api/activity";
import { useHostsStore } from "./hosts";

const STORAGE_KEY = "mold.desktop.live-activity.v1";
const POLL_MS = 5_000;

function loadSnapshots(): Record<string, ActivityHostSnapshot> {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}") as Record<
      string,
      ActivityHostSnapshot
    >;
    for (const snapshot of Object.values(value)) {
      snapshot.routeUrl ??= snapshot.target?.baseUrl ?? "";
      snapshot.instanceId ??= null;
      snapshot.unavailableKinds ??= [];
      snapshot.stale = true;
      snapshot.error ??= "Waiting to reconnect";
    }
    return value;
  } catch {
    return {};
  }
}

export const useLiveActivityStore = defineStore("liveActivity", {
  state: () => ({
    hosts: loadSnapshots(),
    epochs: {} as Record<string, number>,
    timer: null as ReturnType<typeof setInterval> | null,
    refreshing: false,
    refreshQueued: false,
    refreshPromise: null as Promise<void> | null,
    consumers: 0,
  }),
  getters: {
    rows: (state) => mergeFleetActivity(Object.values(state.hosts)),
  },
  actions: {
    persist() {
      if (typeof localStorage === "undefined") return;
      const safe = Object.fromEntries(
        Object.entries(this.hosts).map(([id, { target: _target, ...snapshot }]) => [id, snapshot]),
      );
      localStorage.setItem(STORAGE_KEY, JSON.stringify(safe));
    },
    async refresh() {
      if (this.refreshPromise) {
        this.refreshQueued = true;
        return this.refreshPromise;
      }
      this.refreshing = true;
      const run = async () => {
        do {
          this.refreshQueued = false;
          await this.refreshOnce();
        } while (this.refreshQueued);
      };
      this.refreshPromise = run();
      try {
        await this.refreshPromise;
      } finally {
        this.refreshPromise = null;
        this.refreshing = false;
      }
    },
    async refreshOnce() {
      const hosts = useHostsStore();
      await Promise.all(
        hosts.all.map(async (host) => {
          if (!host.baseUrl) return;
          const route = {
            hostId: host.id,
            hostLabel: host.label,
            target: { baseUrl: host.baseUrl, apiKey: host.apiKey },
          };
          const epoch = (this.epochs[host.id] ?? 0) + 1;
          this.epochs[host.id] = epoch;
          const previous = this.hosts[host.id];
          let result;
          try {
            if (host.status !== "ready") throw new Error("Host is offline");
            result = await listActiveWork(route.target);
          } catch (error) {
            result = error instanceof Error ? error : new Error(String(error));
          }
          if (this.epochs[host.id] !== epoch) return;
          this.hosts[host.id] = reconcileActivityHost(route, previous, result);
        }),
      );
      const configured = new Set(hosts.all.map((host) => host.id));
      for (const id of Object.keys(this.hosts)) {
        if (!configured.has(id)) delete this.hosts[id];
      }
      this.persist();
    },
    start() {
      this.consumers += 1;
      if (this.timer) return;
      const poll = async () => {
        await this.refresh();
        if (this.consumers > 0) this.timer = setTimeout(poll, POLL_MS);
      };
      this.timer = setTimeout(poll, 0);
    },
    stop() {
      this.consumers = Math.max(0, this.consumers - 1);
      if (this.consumers || !this.timer) return;
      clearInterval(this.timer);
      this.timer = null;
    },
  },
});
