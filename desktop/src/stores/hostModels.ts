import { defineStore } from "pinia";
import { filterRestrictedModels } from "@studio/lib/modelAccess";
import { apiJsonTo } from "../lib/api/client";
import type { ModelEntry } from "../lib/api/types";
import { mergeModelPresentationMetadata } from "../lib/models";
import { isGenerationModel, useModelStore } from "./models";
import { useHostsStore } from "./hosts";

/** A per-host `/api/models` result younger than this is not refetched. */
const STALE_MS = 60_000;

interface HostModelList {
  entries: ModelEntry[];
  /** Unix ms of the last successful fetch (0 = never). */
  fetchedAt: number;
  error: string | null;
}

export type UnionModelEntry = ModelEntry & { hostIds: string[] };

/**
 * Per-host model availability for routing and the union model picker. The
 * primary host's canonical list stays in `useModelStore`; this store answers
 * "which hosts already have model X" across every live connection. Fetches
 * are demand-driven (picker open, ready-set changes) — no global timers.
 */
export const useHostModelsStore = defineStore("hostModels", {
  state: () => ({
    byHost: {} as Record<string, HostModelList>,
    loading: false,
  }),
  getters: {
    modelsOn(state) {
      return (hostId: string): ModelEntry[] =>
        filterRestrictedModels(
          state.byHost[hostId]?.entries ?? [],
          useHostsStore().capabilities[hostId],
        );
    },
    /** Downloaded generation models on one host — mirrors `useModelStore.installed`. */
    installedOn(state) {
      return (hostId: string): ModelEntry[] =>
        filterRestrictedModels(
          state.byHost[hostId]?.entries ?? [],
          useHostsStore().capabilities[hostId],
        ).filter((m) => m.downloaded && isGenerationModel(m));
    },
    /**
     * Every installed generation model across all hosts, deduped by name in
     * `hosts.all` order (the first host's entry wins, so the primary's
     * defaults are preferred), with the hosts that have it collected.
     */
    unionInstalled(): UnionModelEntry[] {
      const hosts = useHostsStore();
      const byName = new Map<string, UnionModelEntry>();
      for (const host of hosts.all) {
        for (const m of this.installedOn(host.id)) {
          const existing = byName.get(m.name);
          if (existing) {
            byName.set(m.name, {
              ...mergeModelPresentationMetadata(existing, m),
              hostIds: [...new Set([...existing.hostIds, host.id])],
            });
          } else {
            byName.set(m.name, { ...m, hostIds: [host.id] });
          }
        }
      }
      return [...byName.values()];
    },
    /** Host ids that have `name` installed; [] when unknown or nowhere. */
    hostsFor(): (name: string) => string[] {
      return (name) => this.unionInstalled.find((m) => m.name === name)?.hostIds ?? [];
    },
    /**
     * Resolve the authoritative model row for Create's active host policy.
     * An explicit machine owns runtime defaults and its generation profile;
     * only Auto/Most capable may use the fleet union's preferred row.
     */
    installedEntryForTarget(): (name: string, targetHostId: string | null) => ModelEntry | null {
      return (name, targetHostId) => {
        if (!name) return null;
        const primaryEntry = filterRestrictedModels(
          useModelStore().installed,
          useHostsStore().capabilities.local,
        ).find((model) => model.name === name);
        if (!targetHostId || targetHostId === "capable") {
          return primaryEntry ?? this.unionInstalled.find((model) => model.name === name) ?? null;
        }

        // The primary model store is populated by this exact local server and
        // can become ready before the all-host fanout.
        if (targetHostId === "local" && primaryEntry) {
          return primaryEntry;
        }
        return this.installedOn(targetHostId).find((model) => model.name === name) ?? null;
      };
    },
    allReadyHostsFetched(state): boolean {
      const hosts = useHostsStore();
      return hosts.all
        .filter((host) => host.status === "ready")
        .every((host) => (state.byHost[host.id]?.fetchedAt ?? 0) > 0);
    },
  },
  actions: {
    /**
     * Fan out `/api/models` to every ready host. Hosts fetched under 60s ago
     * are skipped unless `force`; a host that fails keeps its last entries
     * (stale beats empty for availability tags) and records the error.
     */
    async refresh(force = false) {
      const hosts = useHostsStore();
      const now = Date.now();
      const pending = hosts.all
        .filter((h) => h.status === "ready" && h.baseUrl)
        .filter((h) => force || now - (this.byHost[h.id]?.fetchedAt ?? 0) >= STALE_MS)
        .map(async (h) => {
          try {
            const entries = await apiJsonTo<ModelEntry[]>(
              { baseUrl: h.baseUrl!, apiKey: h.apiKey },
              "/api/models",
            );
            this.byHost[h.id] = { entries, fetchedAt: Date.now(), error: null };
          } catch (err) {
            const prev = this.byHost[h.id];
            this.byHost[h.id] = {
              entries: prev?.entries ?? [],
              // Keep the old fetchedAt so the next refresh retries.
              fetchedAt: prev?.fetchedAt ?? 0,
              error: String(err),
            };
          }
        });
      if (pending.length === 0) return;
      this.loading = true;
      try {
        await Promise.all(pending);
      } finally {
        this.loading = false;
      }
    },
  },
});
