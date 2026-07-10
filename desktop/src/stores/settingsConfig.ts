import { defineStore } from "pinia";
import { ApiError } from "../lib/api/client";
import { fetchConfig, fetchProfiles, resetConfig, setConfig, setProfile } from "../lib/api/config";
import { sectionForConfigKey } from "../lib/settingsSchema";
import type { ConfigRow } from "../lib/api/types";

/**
 * Engine configuration (`/api/config`) for the Settings surface. Works
 * identically against the embedded engine and remote hosts; `available`
 * goes false against engines that predate the config API.
 */
export const useSettingsConfigStore = defineStore("settingsConfig", {
  state: () => ({
    rows: [] as ConfigRow[],
    available: null as boolean | null,
    profiles: [] as string[],
    activeProfile: "default",
  }),
  getters: {
    byKey(state): Map<string, ConfigRow> {
      return new Map(state.rows.map((r) => [r.key, r]));
    },
    /** Rows without a curated editor — rendered raw in Advanced. */
    advancedRows(state): ConfigRow[] {
      return state.rows
        .filter((r) => sectionForConfigKey(r.key) === "advanced")
        .sort((a, b) => a.key.localeCompare(b.key));
    },
  },
  actions: {
    async load() {
      try {
        this.rows = await fetchConfig();
        this.available = true;
      } catch (err) {
        this.available = false;
        void err;
      }
      try {
        const p = await fetchProfiles();
        this.profiles = p.profiles;
        this.activeProfile = p.active;
      } catch {
        /* profiles need the metadata DB; General still renders */
      }
    },
    row(key: string): ConfigRow | null {
      return this.byKey.get(key) ?? null;
    },
    async save(key: string, value: ConfigRow["value"]): Promise<string | null> {
      try {
        await setConfig(key, value);
        await this.load();
        return null;
      } catch (err) {
        return err instanceof ApiError ? err.message : String(err);
      }
    },
    async reset(key: string): Promise<string | null> {
      try {
        await resetConfig(key);
        await this.load();
        return null;
      } catch (err) {
        return err instanceof ApiError ? err.message : String(err);
      }
    },
    /** Switching to an unknown name creates the profile (server semantics). */
    async switchProfile(name: string): Promise<string | null> {
      try {
        await setProfile(name);
        await this.load();
        return null;
      } catch (err) {
        return err instanceof ApiError ? err.message : String(err);
      }
    },
  },
});
