import { defineStore } from "pinia";
import { ApiError } from "../lib/api/client";
import { fetchConfig, fetchProfiles, resetConfig, setConfig, setProfile } from "../lib/api/config";
import {
  parsePerStyleKey,
  schemaFor,
  schemasForSection,
  sectionForConfigKey,
  type SectionId,
} from "../lib/settingsSchema";
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
    /**
     * Every row a section renders, so a body names its keys once. Curated keys
     * come out in the order the schema declares them — which is the order the
     * section reads in — and anything this build has never heard of follows,
     * sorted by key, rather than disappearing.
     */
    rowsForSection(state) {
      return (id: SectionId): ConfigRow[] => {
        const order = new Map(
          schemasForSection(id).map((schema, index) => [schema.key, index]),
        );
        return state.rows
          .filter((row) => sectionForConfigKey(row.key) === id)
          .sort((a, b) => {
            const ai = order.get(a.key) ?? order.size;
            const bi = order.get(b.key) ?? order.size;
            return ai - bi || a.key.localeCompare(b.key);
          });
      };
    },
    /** `models.<style>.<field>` rows, in the order the engine reported them —
     *  `groupPerStyleRows` is what turns them into one row per style. */
    perStyleRows(state): ConfigRow[] {
      return state.rows.filter((row) => parsePerStyleKey(row.key) !== null);
    },
    /**
     * The raw engine keys each section actually draws, so the shell's search
     * can match a section on them. Per-style overrides are raw rows too, and
     * they belong to Per-style defaults — searching a style name must find the
     * section it is in, not the Advanced list it left.
     */
    rawKeysBySection(): Partial<Record<SectionId, string[]>> {
      return {
        advanced: this.advancedRows
          .filter((row) => !schemaFor(row.key))
          .map((row) => row.key),
        styleDefaults: this.perStyleRows.map((row) => row.key),
      };
    },
  },
  actions: {
    async load() {
      try {
        this.rows = await fetchConfig();
        this.available = true;
      } catch (err) {
        // Only a 404 means the engine predates the config API. Transient
        // network/auth failures keep the previous knowledge (or stay
        // undetermined) so settings aren't hidden by a blip.
        if (err instanceof ApiError && err.status === 404) this.available = false;
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
