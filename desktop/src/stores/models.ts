import { defineStore } from "pinia";
import { apiJson } from "../lib/api/client";
import type { ModelEntry } from "../lib/api/types";

/** Families that never appear in the generation model picker. */
const NON_GENERATION_FAMILIES = new Set(["real-esrgan", "upscaler", "qwen3-expand", "controlnet"]);

export function isGenerationModel(m: ModelEntry): boolean {
  return !NON_GENERATION_FAMILIES.has(m.family);
}

export const useModelStore = defineStore("models", {
  state: () => ({
    all: [] as ModelEntry[],
    loading: false,
    error: null as string | null,
  }),
  getters: {
    installed: (s) => s.all.filter((m) => m.downloaded && isGenerationModel(m)),
    /** Still-image upscalers for the post-generate select (any download state —
     *  the server auto-pulls on first use). */
    upscalers: (s) => s.all.filter((m) => m.family === "upscaler" || m.family === "real-esrgan"),
    byFamily(): Map<string, ModelEntry[]> {
      const groups = new Map<string, ModelEntry[]>();
      for (const m of this.installed) {
        const list = groups.get(m.family) ?? [];
        list.push(m);
        groups.set(m.family, list);
      }
      return groups;
    },
  },
  actions: {
    async fetch() {
      this.loading = true;
      this.error = null;
      try {
        this.all = await apiJson<ModelEntry[]>("/api/models");
      } catch (err) {
        this.error = String(err);
      } finally {
        this.loading = false;
      }
    },
  },
});
