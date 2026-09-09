import { defineStore } from "pinia";
import { apiJson } from "../lib/api/client";
import { isGenerationModel as isGenerationModelShared } from "@studio/lib/generationModels";
import type { ModelEntry } from "../lib/api/types";

/**
 * Whether a row is a style a person can pick.
 *
 * The list lives in `@studio/lib/generationModels` — this surface and web each
 * had their own and they had already drifted (this one knew `real-esrgan`,
 * web knew `companion` and `control-net`), which is how the 3-D Studio came to
 * offer the prompt expander as a picture style.
 */
export function isGenerationModel(m: ModelEntry): boolean {
  return isGenerationModelShared(m);
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
