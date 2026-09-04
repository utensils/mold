import { describe, expect, it } from "vitest";
import { defineComponent, type ComputedRef } from "vue";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useShellSubtitle } from "./useShellSubtitle";
import { useGenerateFormStore } from "../stores/generateForm";

const stub = { template: "<div />" };

async function subtitleAt(path: string): Promise<ComputedRef<string>> {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: ["/create", "/queue", "/library", "/models", "/machines", "/settings"].map((p) => ({
      path: p,
      component: stub,
    })),
  });
  router.push(path);
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  let subtitle: ComputedRef<string> | null = null;
  mount(
    defineComponent({
      setup() {
        subtitle = useShellSubtitle();
        return () => null;
      },
    }),
    { global: { plugins: [pinia, router] } },
  );
  return subtitle!;
}

describe("useShellSubtitle", () => {
  it("names all three of Create's output kinds, 3-D included", async () => {
    const subtitle = await subtitleAt("/create");
    const draft = useSequenceDraftStore();
    const form = useGenerateFormStore();
    expect(subtitle.value).toBe("Still picture · 0 waiting");

    form.form.family = "hunyuan3d";
    expect(subtitle.value).toBe("3-D object · 0 waiting");

    // A clip is the authored output kind, so it outranks the style's family.
    draft.output = "sequence";
    expect(subtitle.value).toBe("Short clip · 0 waiting");
  });
});
