import { describe, expect, it } from "vitest";
import { defineComponent, type ComputedRef } from "vue";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useShellSubtitle } from "./useShellSubtitle";
import { useConnectionStore } from "../stores/connection";
import { useGalleryStore } from "../stores/gallery";
import { useGenerateFormStore } from "../stores/generateForm";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import type { GalleryImage } from "../lib/api/types";

const stub = { template: "<div />" };

async function subtitleAt(path: string): Promise<ComputedRef<string>> {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      ...["/create", "/queue", "/library", "/models", "/settings"].map((p) => ({
        path: p,
        component: stub,
      })),
      {
        path: "/machines",
        component: stub,
        children: [{ path: "runpod", name: "runpod", component: stub }],
      },
    ],
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

const print = (filename: string): GalleryImage =>
  ({ filename, metadata: {}, timestamp: 1, size_bytes: 10 }) as GalleryImage;

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

/**
 * Three `[data-test='library-count']` assertions were retired on the grounds
 * that "the shell's title bar already says it" — so the title bar is where it
 * has to be proved. One case per branch of the sentence.
 */
describe("useShellSubtitle — one live sentence per workspace", () => {
  it("counts My images in pictures and albums", async () => {
    const subtitle = await subtitleAt("/library");
    const gallery = useGalleryStore();
    expect(subtitle.value).toBe("0 pictures · 0 albums");

    gallery.buckets["local"] = {
      items: [print("a.png"), print("b.png")],
      loading: false,
      error: null,
      loaded: true,
    };
    gallery.collectionsByHost["local"] = {
      items: [{ id: "col-1", name: "Holidays", slug: "holidays", count: 1 } as never],
      loading: false,
      error: null,
      loaded: true,
    };
    expect(subtitle.value).toBe("2 pictures · 1 album");
  });

  it("splits the queue into what is waiting and what is being made", async () => {
    expect((await subtitleAt("/queue")).value).toBe("0 waiting · 0 being made");
  });

  it("counts the styles that are ready", async () => {
    const subtitle = await subtitleAt("/models");
    useModelStore().all = [
      { name: "sdxl-base:fp16", family: "sdxl", downloaded: true } as never,
      { name: "flux-dev:q4", family: "flux", downloaded: true } as never,
    ];
    expect(subtitle.value).toBe("2 styles ready");
  });

  /**
   * The Ready-to-use badge counts the whole fleet. This sentence counted the
   * primary alone, so a connected second machine put "106" beside "25 styles
   * ready" and neither string said which set it meant.
   */
  it("counts the same fleet the Ready-to-use badge counts, and names the scope", async () => {
    const subtitle = await subtitleAt("/models");
    useModelStore().all = [{ name: "flux-dev:q4", family: "flux", downloaded: true } as never];
    const hosts = useHostsStore();
    hosts.extras = [
      {
        id: "plato",
        label: "plato",
        url: "http://plato:7680",
        apiKey: null,
        status: "ready",
        error: null,
        instanceId: "plato-uuid",
      },
    ];
    const hostModels = useHostModelsStore();
    hostModels.byHost = {
      plato: {
        entries: [
          { name: "flux-dev:q4", family: "flux", downloaded: true },
          { name: "wan22-ti2v-5b:q8", family: "wan", downloaded: true },
        ] as never,
        fetchedAt: 1,
        error: null,
      },
    };
    // The remote's extra style counts; the shared one is not counted twice.
    expect(subtitle.value).toBe("2 styles ready");

    // Two machines answering means the sentence has to say whose styles.
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: null };
    conn.status = "ready";
    expect(subtitle.value).toBe("2 styles ready across your machines");
  });

  it("counts the machines that answered, and names the offer on Rent a GPU", async () => {
    expect((await subtitleAt("/machines")).value).toBe("0 machines connected");
    expect((await subtitleAt("/machines/runpod")).value).toBe("Rent a GPU · billed by the minute");
  });

  it("names the machine that makes the pictures on Settings", async () => {
    const subtitle = await subtitleAt("/settings");
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: null };
    conn.status = "ready";
    expect(subtitle.value).toBe(useHostsStore().primaryHost?.label ?? "");
    expect(subtitle.value).not.toBe("");
  });
});
