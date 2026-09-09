/**
 * The two badges a 3-D run puts on a tile, mounted for real.
 *
 * Both live in the tile's bottom-left badge ROW, and that row is shared with
 * the host chip, the Upscaled mark and the media-kind word. A guard on the row
 * that named only those three swallowed the run's badges on any tile whose
 * media needs no kind word — which is every ordinary still, and so every
 * intermediate a run publishes. The badges are exactly what tells those
 * near-identical PNGs apart, so the bug hid the feature where it was needed.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";
import { defineComponent, nextTick } from "vue";

const state = vi.hoisted(() => ({ localImages: [] as unknown[] }));

vi.mock("../lib/api/client", () => ({
  apiFetch: vi.fn(),
  apiFetchTo: vi.fn().mockResolvedValue(new Response()),
  apiJsonTo: vi.fn(),
  conditionalApiJsonTo: vi.fn().mockResolvedValue([]),
  currentTarget: () => ({ baseUrl: "http://x", apiKey: null }),
}));
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    localGalleryDelete: vi.fn(),
    localGalleryList: vi.fn(async () => ({ images: state.localImages, target: null })),
    revealOutputFile: vi.fn(),
    saveOutputBytes: vi.fn(),
  },
}));

import LibraryView from "./LibraryView.vue";
import { useConnectionStore } from "../stores/connection";
import { useGalleryStore } from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import type { GalleryImage, ServerCapabilities } from "../lib/api/types";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";
import { clearSessionScrollForTests } from "@studio/lib/libraryOrganization";

installMemoryLocalStorage();

const VIEWPORT_WIDTH = 1200;
const VIEWPORT_HEIGHT = 800;

/** One print a 3-D run published, carrying the server-minted provenance. */
const runPrint = (
  filename: string,
  timestamp: number,
  role: string,
  stage: number,
  job = "run-1",
): GalleryImage =>
  ({
    filename,
    timestamp,
    // Distinct bytes AND seed per print: the merged grid also collapses
    // cross-host copies by seed + byte size within a time window, so shared
    // values silently fused two unrelated fixtures into one tile.
    size_bytes: 1_000 + timestamp,
    favorite: false,
    tags: [],
    metadata: {
      prompt: "a hand-carved wooden fox",
      model: "hunyuan3d-mini-turbo:fp16",
      seed: 4_000 + timestamp,
      // The Lightbox renders these facts unguarded; a fixture without them
      // crashes the render before any assertion about the run is reached.
      steps: 5,
      guidance: 5,
      mesh_workflow: { job_id: job, mode: "text_to_mesh", role, stage_index: stage },
    },
  }) as unknown as GalleryImage;

const stub = { template: "<div />" };
const mediaStub = defineComponent({
  name: "AuthedMedia",
  props: { path: { type: String, required: true } },
  template: "<div class='media-stub' />",
});

let restoreLayout: (() => void) | null = null;

/** happy-dom lays nothing out; give every element the viewport's box so the
 *  virtualizer and the justified layout render real tiles. */
function fakeLayout() {
  const proto = HTMLElement.prototype;
  const rect = Object.getOwnPropertyDescriptor(Element.prototype, "getBoundingClientRect");
  const saved = ["clientWidth", "clientHeight", "offsetWidth", "offsetHeight"].map(
    (name) => [name, Object.getOwnPropertyDescriptor(proto, name)] as const,
  );
  Object.defineProperty(Element.prototype, "getBoundingClientRect", {
    configurable: true,
    value: () => ({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: VIEWPORT_WIDTH,
      bottom: VIEWPORT_HEIGHT,
      width: VIEWPORT_WIDTH,
      height: VIEWPORT_HEIGHT,
      toJSON: () => ({}),
    }),
  });
  for (const name of ["clientWidth", "offsetWidth"])
    Object.defineProperty(proto, name, { configurable: true, get: () => VIEWPORT_WIDTH });
  for (const name of ["clientHeight", "offsetHeight"])
    Object.defineProperty(proto, name, { configurable: true, get: () => VIEWPORT_HEIGHT });
  restoreLayout = () => {
    if (rect) Object.defineProperty(Element.prototype, "getBoundingClientRect", rect);
    for (const [name, descriptor] of saved) {
      if (descriptor) Object.defineProperty(proto, name, descriptor);
      else delete (proto as unknown as Record<string, unknown>)[name];
    }
  };
}

async function mountGrid(items: GalleryImage[]) {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/library", component: stub },
      { path: "/create", component: stub },
      { path: "/create/3d", component: stub },
    ],
  });
  await router.push("/library");
  const pinia = createPinia();
  setActivePinia(pinia);
  const connection = useConnectionStore();
  connection.info = null;
  connection.status = "error";
  const hosts = useHostsStore();
  hosts.capabilities["local"] = {
    gallery: { can_delete: true, organize: true, trash: { enabled: true, retention_days: 30 } },
  } as unknown as ServerCapabilities;
  const gallery = useGalleryStore();
  state.localImages = items;
  gallery.buckets.local = { items, loading: false, error: null, loaded: true };
  gallery.collectionsByHost["local"] = { items: [], loaded: true } as never;

  const wrapper = mount(LibraryView, {
    attachTo: document.body,
    global: {
      plugins: [pinia, router],
      stubs: { AuthedMedia: mediaStub, HostFilterChips: stub, HistoryDrawer: stub },
    },
  });
  await flushPromises();
  await nextTick();
  return { wrapper, gallery };
}

const badgeTexts = (wrapper: { findAll: (s: string) => { text: () => string }[] }, test: string) =>
  wrapper.findAll(`[data-test="${test}"]`).map((n) => n.text().trim());

beforeEach(() => {
  clearSessionScrollForTests();
  fakeLayout();
});
afterEach(() => {
  restoreLayout?.();
  restoreLayout = null;
  document.body.innerHTML = "";
});

describe("a 3-D run's tile badges", () => {
  /*
   * The reason the role badge exists: opened, a run shows a GLB poster and two
   * PNGs of the same subject. Without the step names the two pictures are
   * indistinguishable, which is the state this shipped in.
   */
  it("names every step inside an opened run, pictures included", async () => {
    const { wrapper, gallery } = await mountGrid([
      runPrint("object.glb", 4, "final_glb", 2),
      runPrint("matted.png", 3, "matted_image", 1),
      runPrint("source.png", 2, "generated_image", 0),
    ]);
    gallery.workflowId = "run-1";
    await nextTick();
    await flushPromises();

    expect(badgeTexts(wrapper, "workflow-role-badge").sort()).toEqual([
      "Background removed",
      "Source picture",
      "The 3-D object",
    ]);
    wrapper.unmount();
  });

  /*
   * A run whose mesh has not landed — still rendering, or trashed on its own —
   * leads with its latest picture. It still collapses to one tile, so it still
   * has to say that more is inside; otherwise it is indistinguishable from an
   * ordinary print and nothing hints that it opens.
   */
  it("marks the stack even when a picture leads the run", async () => {
    const { wrapper } = await mountGrid([
      runPrint("matted.png", 3, "matted_image", 1),
      runPrint("source.png", 2, "generated_image", 0),
    ]);

    expect(badgeTexts(wrapper, "workflow-stack-badge")).toEqual(["2"]);
    wrapper.unmount();
  });

  /*
   * The chip row counts an open run as a filter, so **Clear filters** appears
   * while one is open. It cleared tags, the host chip and an open album and
   * left the run alone: with no tag filter and every host shown, the button
   * appeared and clicking it changed nothing — a control that lies, which is
   * the exact failure the tag chips in the Trash were fixed for.
   */
  it("leaves the run when the chip row's Clear filters is used", async () => {
    const { wrapper, gallery } = await mountGrid([
      runPrint("object.glb", 4, "final_glb", 2),
      runPrint("matted.png", 3, "matted_image", 1),
      runPrint("source.png", 2, "generated_image", 0),
    ]);
    gallery.openWorkflowRun("run-1");
    await nextTick();
    await flushPromises();
    expect(gallery.openWorkflowId).toBe("run-1");

    const clear = wrapper.findAll("button").find((b) => b.text().trim() === "Clear filters");
    expect(clear, "Clear filters is offered while a run is open").toBeTruthy();
    await clear!.trigger("click");
    await nextTick();

    expect(gallery.openWorkflowId).toBeNull();
    expect(wrapper.findAll(".ms-lib-tile")).toHaveLength(1);
    wrapper.unmount();
  });

  /*
   * The header's Everything count and the grid are the same promise, and the
   * defect that started this was them disagreeing: "Everything 6" beside three
   * tiles, because the view re-implemented the filter inline and never ran the
   * collapse. That fix lives in `scopeCounts` in LibraryView, which no store
   * test can reach — the pin has to be the rendered number against the
   * rendered tiles.
   */
  it("renders an Everything count that matches the tiles", async () => {
    const { wrapper } = await mountGrid([
      runPrint("object.glb", 6, "final_glb", 2),
      runPrint("matted.png", 5, "matted_image", 1),
      runPrint("source.png", 4, "generated_image", 0),
      runPrint("other.glb", 3, "final_glb", 1, "run-2"),
      runPrint("other-source.png", 2, "generated_image", 0, "run-2"),
      {
        filename: "plain.png",
        timestamp: 1,
        size_bytes: 10,
        favorite: false,
        tags: [],
        metadata: { prompt: "p", model: "flux-dev:q8", seed: 1 },
      } as unknown as GalleryImage,
    ]);

    // Six prints, two runs: two stacked tiles plus the ordinary print.
    const tiles = wrapper.findAll(".ms-lib-tile");
    expect(tiles).toHaveLength(3);
    const everything = wrapper
      .get('[data-test="library-scope"]')
      .text()
      .match(/Everything\s*(\d+)/)?.[1];
    expect(everything, "the Everything chip carries a count").toBeTruthy();
    expect(Number(everything)).toBe(tiles.length);
    wrapper.unmount();
  });

  /*
   * The layers badge is a STACK marker: it says this tile stands for prints
   * that are not drawn. Under `Pictures` the mesh lead is removed by kind, so
   * the run's pictures all render — and each one wore the run's full count,
   * claiming to hide siblings that were standing right beside it.
   */
  it("marks no stack where the run is scattered rather than collapsed", async () => {
    const { wrapper, gallery } = await mountGrid([
      runPrint("object.glb", 4, "final_glb", 2),
      runPrint("matted.png", 3, "matted_image", 1),
      runPrint("source.png", 2, "generated_image", 0),
    ]);
    // Collapsed: the mesh leads and says how many came with it.
    expect(badgeTexts(wrapper, "workflow-stack-badge")).toEqual(["3"]);

    gallery.mediaKind = "image";
    await nextTick();
    await flushPromises();

    // Scattered: all three pictures stand, and none of them hides anything.
    expect(wrapper.findAll(".ms-lib-tile")).toHaveLength(2);
    expect(badgeTexts(wrapper, "workflow-stack-badge")).toEqual([]);
    wrapper.unmount();
  });

  /*
   * The Trash is never collapsed, so nothing there hides anything and no tile
   * may claim to. The run index is LIVE-only, which made that true by accident
   * — until one machine has a print live that another has trashed, when the
   * shared filename gives the trashed row a live membership with `lead: true`
   * and the badge comes back on a grid where every row is drawn.
   */
  it("marks no stack in the Trash, where nothing is ever collapsed", async () => {
    const live = [
      runPrint("object.glb", 6, "final_glb", 2),
      runPrint("matted.png", 5, "matted_image", 1),
      runPrint("source.png", 4, "generated_image", 0),
    ];
    const { wrapper, gallery } = await mountGrid(live);
    // The same names, trashed on another machine.
    gallery.trashBuckets.local = { items: live, loading: false, error: null, loaded: true };
    gallery.scope = "trash";
    await nextTick();
    await flushPromises();

    expect(gallery.trashFiltered).toHaveLength(3);
    expect(badgeTexts(wrapper, "workflow-stack-badge")).toEqual([]);
    wrapper.unmount();
  });

  /*
   * The Trash offers no run doors at all: `tileMenu` returns Restore, Copy and
   * Delete forever and nothing else. But the Lightbox reads the LIVE run
   * index, so a print trashed here whose name is still live on another machine
   * was told it had pictures to show — and "Show the 3 pictures" leaves the
   * Trash outright, because `openWorkflowRun` moves the scope to Everything.
   */
  it("offers no run doors in the Trash, even for a name still live elsewhere", async () => {
    const live = [
      runPrint("object.glb", 6, "final_glb", 2),
      runPrint("matted.png", 5, "matted_image", 1),
      runPrint("source.png", 4, "generated_image", 0),
    ];
    const { wrapper, gallery } = await mountGrid(live);
    // The same names, trashed here while another machine still holds them live.
    gallery.trashBuckets.local = { items: live, loading: false, error: null, loaded: true };
    gallery.scope = "trash";
    await nextTick();
    await flushPromises();

    await wrapper.get('[data-filename="object.glb"]').trigger("dblclick");
    await flushPromises();
    expect(wrapper.getComponent({ name: "Lightbox" }).props("workflowAssets")).toBe(0);
    wrapper.unmount();
  });

  /* And an ordinary print still wears nothing at all. */
  it("leaves a print no run made unmarked", async () => {
    const { wrapper } = await mountGrid([
      {
        filename: "plain.png",
        timestamp: 9,
        size_bytes: 10,
        favorite: false,
        tags: [],
        metadata: { prompt: "p", model: "flux-dev:q8", seed: 1 },
      } as unknown as GalleryImage,
    ]);

    expect(badgeTexts(wrapper, "workflow-stack-badge")).toEqual([]);
    expect(badgeTexts(wrapper, "workflow-role-badge")).toEqual([]);
    wrapper.unmount();
  });
});
