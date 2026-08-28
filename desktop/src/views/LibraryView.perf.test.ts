/**
 * Library grid performance guards. Unlike the sibling suites this file does
 * NOT mock the virtualizer or the justified layout: it mounts the real grid
 * over 2 000 prints in a 1200×800 viewport and asserts operation counts
 * (tiles in the DOM, media mounts across a reflow, union passes) against
 * budgets — see `@studio/lib/galleryPerfBudget`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";
import { defineComponent, nextTick } from "vue";

const counters = vi.hoisted(() => ({
  unionOrganization: 0,
  /** What the native listing answers; seeded per test before mount. */
  localImages: [] as unknown[],
  mediaMounts: new Map<string, number>(),
  reset() {
    counters.unionOrganization = 0;
    counters.mediaMounts.clear();
  },
}));

vi.mock("@studio/lib/libraryOrganization", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@studio/lib/libraryOrganization")>();
  return {
    ...actual,
    unionOrganization: (...args: Parameters<typeof actual.unionOrganization>) => {
      counters.unionOrganization += 1;
      return actual.unionOrganization(...args);
    },
  };
});
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
    localGalleryList: vi.fn(async () => ({ images: counters.localImages, target: null })),
    revealOutputFile: vi.fn(),
    saveOutputBytes: vi.fn(),
  },
}));

import LibraryView from "./LibraryView.vue";
import { expectOpsUnder } from "@studio/lib/galleryPerfBudget";
import { useConnectionStore } from "../stores/connection";
import { useGalleryStore } from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import type { GalleryImage, ServerCapabilities } from "../lib/api/types";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";
import { clearSessionScrollForTests } from "@studio/lib/libraryOrganization";

installMemoryLocalStorage();

const PRINTS = 2_000;
const VIEWPORT_WIDTH = 1200;
const VIEWPORT_HEIGHT = 800;
/** Rows of 220 px tiles at 1200 px hold ~5–7 prints; 800 px shows ~4 rows,
 *  plus 2 rows of overscan on each side. */
const MAX_TILES_IN_DOM = 80;

function print(index: number): GalleryImage {
  return {
    filename: `print-${index}.png`,
    timestamp: 1_800_000_000 - index * 10,
    size_bytes: 100_000 + index,
    favorite: index % 5 === 0,
    tags: index % 3 === 0 ? ["portrait"] : [],
    metadata: {
      prompt: `print ${index}`,
      model: "flux-dev:q8",
      seed: index,
      steps: 4,
      guidance: 1,
      width: index % 3 === 0 ? 1536 : 1024,
      height: index % 2 === 0 ? 768 : 1024,
    },
  } as GalleryImage;
}

/** The counting media stub: one entry per path, incremented on every mount. */
const countingMedia = defineComponent({
  name: "AuthedMedia",
  props: { path: { type: String, required: true } },
  setup(props) {
    counters.mediaMounts.set(props.path, (counters.mediaMounts.get(props.path) ?? 0) + 1);
    return {};
  },
  template: "<div class='media-stub' />",
});
const stub = { template: "<div />" };

let restoreRect: (() => void) | null = null;

/** happy-dom lays nothing out; give every element the viewport's box so the
 *  virtualizer and the justified layout see a real 1200×800 scroller. */
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
  // TanStack's `getRect` reads offsetWidth/offsetHeight; the justified
  // layout reads clientWidth.
  for (const name of ["clientWidth", "offsetWidth"]) {
    Object.defineProperty(proto, name, { configurable: true, get: () => VIEWPORT_WIDTH });
  }
  for (const name of ["clientHeight", "offsetHeight"]) {
    Object.defineProperty(proto, name, { configurable: true, get: () => VIEWPORT_HEIGHT });
  }
  restoreRect = () => {
    if (rect) Object.defineProperty(Element.prototype, "getBoundingClientRect", rect);
    for (const [name, descriptor] of saved) {
      if (descriptor) Object.defineProperty(proto, name, descriptor);
      else delete (proto as unknown as Record<string, unknown>)[name];
    }
  };
}

async function mountGrid() {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/library", component: stub },
      { path: "/create", component: stub },
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
  const items: GalleryImage[] = [];
  for (let i = 0; i < PRINTS; i++) items.push(print(i));
  counters.localImages = items;
  gallery.buckets.local = { items, loading: false, error: null, loaded: true };
  gallery.collectionsByHost["local"] = { items: [], loaded: true } as never;

  const wrapper = mount(LibraryView, {
    attachTo: document.body,
    global: {
      plugins: [pinia, router],
      stubs: { AuthedMedia: countingMedia, HostFilterChips: stub, HistoryDrawer: stub },
    },
  });
  await flushPromises();
  await nextTick();
  return { wrapper, gallery };
}

beforeEach(() => {
  clearSessionScrollForTests();
  counters.reset();
  fakeLayout();
});
afterEach(() => {
  restoreRect?.();
  restoreRect = null;
  document.body.innerHTML = "";
});

describe("Library grid at 2 000 prints", () => {
  it("keeps only the viewport's rows (plus overscan) in the DOM", async () => {
    const { wrapper } = await mountGrid();
    const tiles = wrapper.findAll(".ms-lib-tile");
    expect(tiles.length).toBeGreaterThan(0);
    expectOpsUnder("tiles in the DOM", tiles.length, MAX_TILES_IN_DOM);
    // Mount sees two data changes — the seeded bucket, then the listing the
    // view fetches on open — so two index passes are legitimate.
    expectOpsUnder("unionOrganization during mount", counters.unionOrganization, 2 * PRINTS);
    wrapper.unmount();
  });

  it("re-flows on a thumbnail-size change without remounting surviving tiles", async () => {
    const { wrapper } = await mountGrid();
    const before = new Set(
      wrapper.findAll(".ms-lib-tile").map((t) => t.attributes("data-filename")),
    );
    counters.reset();

    // Three slider ticks, as a drag delivers them.
    const slider = wrapper.find('input[type="range"][aria-label="Thumbnail size"]');
    expect(slider.exists()).toBe(true);
    for (const px of [230, 240, 250]) {
      await slider.setValue(px);
      await nextTick();
    }
    await flushPromises();

    const after = new Set(
      wrapper.findAll(".ms-lib-tile").map((t) => t.attributes("data-filename")),
    );
    const survivors = [...before].filter((name) => after.has(name));
    expect(survivors.length).toBeGreaterThan(10);
    let remounted = 0;
    for (const [path, mounts] of counters.mediaMounts) {
      const name = decodeURIComponent(path.split("/").pop() ?? "");
      if (before.has(name) && mounts > 0) remounted += mounts;
    }
    expectOpsUnder("AuthedMedia remounts across a slider drag", remounted, 0);
    expectOpsUnder("unionOrganization across a slider drag", counters.unionOrganization, 0);
    wrapper.unmount();
  });
});
